#!/usr/bin/env python
"""Compute mean-pooled ESM-2 protein embeddings for S. pombe or S. cerevisiae.

Usage (from the project root):
    .venv/bin/python -I scripts/compute_esm_embeddings.py --species pombe --model 650M
    .venv/bin/python -I scripts/compute_esm_embeddings.py --species cerevisiae --model 650M
    .venv/bin/python -I scripts/compute_esm_embeddings.py --species pombe --model 650M --benchmark 20

Inputs
    pombe      data/raw/pombase_protein/peptide.fa.gz   (PomBase peptide FASTA; ">SPAC1002.01.1:pep name|product")
               ID -> gene systematic ID: drop the ":pep" tag and the trailing transcript suffix ".<n>"
               (SPAC1002.01.1:pep -> SPAC1002.01). If a gene had >1 transcript the longest protein is kept
               (not the case in the current release: 5126 proteins = 5126 genes).
    cerevisiae data/raw/sgd_protein/orf_trans_all.fasta.gz (SGD ORF translations; ID = first header token,
               the systematic ORF name e.g. YAL001C; includes Dubious/Uncharacterized ORFs).
    weights    data/raw/esm2_weights/esm2_t33_650M_UR50D.pt (or t30_150M) -- downloaded from
               https://dl.fbaipublicfiles.com/fair-esm/models/ ; see DATA.md.
Outputs
    data/processed/esm2_<model>_<species>.parquet   index = gene systematic ID, float32 columns e0..e{d-1}
    results/runtime/esm_<species>.json              run log (model, n proteins, wall-clock, device, timestamp, ...)

Method
    * Final-layer (layer = n_layers) residue representations, BOS/EOS/pad excluded, mean over residues.
    * Terminal '*' stripped; any letter outside the 20 standard amino acids (X, U, B, Z, O, J ...) -> 'X',
      the ESM unknown-residue token (counts reported in the log).
    * Proteins > 1022 aa: overlapping windows of 1022 with 256 overlap (stride 766; last window right-aligned).
      Residue embeddings of overlapping positions are averaged across windows, then mean-pooled over the
      protein, so each residue counts exactly once.
    * Batches are formed from sequences sorted by length (longest first) under a token budget.
    * fp32, model.eval(), torch.no_grad, fixed seeds -> deterministic inference. If a batch fails on MPS it is
      retried one sequence at a time on MPS, then on CPU.
"""
import argparse
import gzip
import json
import os
import pickle
import platform
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
MODEL_FILES = {
    "650M": ("esm2_t33_650M_UR50D", 33),
    "150M": ("esm2_t30_150M_UR50D", 30),
}
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
MAX_LEN = 1022
OVERLAP = 256
SEED = 0


# ----------------------------------------------------------------------------- FASTA

def read_fasta(path):
    """Yield (header, sequence) from a (gzipped) FASTA file."""
    opener = gzip.open if str(path).endswith(".gz") else open
    header, parts = None, []
    with opener(path, "rt") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(parts)
                header, parts = line[1:], []
            elif line:
                parts.append(line)
    if header is not None:
        yield header, "".join(parts)


def clean_sequence(seq):
    """Strip '*' and map non-standard letters to 'X'. Returns (seq, n_replaced)."""
    seq = seq.upper().replace("*", "").replace("-", "").replace(" ", "")
    n_bad = sum(1 for c in seq if c not in STANDARD_AA)
    if n_bad:
        seq = "".join(c if c in STANDARD_AA else "X" for c in seq)
    return seq, n_bad


def load_proteins(species):
    """Return dict gene_id -> cleaned sequence and a stats dict."""
    seqs, stats = {}, {"n_records": 0, "n_nonstd_residues": 0, "n_proteins_with_nonstd": 0, "n_empty_skipped": 0}
    if species == "pombe":
        path = ROOT / "data/raw/pombase_protein/peptide.fa.gz"
    elif species == "cerevisiae":
        path = ROOT / "data/raw/sgd_protein/orf_trans_all.fasta.gz"
    else:
        raise ValueError(species)
    for header, seq in read_fasta(path):
        stats["n_records"] += 1
        tok = header.split()[0]
        if species == "pombe":
            gid = re.sub(r":pep$", "", tok)
            gid = re.sub(r"\.\d+$", "", gid)  # transcript suffix .1 -> gene systematic ID
        else:
            gid = tok
        seq, nbad = clean_sequence(seq)
        if not seq:
            stats["n_empty_skipped"] += 1
            continue
        stats["n_nonstd_residues"] += nbad
        stats["n_proteins_with_nonstd"] += int(nbad > 0)
        if gid in seqs and len(seqs[gid]) >= len(seq):
            continue  # keep the longest isoform
        seqs[gid] = seq
    return seqs, stats


# ----------------------------------------------------------------------------- windows / batching

def window_starts(n, max_len=MAX_LEN, overlap=OVERLAP):
    if n <= max_len:
        return [0]
    stride = max_len - overlap
    starts = list(range(0, n - max_len + 1, stride))
    if starts[-1] + max_len < n:
        starts.append(n - max_len)  # right-aligned last window
    return starts


def make_items(seqs):
    """Return list of (gene_id, start, subseq) windows."""
    items = []
    for gid, s in seqs.items():
        for st in window_starts(len(s)):
            items.append((gid, st, s[st:st + MAX_LEN]))
    return items


def make_batches(items, max_tokens):
    """Indices into items grouped by descending length under a padded-token budget."""
    order = sorted(range(len(items)), key=lambda i: -len(items[i][2]))
    batches, cur, cur_max = [], [], 0
    for i in order:
        L = len(items[i][2]) + 2
        new_max = max(cur_max, L)
        if cur and new_max * (len(cur) + 1) > max_tokens:
            batches.append(cur)
            cur, cur_max = [], 0
            new_max = L
        cur.append(i)
        cur_max = new_max
    if cur:
        batches.append(cur)
    return batches


# ----------------------------------------------------------------------------- model

def set_seeds(seed=SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_model(model_tag, device):
    import esm
    name, nlayers = MODEL_FILES[model_tag]
    wpath = ROOT / "data/raw/esm2_weights" / f"{name}.pt"
    if not wpath.exists():
        raise FileNotFoundError(f"{wpath} missing; download from https://dl.fbaipublicfiles.com/fair-esm/models/{name}.pt")
    import argparse as _ap
    # checkpoint stores an argparse.Namespace; allow-list only that class (keeps weights_only=True safety)
    with torch.serialization.safe_globals([_ap.Namespace]):
        model, alphabet = esm.pretrained.load_model_and_alphabet_local(str(wpath))
    model.eval().to(device)
    return model, alphabet, nlayers


@torch.no_grad()
def embed_batch(model, alphabet, nlayers, subseqs, device):
    """Residue embeddings for a batch: list of (L_i, d) float32 numpy arrays (no BOS/EOS)."""
    bc = alphabet.get_batch_converter()
    _, _, toks = bc([(str(i), s) for i, s in enumerate(subseqs)])
    toks = toks.to(device)
    out = model(toks, repr_layers=[nlayers], return_contacts=False)["representations"][nlayers]
    out = out.float().cpu().numpy()
    return [out[i, 1:1 + len(s)] for i, s in enumerate(subseqs)]


class Embedder:
    def __init__(self, model_tag, device):
        self.model_tag = model_tag
        self.device = device
        set_seeds()
        self.model, self.alphabet, self.nlayers = load_model(model_tag, device)
        self._cpu = None
        self.n_fallback_cpu = 0
        self.n_fallback_single = 0
        self.fallback_ids = []

    def _cpu_model(self):
        if self._cpu is None:
            self._cpu = load_model(self.model_tag, "cpu")[0]
        return self._cpu

    def run(self, subseqs, ids=None):
        try:
            return embed_batch(self.model, self.alphabet, self.nlayers, subseqs, self.device)
        except Exception as e:  # noqa: BLE001  (MPS OOM / unsupported op)
            if self.device == "cpu":
                raise
            print(f"  [warn] batch of {len(subseqs)} failed on {self.device} ({type(e).__name__}); retrying singly", flush=True)
        res = []
        for k, s in enumerate(subseqs):
            try:
                res.extend(embed_batch(self.model, self.alphabet, self.nlayers, [s], self.device))
                self.n_fallback_single += 1
            except Exception as e:  # noqa: BLE001
                print(f"  [warn] seq len {len(s)} failed on {self.device} ({type(e).__name__}); using CPU", flush=True)
                self.n_fallback_cpu += 1
                if ids is not None:
                    self.fallback_ids.append(ids[k])
                res.extend(embed_batch(self._cpu_model(), self.alphabet, self.nlayers, [s], "cpu"))
        return res

    def embed_proteins(self, seqs, max_tokens=4096, log_every=60, checkpoint=None, quiet=False):
        """seqs: dict id -> sequence. Returns dict id -> float32 (d,) mean-pooled embedding."""
        items = make_items(seqs)
        batches = make_batches(items, max_tokens)
        acc, cnt = {}, {}  # per-protein per-residue sums and window counts (windowed proteins only)
        done = {}
        if checkpoint and Path(checkpoint).exists():
            with open(checkpoint, "rb") as fh:
                ck = pickle.load(fh)
            if ck["model"] == self.model_tag and ck["n_items"] == len(items):
                done_batches = ck["done_batches"]
                acc, cnt, done = ck["acc"], ck["cnt"], ck["done"]
                print(f"resuming from checkpoint: {done_batches}/{len(batches)} batches", flush=True)
            else:
                done_batches = 0
        else:
            done_batches = 0
        t0, last = time.time(), time.time()
        for bi in range(done_batches, len(batches)):
            idx = batches[bi]
            subs = [items[i][2] for i in idx]
            gids = [items[i][0] for i in idx]
            embs = self.run(subs, gids)
            for i, e in zip(idx, embs):
                gid, st, sub = items[i]
                n = len(seqs[gid])
                if n <= MAX_LEN:
                    done[gid] = e.mean(0).astype(np.float32)
                else:
                    if gid not in acc:
                        acc[gid] = np.zeros((n, e.shape[1]), dtype=np.float64)
                        cnt[gid] = np.zeros(n, dtype=np.int32)
                    acc[gid][st:st + len(sub)] += e
                    cnt[gid][st:st + len(sub)] += 1
            if not quiet and time.time() - last > log_every:
                last = time.time()
                el = last - t0
                print(f"  batch {bi + 1}/{len(batches)}  elapsed {el / 60:.1f} min", flush=True)
            if checkpoint and (bi % 50 == 49):
                with open(checkpoint, "wb") as fh:
                    pickle.dump({"model": self.model_tag, "n_items": len(items), "done_batches": bi + 1,
                                 "acc": acc, "cnt": cnt, "done": done}, fh)
        for gid, a in acc.items():
            c = cnt[gid]
            assert (c > 0).all(), f"uncovered residues in {gid}"
            done[gid] = (a / c[:, None]).mean(0).astype(np.float32)  # overlap-averaged, each residue once
        return done


# ----------------------------------------------------------------------------- main

def pick_device(arg):
    if arg != "auto":
        return arg
    return "mps" if torch.backends.mps.is_available() else "cpu"


def benchmark(args, seqs):
    rng = np.random.default_rng(SEED)
    ids = list(seqs)
    pick = rng.choice(len(ids), size=min(args.benchmark, len(ids)), replace=False)
    sub = {ids[i]: seqs[ids[i]] for i in pick}
    res = {}
    for dev in ["mps", "cpu"]:
        if dev == "mps" and not torch.backends.mps.is_available():
            continue
        emb = Embedder(args.model, dev)
        emb.embed_proteins({k: v[:50] for k, v in list(sub.items())[:2]}, quiet=True)  # warm-up
        if dev == "mps":
            torch.mps.synchronize()
        t = time.time()
        emb.embed_proteins(sub, max_tokens=args.max_tokens, quiet=True)
        if dev == "mps":
            torch.mps.synchronize()
        dt = time.time() - t
        nres = sum(len(s) for s in sub.values())
        res[dev] = {"seconds": dt, "residues": nres, "residues_per_s": nres / dt}
        print(f"benchmark {args.model} {dev}: {len(sub)} proteins, {nres} aa, {dt:.1f}s, {nres / dt:.0f} aa/s", flush=True)
        del emb
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", required=True, choices=["pombe", "cerevisiae"])
    ap.add_argument("--model", default="650M", choices=list(MODEL_FILES))
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])
    ap.add_argument("--max-tokens", type=int, default=4096, help="padded-token budget per batch")
    ap.add_argument("--benchmark", type=int, default=0, help="benchmark N random proteins on MPS vs CPU and exit")
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N proteins (not written to the standard output)")
    ap.add_argument("--no-checkpoint", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(os.cpu_count() or 10)
    set_seeds()
    seqs, stats = load_proteins(args.species)
    print(f"{args.species}: {len(seqs)} proteins, {sum(map(len, seqs.values()))} aa, "
          f"{sum(len(s) > MAX_LEN for s in seqs.values())} > {MAX_LEN} aa, stats={stats}", flush=True)
    if args.benchmark:
        res = benchmark(args, seqs)
        total_aa = sum(map(len, seqs.values()))
        for dev, r in res.items():
            print(f"  est. full {args.species} on {dev}: {total_aa / r['residues_per_s'] / 3600:.2f} h")
        return
    if args.limit:
        seqs = dict(list(seqs.items())[:args.limit])

    device = pick_device(args.device)
    t0 = time.time()
    emb = Embedder(args.model, device)
    ck = None
    if not args.no_checkpoint and not args.limit:
        (ROOT / "data/cache").mkdir(parents=True, exist_ok=True)
        ck = ROOT / f"data/cache/esm_ckpt_{args.model}_{args.species}.pkl"
    out = emb.embed_proteins(seqs, max_tokens=args.max_tokens, checkpoint=ck)
    wall = time.time() - t0

    ids = [g for g in seqs]  # FASTA order
    mat = np.stack([out[g] for g in ids]).astype(np.float32)
    df = pd.DataFrame(mat, index=pd.Index(ids, name="gene_id"), columns=[f"e{i}" for i in range(mat.shape[1])])
    assert not df.isna().any().any()
    suffix = "_debug" if args.limit else ""
    outp = ROOT / f"data/processed/esm2_{args.model}_{args.species}{suffix}.parquet"
    outp.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(outp)

    log = {
        "species": args.species,
        "model": MODEL_FILES[args.model][0],
        "model_tag": args.model,
        "n_proteins": len(ids),
        "n_residues": int(sum(len(s) for s in seqs.values())),
        "n_windowed_proteins": int(sum(len(s) > MAX_LEN for s in seqs.values())),
        "embedding_dim": int(mat.shape[1]),
        "wall_clock_s": round(wall, 1),
        "device": device,
        "max_tokens_per_batch": args.max_tokens,
        "n_singleton_retries_after_batch_failure": emb.n_fallback_single,
        "n_cpu_fallback_sequences": emb.n_fallback_cpu,
        "cpu_fallback_windows_gene_ids": sorted(set(emb.fallback_ids)),
        "fasta_stats": stats,
        "seed": SEED,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "output": str(outp.relative_to(ROOT)),
    }
    try:
        import esm
        log["fair_esm"] = getattr(esm, "__version__", "2.0.0")
    except Exception:  # noqa: BLE001
        pass
    if not args.limit:
        logp = ROOT / f"results/runtime/esm_{args.species}.json"
        logp.parent.mkdir(parents=True, exist_ok=True)
        logp.write_text(json.dumps(log, indent=2))
        if ck and Path(ck).exists():
            Path(ck).unlink()
    print(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
