"""M3: inductive neural pair model (DESIGN.md §5; config/m5m3.yaml `m3`).

    gene encoder f(x_g) -> e_g (shared by both genes; inputs = label-free gene-table columns)
    score = MLP([e_a + e_b, e_a * e_b, P])   (P = pair feature vector, symmetric in (a, b))

Two linear heads on the last hidden layer: a Huber regression head on t = -min(S2, 0) (the score) and an auxiliary
BCE head on the protocol label. No gene-ID embeddings, so unseen genes are scored exactly like seen ones.
Symmetry: e_a + e_b and e_a * e_b are exactly commutative and P is symmetric, so score(a, b) == score(b, a).

Pure functions on numpy arrays (no data access): `Preprocessor` (quantile-normal transform + missing indicators,
fitted on the fit rows only), `fit_m3`, `predict_m3`.
"""
from __future__ import annotations

import copy
import time

import numpy as np

DEFAULT_CFG = {
    "n_quantiles": 1000, "subsample": 200_000, "clip": 5.0,
    "input_dropout": 0.1, "gene_tower": [128, 64], "pair_mlp": [256, 64], "hidden_dropout": 0.2,
    "bce_weight": 1.0, "lr": 1e-3, "weight_decay": 1e-4, "batch": 2048, "max_epochs": 25, "patience": 4,
}


def resolve_cfg(cfg: dict | None) -> dict:
    return {**DEFAULT_CFG, **(cfg or {})}


# =========================================================================================== preprocessing
class Preprocessor:
    """Quantile-normal transform per column, NaN -> 0 plus a missing indicator for every column with any NaN in
    the fit rows, clipped to [-clip, clip]. Fitted on `fit_rows` of X only."""

    def __init__(self, n_quantiles=1000, subsample=200_000, clip=5.0, random_state=0):
        self.n_quantiles, self.subsample, self.clip, self.random_state = n_quantiles, subsample, clip, random_state

    def fit(self, X: np.ndarray):
        from sklearn.preprocessing import QuantileTransformer
        X = np.asarray(X, dtype=np.float32)
        nan = np.isnan(X)
        self.ind_cols = np.flatnonzero(nan.any(axis=0))
        allnan = nan.all(axis=0)
        Xf = np.where(allnan[None, :], 0.0, X)          # all-NaN columns: constant 0 (QuantileTransformer needs data)
        self.qt = QuantileTransformer(n_quantiles=min(self.n_quantiles, len(X)), output_distribution="normal",
                                      subsample=self.subsample, random_state=self.random_state, copy=True)
        self.qt.fit(Xf)
        self.allnan = allnan
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        nan = np.isnan(X)
        Xf = np.where(self.allnan[None, :], 0.0, X)
        Z = self.qt.transform(Xf).astype(np.float32)
        Z = np.nan_to_num(Z, nan=0.0)
        np.clip(Z, -self.clip, self.clip, out=Z)
        return np.hstack([Z, nan[:, self.ind_cols].astype(np.float32)])

    @property
    def n_out(self) -> int:
        return len(self.allnan) + len(self.ind_cols)


# =========================================================================================== network
def _mlp(sizes, dropout_after_first=0.0):
    import torch.nn as nn
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
            if i == 0 and dropout_after_first:
                layers.append(nn.Dropout(dropout_after_first))
    return nn.Sequential(*layers)


def build_net(d_gene: int, d_pair: int, cfg: dict):
    import torch
    import torch.nn as nn

    class M3Net(nn.Module):
        def __init__(self):
            super().__init__()
            gt = cfg["gene_tower"]
            self.in_drop = nn.Dropout(cfg["input_dropout"])
            self.tower = _mlp([d_gene] + list(gt))
            pm = cfg["pair_mlp"]
            self.trunk = nn.Sequential(_mlp([2 * gt[-1] + d_pair] + list(pm), cfg["hidden_dropout"]), nn.ReLU())
            self.head_huber = nn.Linear(pm[-1], 1)
            self.head_bce = nn.Linear(pm[-1], 1)

        def forward(self, ga, gb, p):
            ea = self.tower(self.in_drop(ga))
            eb = self.tower(self.in_drop(gb))
            h = self.trunk(torch.cat([ea + eb, ea * eb, self.in_drop(p)], dim=1))
            return self.head_huber(h).squeeze(1), self.head_bce(h).squeeze(1)

    return M3Net()


# =========================================================================================== fit / predict
def _wap(y, s, w) -> float:
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y, s, sample_weight=w))


def predict_m3(net, G: np.ndarray, ia: np.ndarray, ib: np.ndarray, P: np.ndarray, batch: int = 65536) -> np.ndarray:
    """Huber-head score for rows (ia[i], ib[i], P[i]); G = transformed gene matrix (genes x d)."""
    import torch
    net.eval()
    Gt = torch.from_numpy(np.ascontiguousarray(G, dtype=np.float32))
    out = np.empty(len(P), dtype=np.float64)
    with torch.no_grad():
        for s in range(0, len(P), batch):
            sl = slice(s, s + batch)
            pa = torch.from_numpy(np.ascontiguousarray(P[sl], dtype=np.float32))
            out[sl] = net(Gt[torch.from_numpy(ia[sl])], Gt[torch.from_numpy(ib[sl])], pa)[0].double().numpy()
    return out


def fit_m3(G: np.ndarray, ia_f, ib_f, P_f, t_f, y_f, w_f, ia_v, ib_v, P_v, y_v, w_v, cfg: dict, seed: int,
           huber_delta: float, log=None):
    """Train one M3 network (inputs already transformed). Early stopping on the class-weighted AP of y_v by the
    huber-head score; best epoch restored. Returns (net, info)."""
    import torch
    import torch.nn.functional as F
    cfg = resolve_cfg(cfg)
    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed))
    net = build_net(G.shape[1], P_f.shape[1], cfg)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    Gt = torch.from_numpy(np.ascontiguousarray(G, dtype=np.float32))
    Pt = torch.from_numpy(np.ascontiguousarray(P_f, dtype=np.float32))
    iat, ibt = torch.from_numpy(np.asarray(ia_f, np.int64)), torch.from_numpy(np.asarray(ib_f, np.int64))
    tt = torch.from_numpy(np.asarray(t_f, np.float32))
    yt = torch.from_numpy(np.asarray(y_f, np.float32))
    wt = torch.from_numpy(np.asarray(w_f, np.float32))
    n, bs = len(P_f), int(cfg["batch"])
    best, best_ep, best_state, hist = -np.inf, -1, None, []
    t0 = time.perf_counter()
    for ep in range(int(cfg["max_epochs"])):
        net.train()
        perm = torch.from_numpy(rng.permutation(n))
        tot = 0.0
        for s in range(0, n, bs):
            b = perm[s:s + bs]
            gh, gl = net(Gt[iat[b]], Gt[ibt[b]], Pt[b])
            wb = wt[b]
            lh = F.huber_loss(gh, tt[b], reduction="none", delta=huber_delta)
            lb = F.binary_cross_entropy_with_logits(gl, yt[b], reduction="none")
            loss = ((lh + cfg["bce_weight"] * lb) * wb).sum() / wb.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        sv = predict_m3(net, G, np.asarray(ia_v, np.int64), np.asarray(ib_v, np.int64), P_v)
        ap = _wap(y_v, sv, w_v)
        hist.append({"epoch": ep, "train_loss": tot / n, "valid_ap_weighted": ap,
                     "elapsed_s": round(time.perf_counter() - t0, 1)})
        if log:
            log(f"    epoch {ep}: loss {tot / n:.4f} inner-valid wAP {ap:.4f} ({hist[-1]['elapsed_s']} s)")
        if ap > best:
            best, best_ep, best_state = ap, ep, copy.deepcopy(net.state_dict())
        elif ep - best_ep >= int(cfg["patience"]):
            break
    net.load_state_dict(best_state)
    info = {"best_epoch": int(best_ep), "epochs_run": len(hist), "valid_ap_weighted": float(best),
            "n_fit": int(n), "n_valid": int(len(P_v)), "pos_fit": int(np.sum(y_f)), "pos_valid": int(np.sum(y_v)),
            "history": hist, "train_s": round(time.perf_counter() - t0, 1)}
    return net, info
