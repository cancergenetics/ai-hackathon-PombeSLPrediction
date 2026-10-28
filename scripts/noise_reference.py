#!/usr/bin/env python
"""Noise reference (DESIGN.md §3; not a ceiling). Dev genes only; aggregate output only.

Among reciprocal dev pairs (measured with both genes as query), predict orientation B's
single-orientation label (raw_B < t1) from orientation A's raw score (score = -raw_A), where
t1 = the raw-score equivalent of the label threshold for SINGLE-orientation pairs, obtained by inverting
the empirical monotone map raw -> S2 for one-orientation pairs (data/processed/ryan_s2_transform.tsv;
~ -3.76 for S2 = -2.3). Both directions (A = a-as-query, A = b-as-query) are reported, plus their mean.
Output: results/noise_reference.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import protocol  # noqa: E402


def main() -> int:
    thr = protocol.threshold()
    tr = pd.read_csv(protocol.data_root() / "ryan_s2_transform.tsv", sep="\t")
    one = tr[tr.orientations == 1]
    t1 = float(np.interp(thr, one.s_score.to_numpy(), one.s_raw_mean.to_numpy()))
    dev = pd.read_parquet(protocol.splits_dir() / "labels_dev.parquet",
                          columns=["s_raw_ab", "s_raw_ba", "both_orientations", "eval_eligible"])
    out = {"label_threshold": thr, "single_orientation_raw_threshold": round(t1, 4),
           "source": "labels_dev (dev genes only, lockbox-C1 pairs excluded)"}
    for name, sub in (("all_dev_reciprocal", dev[dev.both_orientations]),
                      ("eval_eligible_reciprocal", dev[dev.both_orientations & dev.eval_eligible])):
        r = {"n_pairs": int(len(sub))}
        aps = []
        for d, (src, tgt) in {"a_predicts_b": ("s_raw_ab", "s_raw_ba"), "b_predicts_a": ("s_raw_ba", "s_raw_ab")}.items():
            y = sub[tgt].to_numpy() < t1
            ap = float(average_precision_score(y, -sub[src].to_numpy()))
            aps.append(ap)
            r[d] = {"ap": round(ap, 4), "prevalence": round(float(y.mean()), 5), "lift": round(ap / y.mean(), 2)}
        r["mean_ap"] = round(float(np.mean(aps)), 4)
        r["pearson_r"] = round(float(np.corrcoef(sub.s_raw_ab, sub.s_raw_ba)[0, 1]), 4)
        out[name] = r
    dest = protocol.results_root() / "noise_reference.json"
    dest.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
