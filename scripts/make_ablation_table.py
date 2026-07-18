#!/usr/bin/env python3
"""Scoring-choice ablation table at keep-50%, built from ONE self-contained
run (variants + canonical comparators + random, all at the same fixed scoring
draw). For each method: mean/std over seeds of clean acc, mCA, clean/corrupted
ECE, the delta vs the in-run random baseline, and the paired Wilcoxon against
random over the 15 corruption blocks (read from stats/wilcoxon_vs_random.json,
which aggregate_stats.py fills because random is inside the run).

Output: <ablation-dir>/tables/ablation.csv

Usage:
  python scripts/make_ablation_table.py --ablation-dir runs/ablation
"""
import argparse
import json
from pathlib import Path

import pandas as pd

FAMILY = {  # method -> (factor family, role, setting)
    "random": ("baseline", "baseline", "-"),
    "el2n": ("EL2N scoring epoch", "canonical", "epoch 10"),
    "el2n_e5": ("EL2N scoring epoch", "variant", "epoch 5"),
    "el2n_e20": ("EL2N scoring epoch", "variant", "epoch 20"),
    "grand": ("GraNd scoring epoch", "canonical", "epoch 2"),
    "grand_e1": ("GraNd scoring epoch", "variant", "epoch 1"),
    "grand_e5": ("GraNd scoring epoch", "variant", "epoch 5"),
    "clipfeat": ("CLIP backbone", "canonical", "ViT-B/32"),
    "clipfeat_b16": ("CLIP backbone", "variant", "ViT-B/16"),
    "dinov2feat": ("DINOv2 backbone", "canonical", "ViT-S/14"),
    "dinov2feat_b14": ("DINOv2 backbone", "variant", "ViT-B/14"),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ablation-dir", required=True)
    ap.add_argument("--ratio", type=int, default=50)
    a = ap.parse_args()
    abl = Path(a.ablation_dir)
    units = pd.read_csv(abl / "tables" / "units.csv")
    units = units[units["ratio"] == a.ratio]
    if "random" not in set(units["method"]):
        raise SystemExit("[ablation] no in-run random baseline at "
                         f"ratio {a.ratio} — the table needs it; check the "
                         "ablation config methods list")
    wpath = abl / "stats" / "wilcoxon_vs_random.json"
    wil = json.loads(wpath.read_text()) if wpath.exists() else {}

    metrics = [m for m in ("clean_acc", "mca", "clean_ece", "corr_ece")
               if m in units.columns and units[m].notna().any()]
    rows = []
    for ds, g in units.groupby("dataset"):
        base = g[g["method"] == "random"]
        wkey = f"{ds}|r{a.ratio}"
        for m, gm in g.groupby("method"):
            fam, role, setting = FAMILY.get(m, ("other", "other", "-"))
            row = {"dataset": ds, "family": fam, "method": m, "role": role,
                   "setting": setting, "n_seeds": len(gm)}
            for col in metrics:
                row[f"{col}_mean"] = float(gm[col].mean())
                row[f"{col}_std"] = (float(gm[col].std(ddof=1))
                                     if len(gm) > 1 else 0.0)
                if m != "random":
                    row[f"d_{col}_pp"] = round(
                        (gm[col].mean() - base[col].mean()) * 100, 3)
            w = wil.get(wkey, {}).get(m)
            if w and "p" in w:
                row["wilcoxon_p_vs_random"] = w["p"]
                row["median_d_acc_vs_random"] = w["median_delta_acc"]
            rows.append(row)
    df = pd.DataFrame(rows).sort_values(["dataset", "family", "role", "method"])
    out = abl / "tables" / "ablation.csv"
    df.to_csv(out, index=False)
    cols = ["dataset", "family", "method", "role", "setting", "mca_mean",
            "d_mca_pp", "wilcoxon_p_vs_random", "clean_acc_mean",
            "d_clean_acc_pp"]
    with pd.option_context("display.width", 200):
        print(df[[c for c in cols if c in df.columns]].round(4)
                .to_string(index=False))
    print(f"[ablation] -> {out}")


if __name__ == "__main__":
    main()
