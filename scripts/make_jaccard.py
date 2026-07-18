#!/usr/bin/env python3
"""Jaccard stability of the KEPT subsets across scoring seeds, recomputed
offline from the score files (selection = deterministic top-k, so no unit
reruns needed). Emits tables/jaccard.csv and a stability figure. Random is
included as the analytic baseline (E[J] = r/(2-r) for keep-ratio r); FM
scores are deterministic (sNA) and are skipped. CPU, seconds.

Usage:
  python scripts/make_jaccard.py --config configs/d1.yaml \
      --scores-dir scores_v2 --outdir runs/final_v2_XXX
"""
import argparse
import itertools
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def kept_set(score: np.ndarray, ratio: int) -> set:
    k = int(round(len(score) * ratio / 100))
    return set(np.argsort(-score, kind="stable")[:k].tolist())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--scores-dir", required=True)
    ap.add_argument("--outdir", required=True,
                    help="run dir; writes tables/jaccard.csv + figures/")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    sdir = Path(a.scores_dir)
    out = Path(a.outdir)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    rows = []
    for d in cfg["datasets"]:
        for m in cfg["methods"]:
            if m in cfg.get("fm_models", {}):
                continue  # deterministic, J=1 by construction
            files = sorted(sdir.glob(f"{d}__{m}__s[0-9]*.npz"),
                           key=lambda p: int(re.search(r"__s(\d+)", p.stem)[1]))
            if len(files) < 2:
                continue
            scores = [np.load(f)["score"] for f in files]
            for r in cfg["keep_ratios"]:
                sets = [kept_set(s, r) for s in scores]
                js = [len(x & y) / len(x | y)
                      for x, y in itertools.combinations(sets, 2)]
                rows.append({"dataset": d, "method": m, "ratio": r,
                             "n_sseeds": len(sets), "n_pairs": len(js),
                             "jaccard_mean": float(np.mean(js)),
                             "jaccard_std": float(np.std(js)),
                             "jaccard_min": float(np.min(js))})
    if not rows:
        raise SystemExit("[jaccard] no multi-sseed score files found — "
                         "was scoring run in per_seed mode?")
    df = pd.DataFrame(rows)
    df.to_csv(out / "tables" / "jaccard.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})
    PALETTE = {"random": "#000000", "el2n": "#0072B2", "grand": "#D55E00",
               "forgetting": "#009E73", "entropy": "#CC79A7"}
    datasets = sorted(df["dataset"].unique())
    fig, axes = plt.subplots(1, len(datasets),
                             figsize=(4.2 * len(datasets), 3.2), squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        g = df[df["dataset"] == ds]
        for m in sorted(g["method"].unique()):
            gm = g[g["method"] == m].sort_values("ratio")
            ax.errorbar(gm["ratio"], gm["jaccard_mean"], yerr=gm["jaccard_std"],
                        marker="o", ms=3, lw=1.4, capsize=2,
                        ls="--" if m == "random" else "-",
                        color=PALETTE.get(m, "#666666"), label=m)
        rr = np.array(sorted(g["ratio"].unique())) / 100
        ax.plot(rr * 100, rr / (2 - rr), ":", color="#999999", lw=1,
                label="random (analytic)")
        ax.set_xlabel("keep ratio (%)")
        ax.set_ylabel("Jaccard of kept subsets across scoring seeds")
        ax.set_title(ds)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
    axes[0][-1].legend(fontsize=7, frameon=False)
    fig.suptitle("Selection stability across scoring seeds", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "figures" / "jaccard_stability.pdf")
    print(f"[jaccard] {len(df)} rows -> {out/'tables'/'jaccard.csv'} + "
          f"figures/jaccard_stability.pdf")


if __name__ == "__main__":
    main()
