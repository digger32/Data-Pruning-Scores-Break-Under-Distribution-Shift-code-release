#!/usr/bin/env python3
"""Figures from the merged tables (runs AFTER the gate; best-effort per
figure so a partial smoke/pilot grid still exits 0). Vector PDF output,
colourblind-safe palette, greyscale-distinguishable line styles."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PALETTE = {"random": "#000000", "el2n": "#0072B2", "grand": "#D55E00",
           "forgetting": "#009E73", "entropy": "#CC79A7",
           "clipfeat": "#E69F00", "dinov2feat": "#56B4E9", "full": "#999999"}
STYLE = {"random": "--", "full": ":"}
plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})


def curves(units, metric, fname, figdir, title):
    datasets = sorted(units["dataset"].unique())
    fig, axes = plt.subplots(1, len(datasets),
                             figsize=(4.2 * len(datasets), 3.2), squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        g = units[units["dataset"] == ds]
        for m in [m for m in PALETTE if m in set(g["method"])]:
            gm = g[g["method"] == m]
            if m == "full":
                mu = gm[metric].mean()
                ax.axhline(mu, color=PALETTE[m], ls=":", lw=1,
                           label="full data")
                continue
            agg = gm.groupby("ratio")[metric].agg(["mean", "std", "count"])
            agg = agg.sort_index()
            ci = 1.96 * agg["std"] / np.sqrt(agg["count"].clip(lower=1))
            ax.plot(agg.index, agg["mean"], STYLE.get(m, "-"),
                    color=PALETTE[m], marker="o", ms=3, lw=1.4, label=m)
            ax.fill_between(agg.index, agg["mean"] - ci, agg["mean"] + ci,
                            color=PALETTE[m], alpha=0.15, lw=0)
        ax.set_xlabel("keep ratio (%)")
        ax.set_ylabel(metric)
        ax.set_title(ds)
        ax.invert_xaxis()
        ax.grid(alpha=0.3)
    axes[0][-1].legend(fontsize=7, frameon=False)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(figdir / fname)
    plt.close(fig)
    print(f"[figs] {fname}")


def cd_diagram(corr, figdir, ratio):
    import scikit_posthocs as sp
    g = corr[(corr["ratio"] == ratio) & (corr["method"] != "full")]
    blocks = (g.groupby(["dataset", "corruption", "method"])["acc"].mean()
               .unstack("method").dropna())
    if blocks.shape[1] < 3 or len(blocks) < 3:
        print(f"[figs] cd_diagram_r{ratio}: skipped (grid too small)")
        return
    ranks = blocks.rank(axis=1, ascending=False).mean(axis=0)
    pm = sp.posthoc_nemenyi_friedman(blocks.values)
    pm.index = pm.columns = blocks.columns
    fig, ax = plt.subplots(figsize=(6.4, 2.2))
    sp.critical_difference_diagram(ranks, pm, ax=ax)
    ax.set_title(f"Mean ranks over dataset x corruption blocks, "
                 f"keep ratio {ratio}% (Nemenyi)", fontsize=9)
    fig.tight_layout()
    fig.savefig(figdir / f"cd_diagram_r{ratio}.pdf")
    plt.close(fig)
    print(f"[figs] cd_diagram_r{ratio}.pdf")


def tradeoff(units, figdir):
    pruned = units[~units["method"].isin(["full", "random"])]
    rnd = (units[units["method"] == "random"]
           .groupby(["dataset", "ratio"])[["clean_acc", "mca"]].mean())
    if pruned.empty or rnd.empty:
        print("[figs] tradeoff: skipped (need random + scored methods)")
        return
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    for (ds, m), g in pruned.groupby(["dataset", "method"]):
        agg = g.groupby("ratio")[["clean_acc", "mca"]].mean()
        common = agg.index.intersection(
            rnd.loc[ds].index if ds in rnd.index.get_level_values(0) else [])
        if len(common) == 0:
            continue
        dx = agg.loc[common, "clean_acc"] - rnd.loc[ds].loc[common, "clean_acc"]
        dy = agg.loc[common, "mca"] - rnd.loc[ds].loc[common, "mca"]
        ax.scatter(dx, dy, s=14 + 0.3 * np.asarray(common, dtype=float),
                   color=PALETTE.get(m, "#333333"),
                   marker="o" if ds.endswith("10") else "^",
                   label=f"{m}/{ds}", alpha=0.8)
    ax.axhline(0, color="k", lw=0.6)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel(r"$\Delta$ clean accuracy vs random")
    ax.set_ylabel(r"$\Delta$ corrupted accuracy (mCA) vs random")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(figdir / "tradeoff_clean_vs_robust.pdf")
    plt.close(fig)
    print("[figs] tradeoff_clean_vs_robust.pdf")


def main(outdir: str):
    out = Path(outdir)
    figdir = out / "figures"
    figdir.mkdir(exist_ok=True)
    units = pd.read_csv(out / "tables" / "units.csv")
    corr = pd.read_csv(out / "tables" / "corr.csv")
    jobs = [
        lambda: curves(units, "mca", "robustness_curves.pdf", figdir,
                       "Corrupted-test accuracy vs pruning aggressiveness"),
        lambda: curves(units, "clean_acc", "clean_curves.pdf", figdir,
                       "Clean-test accuracy vs pruning aggressiveness"),
        lambda: cd_diagram(corr, figdir, 30),
        lambda: cd_diagram(corr, figdir, 10),
        lambda: tradeoff(units, figdir),
    ]
    for j in jobs:
        try:
            j()
        except Exception as e:  # best-effort per figure
            print(f"[figs] skipped one figure: {type(e).__name__}: {e}")
    print(f"[figs] -> {figdir}")


if __name__ == "__main__":
    main(sys.argv[1])
