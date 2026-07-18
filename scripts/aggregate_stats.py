#!/usr/bin/env python3
"""Statistics over the merged tables. Emits EXACTLY the artifacts the gate
reads (stats/omnibus.json, stats/posthoc.json) plus wilcoxon_vs_random.json
and summary.csv with bootstrap CIs. Degrades gracefully on partial grids
(smoke/pilot): entries are marked skipped with a reason, files always exist.

Blocks for the omnibus/post-hoc at each (dataset, ratio): corruption types,
values = accuracy averaged over seeds and severities. Wilcoxon vs random is
paired over the same corruption blocks."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as ss


def boot_ci(v: np.ndarray, n=10000, alpha=0.05, rng=None):
    rng = rng or np.random.default_rng(0)
    if len(v) < 2:
        return float("nan"), float("nan")
    means = rng.choice(v, size=(n, len(v)), replace=True).mean(1)
    return (float(np.quantile(means, alpha / 2)),
            float(np.quantile(means, 1 - alpha / 2)))


def main(outdir: str):
    out = Path(outdir)
    corr = pd.read_csv(out / "tables" / "corr.csv")
    units = pd.read_csv(out / "tables" / "units.csv")
    stats_dir = out / "stats"
    stats_dir.mkdir(exist_ok=True)

    omnibus, posthoc, wilcoxon = {}, {}, {}
    pruned = corr[corr["method"] != "full"]
    for (ds, ratio), g in pruned.groupby(["dataset", "ratio"]):
        key = f"{ds}|r{ratio}"
        blocks = (g.groupby(["corruption", "method"])["acc"].mean()
                   .unstack("method"))
        methods = list(blocks.columns)
        if blocks.isna().any().any():
            omnibus[key] = {"skipped": "incomplete method x corruption grid"}
            posthoc[key] = {"skipped": "incomplete grid"}
        elif len(methods) < 3 or len(blocks) < 3:
            omnibus[key] = {"skipped": f"needs >=3 methods and >=3 blocks, "
                                       f"have {len(methods)}x{len(blocks)}"}
            posthoc[key] = {"skipped": "too few methods/blocks"}
        else:
            f = ss.friedmanchisquare(*[blocks[m].values for m in methods])
            omnibus[key] = {"test": "friedman", "statistic": float(f.statistic),
                            "p": float(f.pvalue), "n_blocks": len(blocks),
                            "k_methods": len(methods), "methods": methods}
            import scikit_posthocs as sp
            pm = sp.posthoc_nemenyi_friedman(blocks.values)
            posthoc[key] = {"test": "nemenyi", "methods": methods,
                            "p_matrix": np.asarray(pm).round(6).tolist(),
                            "avg_ranks": blocks.rank(axis=1, ascending=False)
                                               .mean(axis=0).round(4).to_dict()}
        if "random" in methods:
            wilcoxon[key] = {}
            for m in methods:
                if m == "random":
                    continue
                x = blocks[m].dropna()
                y = blocks["random"].reindex(x.index).dropna()
                x = x.reindex(y.index)
                if len(x) < 5:
                    wilcoxon[key][m] = {"skipped": f"only {len(x)} blocks"}
                    continue
                d = (x - y).values
                if np.allclose(d, 0):
                    wilcoxon[key][m] = {"skipped": "all differences zero"}
                    continue
                w = ss.wilcoxon(x, y)
                wilcoxon[key][m] = {"statistic": float(w.statistic),
                                    "p": float(w.pvalue),
                                    "median_delta_acc": float(np.median(d)),
                                    "n_blocks": int(len(x))}

    rows = []
    rng = np.random.default_rng(0)
    metrics = [m for m in ("clean_acc", "mca", "clean_ece", "corr_ece")
               if m in units.columns and units[m].notna().any()]
    for (ds, m, r), g in units.groupby(["dataset", "method", "ratio"]):
        for metric in metrics:
            v = g[metric].dropna().values
            lo, hi = boot_ci(v, rng=rng)
            rows.append({"dataset": ds, "method": m, "ratio": r,
                         "metric": metric, "mean": float(np.mean(v)),
                         "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                         "ci_lo": lo, "ci_hi": hi, "n_seeds": len(v)})
    pd.DataFrame(rows).to_csv(stats_dir / "summary.csv", index=False)

    (stats_dir / "omnibus.json").write_text(json.dumps(omnibus, indent=2))
    (stats_dir / "posthoc.json").write_text(json.dumps(posthoc, indent=2))
    (stats_dir / "wilcoxon_vs_random.json").write_text(
        json.dumps(wilcoxon, indent=2))
    n_tested = sum(1 for v in omnibus.values() if "p" in v)
    print(f"[stats] omnibus cells: {len(omnibus)} ({n_tested} tested), "
          f"artifacts -> {stats_dir}")


if __name__ == "__main__":
    main(sys.argv[1])
