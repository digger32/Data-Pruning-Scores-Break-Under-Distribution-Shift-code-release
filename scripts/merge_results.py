#!/usr/bin/env python3
"""Merge per-unit JSONs into <outdir>/tables/units.csv (one row per unit) and
<outdir>/tables/corr.csv (one row per unit x corruption x severity)."""
import json
import sys
from pathlib import Path

import pandas as pd


def main(outdir: str):
    out = Path(outdir)
    rows, crows, brows, pcrows = [], [], [], []
    for p in sorted((out / "results").glob("*.json")):
        u = json.loads(p.read_text())
        m = u["metrics"]
        rows.append({"uid": p.stem, "dataset": u["dataset"],
                     "method": u["method"], "ratio": u["ratio"],
                     "seed": u["seed"], "sseed": u.get("sseed"),
                     "clean_acc": m["clean_acc"], "mca": m["mca"],
                     "clean_ece": m.get("clean_ece"),
                     "corr_ece": m.get("corr_ece"),
                     "n_train": u["n_train"],
                     "score_sha256": u.get("score_sha256"),
                     "wall_s": u.get("wall_s"),
                     "peak_vram_mb": u.get("peak_vram_mb"),
                     "fake": u.get("fake", False)})
        base = {"uid": p.stem, "dataset": u["dataset"], "method": u["method"],
                "ratio": u["ratio"], "seed": u["seed"]}
        for cname, sevs in u["corr"].items():
            for sev, acc in sevs.items():
                crows.append({**base, "corruption": cname,
                              "severity": int(sev), "acc": acc})
        for cls, cnt in enumerate(u.get("class_hist", [])):
            brows.append({**base, "class": cls, "kept_count": cnt})
        pc = u.get("per_class")
        if pc:
            for cls, (ca, xa) in enumerate(zip(pc["clean"], pc["corr_mean"])):
                pcrows.append({**base, "class": cls, "clean_acc": ca,
                               "corr_acc": xa})
    if not rows:
        sys.exit(f"[merge] no unit JSONs under {out/'results'}")
    (out / "tables").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "tables" / "units.csv", index=False)
    pd.DataFrame(crows).to_csv(out / "tables" / "corr.csv", index=False)
    pd.DataFrame(brows).to_csv(out / "tables" / "class_balance.csv",
                               index=False)
    if pcrows:
        pd.DataFrame(pcrows).to_csv(out / "tables" / "per_class.csv",
                                    index=False)
    print(f"[merge] {len(rows)} units, {len(crows)} corruption rows, "
          f"{len(brows)} class-balance rows, {len(pcrows)} per-class rows -> "
          f"{out/'tables'}")


if __name__ == "__main__":
    main(sys.argv[1])
