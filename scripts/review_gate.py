#!/usr/bin/env python3
"""Review-proofing gate for D1. Reads run_meta.json + manifest.jsonl +
results/*.json in <outdir> against a gate config and exits NON-ZERO on any
failure, so it blocks freezing in pipeline.sh.

Assertions:
  A1 clean-final-run     resume DISABLED, no skips, single run_started
  F0 no-fake-data        run and units were produced with real data
  G0 grid-complete       every unit implied by run_meta axes is present & ok
  B1 external-validity   each comparative claim covered by its datasets
  S1 score-freeze        one score hash per (dataset, method, sseed); disk match
  S2 corruption-coverage every unit covers the declared suite; final config
                         additionally requires the full canonical 15 x 5 suite
  E1 stats-present       stats/omnibus.json + stats/posthoc.json exist

Usage: python review_gate.py <outdir> --config gate_config.yaml
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

CANON_CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
    "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog",
    "brightness", "contrast", "elastic_transform", "pixelate",
    "jpeg_compression",
]
CANON_SEVERITIES = [1, 2, 3, 4, 5]


def load_manifest(outdir: Path):
    mf = outdir / "manifest.jsonl"
    if not mf.exists():
        return {}
    last = {}
    for line in mf.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            last[r["unit"]] = r  # last record wins (resume reruns)
    return last


def load_units(outdir: Path):
    units = {}
    for p in sorted((outdir / "results").glob("*.json")):
        try:
            units[p.stem] = json.loads(p.read_text())
        except Exception:
            pass
    return units


def expected_uids(meta):
    ax = meta.get("axes", {})
    uids = []
    for d in ax.get("datasets", []):
        for m in ax.get("methods", []):
            for r in ax.get("ratios", []):
                for s in ax.get("seeds", []):
                    uids.append(f"{d}__{m}__r{r}__seed{s}")
        if ax.get("include_full"):
            for s in ax.get("seeds", []):
                uids.append(f"{d}__full__r100__seed{s}")
    return set(uids)


def check_A1(meta, manifest):
    if meta is None:
        return False, "run_meta.json missing — cannot verify the final pass"
    if not meta.get("no_resume", False):
        return False, "pass ran WITHOUT --no-resume (resume was enabled)"
    if any(r.get("status") == "skip" for r in manifest.values()):
        return False, "manifest shows skipped units in a no-resume pass"
    stale = [u for u, r in manifest.items()
             if r.get("started") != meta.get("run_started")]
    if stale:
        return False, (f"{len(stale)} unit(s) carry a different run_started "
                       f"(carry-over): {stale[:3]}...")
    return True, "no-resume, no skips, single run_started"


def check_F0(meta, units):
    if meta and meta.get("fake"):
        return False, "run_meta says fake=true (D1_FAKE plumbing run)"
    fakes = [u for u, j in units.items() if j.get("fake")]
    if fakes:
        return False, f"{len(fakes)} unit(s) produced with fake data: {fakes[:3]}"
    return True, "real data throughout"


def check_G0(meta, manifest, units):
    if meta is None:
        return False, "run_meta.json missing"
    exp = expected_uids(meta)
    if not exp:
        return False, "run_meta axes empty — cannot derive expected grid"
    ok = {u for u, r in manifest.items() if r.get("status") == "ok"}
    missing = sorted(exp - (ok & set(units)))
    if missing:
        return False, (f"{len(missing)}/{len(exp)} expected unit(s) missing "
                       f"or not ok: {missing[:4]}...")
    return True, f"all {len(exp)} expected units present with status ok"


def check_B1(cfg, units):
    claims = cfg.get("comparative_claims", [])
    if not claims:
        return False, "no comparative_claims declared in config — declare them"
    present = {j.get("dataset") for j in units.values()}
    fails = []
    for c in claims:
        needed = set(c.get("independent_datasets", []))
        cid = c.get("id", "?")
        if c.get("waive"):
            if not c.get("waiver_justification"):
                fails.append(f"claim '{cid}' waived without justification")
            continue
        if not needed:
            fails.append(f"claim '{cid}' lists no independent_datasets")
        elif c.get("require_all", False) and not needed <= present:
            fails.append(f"claim '{cid}' missing dataset(s) "
                         f"{sorted(needed - present)}")
        elif not needed & present:
            fails.append(f"claim '{cid}' has no run on any of {sorted(needed)}")
    if fails:
        return False, "; ".join(fails)
    return True, f"{len(claims)} claim(s) covered by present datasets"


def check_S1(cfg, units, repo_root: Path):
    groups: dict[tuple, set] = {}
    files: dict[tuple, str] = {}
    for uid, j in units.items():
        if j.get("method") in (None, "full"):
            continue
        if not j.get("score_sha256"):
            return False, f"unit {uid} recorded no score_sha256"
        key = (j["dataset"], j["method"], j.get("sseed"))
        groups.setdefault(key, set()).add(j["score_sha256"])
        files[key] = j.get("score_file")
    bad = {k: v for k, v in groups.items() if len(v) > 1}
    if bad:
        return False, (f"score hash NOT frozen within {len(bad)} group(s): "
                       f"{list(bad)[:2]} — scores changed mid-run")
    if cfg.get("verify_score_files_on_disk", True):
        import numpy as np
        for key, f in files.items():
            if not f:
                continue
            p = (repo_root / f) if not Path(f).is_absolute() else Path(f)
            if not p.exists():
                return False, f"score file missing on disk: {f}"
            arr = np.load(p)["score"].astype(np.float32)
            sha = hashlib.sha256(arr.tobytes()).hexdigest()
            if sha not in groups[key]:
                return False, (f"score file on disk differs from the one used "
                               f"by the run: {f} (post-hoc edit?)")
    return True, f"{len(groups)} score group(s), one frozen hash each"


def check_S2(cfg, meta, units):
    if meta is None:
        return False, "run_meta.json missing"
    suite = meta.get("corruption_suite", {})
    names, sevs = suite.get("corruptions", []), suite.get("severities", [])
    if cfg.get("require_full_suite", False):
        if sorted(names) != sorted(CANON_CORRUPTIONS):
            return False, (f"declared suite has {len(names)} corruptions, "
                           f"final requires the canonical 15")
        if sorted(sevs) != CANON_SEVERITIES:
            return False, f"declared severities {sevs} != {CANON_SEVERITIES}"
        if suite.get("subset"):
            return False, (f"corrupted eval subsampled to {suite['subset']} — "
                           f"final requires the full test set")
    for uid, j in units.items():
        corr = j.get("corr", {})
        for cname in names:
            if cname not in corr:
                return False, f"unit {uid} missing corruption '{cname}'"
            for sev in sevs:
                if corr[cname].get(str(sev)) is None:
                    return False, (f"unit {uid} missing severity {sev} of "
                                   f"'{cname}'")
    return True, (f"every unit covers the declared suite "
                  f"({len(names)} corruptions x {len(sevs)} severities)")


def check_E1(cfg, outdir: Path):
    art = cfg.get("stats_artifacts",
                  ["stats/omnibus.json", "stats/posthoc.json"])
    missing = [a for a in art if not (outdir / a).exists()]
    if missing:
        return False, f"missing stats artifacts: {missing}"
    return True, f"stats artifacts present: {art}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("outdir")
    ap.add_argument("--config", default="gate_config.yaml")
    ap.add_argument("--repo-root", default=".")
    a = ap.parse_args()
    outdir = Path(a.outdir)
    cfg = yaml.safe_load(Path(a.config).read_text())
    meta_p = outdir / "run_meta.json"
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else None
    manifest = load_manifest(outdir)
    units = load_units(outdir)

    checks = [
        ("A1 clean-final-run", *check_A1(meta, manifest)),
        ("F0 no-fake-data", *check_F0(meta, units)),
        ("G0 grid-complete", *check_G0(meta, manifest, units)),
        ("B1 external-validity", *check_B1(cfg, units)),
        ("S1 score-freeze", *check_S1(cfg, units, Path(a.repo_root))),
        ("S2 corruption-coverage", *check_S2(cfg, meta, units)),
    ]
    if cfg.get("require_stats", True):
        checks.append(("E1 stats-present", *check_E1(cfg, outdir)))

    print("=" * 64)
    print(f"REVIEW-PROOFING GATE | outdir={outdir} | config={a.config}")
    print("=" * 64)
    ok = True
    for name, passed, msg in checks:
        print(f"[{'PASS' if passed else 'FAIL'}] {name:24s} {msg}")
        ok = ok and passed
    print("=" * 64)
    if not ok:
        print("GATE FAILED — do not freeze these numbers into figures.")
        sys.exit(1)
    print("GATE PASSED — numbers are clean to freeze.")


if __name__ == "__main__":
    main()
