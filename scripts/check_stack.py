#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Currency checkpoint helper: detect the actual stack a run depends on, resolve
installed versions, and emit the verification worklist.

Scans the runner's Python files for third-party imports, resolves each to an
installed distribution and version, prints a STACK DETECTED block, and writes a
`SOURCES.md` skeleton with one row per package to be verified against official
documentation before the final run. Nothing is fetched here — the web check is
done manually against official documentation; this script says WHAT to check.

USAGE:
  python3 check_stack.py run_bench.py optimise.py -o SOURCES.md
  python3 check_stack.py src/ -o SOURCES.md --requirements requirements.txt
"""
import argparse
import ast
import sys
from pathlib import Path

try:
    from importlib.metadata import version, packages_distributions
except ImportError:  # py<3.10
    sys.exit("needs Python 3.10+ (importlib.metadata.packages_distributions)")

STDLIB = set(sys.stdlib_module_names)
# modules whose import name differs from the distribution name
ALIASES = {"sklearn": "scikit-learn", "cv2": "opencv-python", "PIL": "pillow",
           "yaml": "PyYAML", "fitz": "PyMuPDF"}
# stacks that drift fast and have bitten this pipeline before
HIGH_DRIFT = {"torch", "transformers", "optuna", "pymoo", "catboost", "xgboost",
              "lightgbm", "sklearn", "scikit-learn", "pandas", "scipy",
              "statsmodels", "shap", "tabpfn", "huggingface_hub", "datasets"}


def imports_of(path: Path):
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as exc:
        print(f"[stack] skipping {path}: {exc}")
        return set()
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            mods.add(node.module.split(".")[0])
    return mods


def collect(paths):
    files = []
    for p in map(Path, paths):
        files += sorted(p.rglob("*.py")) if p.is_dir() else [p]
    mods = set()
    for f in files:
        mods |= imports_of(f)
    return {m for m in mods if m not in STDLIB and not m.startswith("_")}, files


def resolve(mod):
    dist = ALIASES.get(mod, mod)
    try:
        return dist, version(dist)
    except Exception:
        pass
    try:  # import name -> distribution name
        cand = packages_distributions().get(mod, [])
        if cand:
            return cand[0], version(cand[0])
    except Exception:
        pass
    return dist, None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="runner .py files or a source directory")
    ap.add_argument("-o", "--out", default="SOURCES.md")
    ap.add_argument("--requirements", help="also cross-check against a requirements.txt")
    a = ap.parse_args()

    mods, files = collect(a.paths)
    rows = sorted((resolve(m) + (m,) for m in mods), key=lambda r: r[0].lower())

    print(f"STACK DETECTED (from {len(files)} file(s)):")
    unresolved = []
    for dist, ver, mod in rows:
        drift = " [HIGH-DRIFT]" if mod in HIGH_DRIFT or dist in HIGH_DRIFT else ""
        if ver:
            print(f"  - {dist} {ver}{drift}")
        else:
            print(f"  - {dist} (NOT INSTALLED — version unknown){drift}")
            unresolved.append(dist)

    if a.requirements and Path(a.requirements).exists():
        pinned = {l.split("==")[0].strip().lower()
                  for l in Path(a.requirements).read_text().splitlines()
                  if l.strip() and not l.startswith("#")}
        missing = {d.lower() for d, _, _ in rows} - pinned
        if missing:
            print(f"\n[stack] imported but NOT pinned in {a.requirements}: {', '.join(sorted(missing))}")

    lines = ["# SOURCES — currency verification for the final run", "",
             "One row per third-party dependency. Before the final run, verify each",
             "against OFFICIAL documentation (not blogs, not Stack Overflow, not memory)",
             "and paste the deep link plus the passage that settles a non-obvious choice.",
             "Mark anything you could not verify as UNVERIFIED rather than assuming.", "",
             "| Package | Pinned version | API used | Official doc (deep link) | Deprecated? | Checked |",
             "|---|---|---|---|---|---|"]
    for dist, ver, mod in rows:
        flag = " **(high-drift)**" if mod in HIGH_DRIFT or dist in HIGH_DRIFT else ""
        lines.append(f"| {dist}{flag} | {ver or '?'} | | | | [ ] |")
    lines += ["", "## Unverified", "(list anything with no official documentation found)", ""]
    Path(a.out).write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\n[stack] wrote verification worklist -> {a.out}")
    print("[stack] next: ask the user to authorise the currency re-check, then verify each row "
          "against official docs and pin the confirmed versions into requirements.txt.")
    if unresolved:
        print(f"[stack] WARNING: not installed here, version unknown: {', '.join(unresolved)}")


if __name__ == "__main__":
    main()
