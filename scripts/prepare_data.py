#!/usr/bin/env python3
"""Download CIFAR-10/100 (torchvision) and the OFFICIAL CIFAR-10-C /
CIFAR-100-C from Zenodo (the standard, citable corruption sets; ~2.9GB each).
Idempotent; records sha256 of every downloaded tar into data/checksums.json
as self-recorded provenance. Run once on the box, before any stage:

    tmux new -s d1_data -d 'python scripts/prepare_data.py --data-dir data'
"""
import argparse
import hashlib
import json
import tarfile
import urllib.request
from pathlib import Path

URLS = {
    "CIFAR-10-C": "https://zenodo.org/records/2535967/files/CIFAR-10-C.tar?download=1",
    "CIFAR-100-C": "https://zenodo.org/records/3555552/files/CIFAR-100-C.tar?download=1",
}
EXPECT_SHAPE = (50000, 32, 32, 3)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data")
    a = ap.parse_args()
    d = Path(a.data_dir)
    d.mkdir(exist_ok=True)

    import torchvision
    torchvision.datasets.CIFAR10(root=str(d), train=True, download=True)
    torchvision.datasets.CIFAR10(root=str(d), train=False, download=True)
    torchvision.datasets.CIFAR100(root=str(d), train=True, download=True)
    torchvision.datasets.CIFAR100(root=str(d), train=False, download=True)
    print("[data] CIFAR-10/100 ready")

    checks_p = d / "checksums.json"
    checks = json.loads(checks_p.read_text()) if checks_p.exists() else {}
    import numpy as np
    for name, url in URLS.items():
        target = d / name
        if (target / "labels.npy").exists():
            print(f"[data] {name} already extracted")
        else:
            tar = d / f"{name}.tar"
            if not tar.exists():
                print(f"[data] downloading {name} ...")
                urllib.request.urlretrieve(url, tar)
            checks[f"{name}.tar"] = sha256_file(tar)
            checks_p.write_text(json.dumps(checks, indent=2))
            print(f"[data] {name}.tar sha256={checks[f'{name}.tar'][:16]}...")
            with tarfile.open(tar) as tf:
                tf.extractall(d, filter="data")
            tar.unlink()
        arr = np.load(target / "gaussian_noise.npy", mmap_mode="r")
        assert arr.shape == EXPECT_SHAPE, f"{name} bad shape {arr.shape}"
        assert len(np.load(target / "labels.npy")) == 50000
        print(f"[data] {name} verified: shape {EXPECT_SHAPE}")
    print("[data] all datasets ready; checksums in", checks_p)


if __name__ == "__main__":
    main()
