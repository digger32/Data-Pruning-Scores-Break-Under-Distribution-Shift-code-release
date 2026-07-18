#!/usr/bin/env python3
"""D1 per-unit ML code: datasets, model, training loop, clean+corrupted eval,
probe training for loss-geometry scores, FM-feature scoring, and the training
unit itself. Heavy imports live here so the orchestrators stay light.

Determinism: every entry point seeds python/numpy/torch and uses a seeded
DataLoader generator + worker_init_fn. cudnn.benchmark is enabled for speed;
run-to-run identity is provided by seeds, not by deterministic kernels
(recorded in run_meta and stated in the paper's reproducibility note).

D1_FAKE=1 (env) swaps real data and pretrained FM weights for small synthetic
stand-ins so the full pipeline can be plumbing-tested on a CPU-only box. Never
use fake mode for real numbers; run_meta records the flag and the gate refuses
fake runs.
"""
import json
import math
import os
import random
from pathlib import Path

import numpy as np

FAKE = os.environ.get("D1_FAKE", "0") == "1"

CANON_CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
    "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog",
    "brightness", "contrast", "elastic_transform", "pixelate",
    "jpeg_compression",
]
CANON_SEVERITIES = [1, 2, 3, 4, 5]

STATS = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
}
N_CLASSES = {"cifar10": 10, "cifar100": 100}
C_DIR = {"cifar10": "CIFAR-10-C", "cifar100": "CIFAR-100-C"}


def set_seed(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def device_str() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- #
# Data                                                                         #
# --------------------------------------------------------------------------- #
def _fake_split(dataset: str, n: int, seed: int):
    rng = np.random.default_rng(abs(hash((dataset, "fake", seed))) % 2**32)
    x = rng.integers(0, 256, size=(n, 32, 32, 3), dtype=np.uint8)
    y = rng.integers(0, N_CLASSES[dataset], size=n, dtype=np.int64)
    return x, y


def load_dataset(dataset: str, data_dir: str, train_subset: int | None = None,
                 test_subset: int | None = None) -> dict:
    """Return {'xtr','ytr','xte','yte'} as uint8 NHWC / int64 arrays.
    Optional subsets are taken deterministically (fixed rng(0) permutation)."""
    if FAKE:
        xtr, ytr = _fake_split(dataset, train_subset or 512, 0)
        xte, yte = _fake_split(dataset, test_subset or 256, 1)
        return {"xtr": xtr, "ytr": ytr, "xte": xte, "yte": yte}
    import torchvision
    cls = {"cifar10": torchvision.datasets.CIFAR10,
           "cifar100": torchvision.datasets.CIFAR100}[dataset]
    tr = cls(root=data_dir, train=True, download=True)
    te = cls(root=data_dir, train=False, download=True)
    xtr, ytr = tr.data, np.asarray(tr.targets, dtype=np.int64)
    xte, yte = te.data, np.asarray(te.targets, dtype=np.int64)
    if train_subset:
        idx = np.random.default_rng(0).permutation(len(ytr))[:train_subset]
        xtr, ytr = xtr[idx], ytr[idx]
    if test_subset:
        xte, yte = xte[:test_subset], yte[:test_subset]
    return {"xtr": xtr, "ytr": ytr, "xte": xte, "yte": yte}


def load_corruption(dataset: str, name: str, data_dir: str,
                    n_test: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (x[5, n, 32, 32, 3] uint8, labels[n]) for one corruption type.
    Real mode reads the official CIFAR-10-C / CIFAR-100-C .npy files
    (10000 test images x severities 1..5 stacked along axis 0)."""
    if FAKE:
        rng = np.random.default_rng(abs(hash((dataset, name, "corr"))) % 2**32)
        x = rng.integers(0, 256, size=(5, n_test, 32, 32, 3), dtype=np.uint8)
        _, yte = _fake_split(dataset, n_test, 1)
        return x, yte
    d = Path(data_dir) / C_DIR[dataset]
    arr = np.load(d / f"{name}.npy", mmap_mode="r")
    labels = np.load(d / "labels.npy")[:10000].astype(np.int64)
    n = min(n_test, 10000)
    x = np.stack([arr[s * 10000:s * 10000 + n] for s in range(5)])
    return x, labels[:n]


class ArrayDataset:
    """uint8 NHWC arrays -> normalized CHW float tensors, optional CIFAR aug."""

    def __init__(self, x: np.ndarray, y: np.ndarray, dataset: str, aug: bool):
        import torch
        from torchvision.transforms import v2
        self.x, self.y = x, y
        mean, std = STATS[dataset]
        ops = []
        if aug:
            ops += [v2.RandomCrop(32, padding=4), v2.RandomHorizontalFlip()]
        ops += [v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(list(mean), list(std))]
        self.tfm = v2.Compose(ops)
        self._torch = torch

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        t = self._torch.from_numpy(
            np.ascontiguousarray(self.x[i].transpose(2, 0, 1)))
        return self.tfm(t), int(self.y[i])


def make_loader(ds, batch: int, shuffle: bool, seed: int, workers: int):
    import torch
    from torch.utils.data import DataLoader

    def winit(wid):
        s = (seed * 1000 + wid) % (2**32)
        np.random.seed(s)
        random.seed(s)

    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, generator=g,
                      worker_init_fn=winit, num_workers=workers,
                      pin_memory=(device_str() == "cuda"), drop_last=False)


# --------------------------------------------------------------------------- #
# Model                                                                        #
# --------------------------------------------------------------------------- #
def make_model(n_classes: int):
    """ResNet-18 CIFAR variant: 3x3 stem, no maxpool."""
    import torch.nn as nn
    import torchvision
    m = torchvision.models.resnet18(num_classes=n_classes)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m


def feats_logits(model, x):
    """Penultimate features + logits (exact last-layer geometry for GraNd)."""
    import torch
    h = model.conv1(x)
    h = model.bn1(h)
    h = model.relu(h)
    h = model.maxpool(h)
    h = model.layer1(h)
    h = model.layer2(h)
    h = model.layer3(h)
    h = model.layer4(h)
    h = model.avgpool(h)
    f = torch.flatten(h, 1)
    return f, model.fc(f)


# --------------------------------------------------------------------------- #
# Train / eval                                                                 #
# --------------------------------------------------------------------------- #
def train_model(x, y, dataset: str, tcfg: dict, seed: int, epoch_hook=None):
    import torch
    set_seed(seed)
    dev = device_str()
    model = make_model(N_CLASSES[dataset]).to(dev)
    dl = make_loader(ArrayDataset(x, y, dataset, aug=True), tcfg["batch"],
                     shuffle=True, seed=seed, workers=tcfg.get("workers", 4))
    opt = torch.optim.SGD(model.parameters(), lr=tcfg["lr"],
                          momentum=tcfg["momentum"], nesterov=True,
                          weight_decay=tcfg["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=tcfg["lr"], total_steps=tcfg["epochs"] * len(dl))
    use_amp = bool(tcfg.get("amp", True)) and dev == "cuda"
    scaler = torch.amp.GradScaler(dev, enabled=use_amp)
    loss_fn = torch.nn.CrossEntropyLoss()
    for ep in range(tcfg["epochs"]):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(dev, non_blocking=True), yb.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(dev, enabled=use_amp):
                loss = loss_fn(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
        if epoch_hook is not None:
            epoch_hook(model, ep + 1)  # 1-based epoch number
    return model


def predict_probs(model, x, y, dataset: str, batch: int,
                  want_featnorm: bool = False):
    """No-aug forward pass. Returns (probs float32 [N,C], featnorm|None)."""
    import torch
    dev = device_str()
    dl = make_loader(ArrayDataset(x, y, dataset, aug=False), batch,
                     shuffle=False, seed=0, workers=2)
    model.eval()
    probs, fnorms = [], []
    with torch.no_grad(), torch.amp.autocast(dev, enabled=(dev == "cuda")):
        for xb, _ in dl:
            f, lg = feats_logits(model, xb.to(dev, non_blocking=True))
            probs.append(torch.softmax(lg.float(), dim=1).cpu())
            if want_featnorm:
                fnorms.append(f.float().norm(dim=1).cpu())
    probs = torch.cat(probs).numpy().astype(np.float32)
    fn = torch.cat(fnorms).numpy().astype(np.float32) if want_featnorm else None
    return probs, fn


def accuracy(model, x, y, dataset: str, batch: int) -> float:
    probs, _ = predict_probs(model, x, y, dataset, batch)
    return float((probs.argmax(1) == y).mean())


def eval_metrics(model, x, y, dataset: str, batch: int) -> dict:
    """Accuracy + per-class accuracy + 15-bin expected calibration error."""
    probs, _ = predict_probs(model, x, y, dataset, batch)
    pred = probs.argmax(1)
    correct = pred == y
    n_cls = N_CLASSES[dataset]
    cls_correct = np.bincount(y[correct], minlength=n_cls).astype(np.float64)
    cls_total = np.bincount(y, minlength=n_cls).astype(np.float64)
    per_class = np.divide(cls_correct, cls_total,
                          out=np.full(n_cls, np.nan), where=cls_total > 0)
    conf = probs.max(1)
    ece = 0.0
    n = len(y)
    for lo in np.linspace(0, 1, 16)[:-1]:
        m = (conf > lo) & (conf <= lo + 1 / 15)
        if m.any():
            ece += m.sum() / n * abs(correct[m].mean() - conf[m].mean())
    return {"acc": float(correct.mean()), "per_class": per_class,
            "cls_correct": cls_correct, "cls_total": cls_total,
            "ece": float(ece)}


def _el2n(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y)), y] = 1.0
    return np.linalg.norm(probs - onehot, axis=1)


# --------------------------------------------------------------------------- #
# Probe unit (loss-geometry scores)                                            #
# --------------------------------------------------------------------------- #
def run_probe_unit(dataset: str, sseed: int, k: int, cfg: dict,
                   out_path: Path) -> None:
    """Train one probe model on the FULL clean train set and record the
    per-example statistics behind EL2N / GraNd / forgetting / entropy.
    Snapshots are taken at every epoch in probe.snapshot_epochs (union with
    the canonical el2n/grand epochs), storing el2n_e{N} and grand_e{N} keys —
    this feeds both the main scores and the scoring-choice epoch ablation.
    Forgetting is counted on per-epoch no-aug eval passes over the train set
    (DeepCore-style deterministic variant of Toneva et al.)."""
    pcfg = cfg["probe"]
    data = load_dataset(dataset, cfg["data_dir"],
                        cfg.get("train_subset"), cfg.get("test_subset"))
    xtr, ytr = data["xtr"], data["ytr"]
    n, e_total = len(ytr), pcfg["epochs"]
    snap_eps = sorted({int(e) for e in pcfg.get("snapshot_epochs", [])}
                      | {int(pcfg["el2n_epoch"]), int(pcfg["grand_epoch"])})
    snap_eps = [e for e in snap_eps if 1 <= e <= e_total]
    correct = np.zeros((e_total, n), dtype=bool)
    snap: dict[str, np.ndarray] = {}

    def hook(model, ep):
        want = ep in snap_eps
        probs, fn = predict_probs(model, xtr, ytr, dataset, pcfg["batch"],
                                  want_featnorm=want)
        correct[ep - 1] = probs.argmax(1) == ytr
        if want:
            e = _el2n(probs, ytr)
            snap[f"el2n_e{ep}"] = e
            snap[f"grand_e{ep}"] = e * fn  # exact last-layer grad norm
        if ep == e_total:
            snap["entropy"] = -(probs * np.log(probs + 1e-12)).sum(1)

    probe_seed = 100000 + sseed * 1000 + k  # disjoint from train seeds
    train_model(xtr, ytr, dataset, pcfg, probe_seed, epoch_hook=hook)
    forget = (correct[:-1] & ~correct[1:]).sum(0).astype(np.float32)
    never_learned = ~correct.any(0)
    forget[never_learned] = e_total  # Toneva: unlearned examples rank hardest
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.npz")
    np.savez(tmp, forgetting=forget, entropy=snap["entropy"],
             **{k_: v for k_, v in snap.items() if k_ != "entropy"},
             meta=json.dumps({"dataset": dataset, "sseed": sseed, "k": k,
                              "probe_cfg": pcfg, "snapshot_epochs": snap_eps,
                              "fake": FAKE}))
    os.replace(tmp, out_path)


# --------------------------------------------------------------------------- #
# FM-feature scoring unit                                                      #
# --------------------------------------------------------------------------- #
def run_fm_unit(dataset: str, method: str, cfg: dict, out_path: Path) -> None:
    """Score = L2 distance to the class centroid in the FM embedding space
    (embeddings L2-normalised first); high = hard/atypical, matching the
    keep-highest convention of the loss-based scores (Sorscher et al. 2022
    prototype framing)."""
    import torch
    import torch.nn.functional as F
    set_seed(0)
    dev = device_str()
    data = load_dataset(dataset, cfg["data_dir"],
                        cfg.get("train_subset"), cfg.get("test_subset"))
    xtr, ytr = data["xtr"], data["ytr"]
    model_name = cfg["fm_models"][method]
    img_size = int(cfg.get("fm_img_size", 224))
    import timm
    model = timm.create_model(model_name, pretrained=not FAKE, num_classes=0,
                              img_size=img_size).to(dev).eval()
    dc = timm.data.resolve_model_data_config(model)
    mean = torch.tensor(dc["mean"], device=dev).view(1, 3, 1, 1)
    std = torch.tensor(dc["std"], device=dev).view(1, 3, 1, 1)
    embs = []
    bs = int(cfg.get("fm_batch", 256))
    with torch.no_grad(), torch.amp.autocast(dev, enabled=(dev == "cuda")):
        for i in range(0, len(ytr), bs):
            xb = torch.from_numpy(
                np.ascontiguousarray(
                    xtr[i:i + bs].transpose(0, 3, 1, 2))).to(dev).float() / 255.0
            xb = F.interpolate(xb, size=img_size, mode="bicubic",
                               align_corners=False)
            xb = (xb - mean) / std
            embs.append(model(xb).float().cpu())
    e = F.normalize(torch.cat(embs), dim=1).numpy()
    score = np.empty(len(ytr), dtype=np.float32)
    for c in np.unique(ytr):
        m = ytr == c
        cent = e[m].mean(0)
        cent /= (np.linalg.norm(cent) + 1e-12)
        score[m] = np.linalg.norm(e[m] - cent, axis=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.npz")
    np.savez(tmp, score=score,
             meta=json.dumps({"dataset": dataset, "method": method,
                              "model": model_name, "img_size": img_size,
                              "fake": FAKE}))
    os.replace(tmp, out_path)


# --------------------------------------------------------------------------- #
# Training unit (the benchmark unit)                                           #
# --------------------------------------------------------------------------- #
def score_file_for(scores_dir: Path, dataset: str, method: str,
                   sseed: int, cfg: dict) -> Path:
    fm = method in cfg.get("fm_models", {})
    tag = "sNA" if fm else f"s{sseed}"
    return Path(scores_dir) / f"{dataset}__{method}__{tag}.npz"


def run_train_unit(dataset: str, method: str, ratio: int, seed: int,
                   sseed: int, cfg: dict, scores_dir: str,
                   out_path: Path) -> None:
    import torch
    t0 = __import__("time").time()
    data = load_dataset(dataset, cfg["data_dir"],
                        cfg.get("train_subset"), cfg.get("test_subset"))
    xtr, ytr, xte, yte = data["xtr"], data["ytr"], data["xte"], data["yte"]
    n = len(ytr)

    if method == "full":
        idx = np.arange(n)
        sfile, sha = None, None
    else:
        sf = score_file_for(scores_dir, dataset, method, sseed, cfg)
        arr = np.load(sf)["score"].astype(np.float32)
        if len(arr) != n:
            raise RuntimeError(f"score length {len(arr)} != train size {n} ({sf})")
        sha = __import__("hashlib").sha256(arr.tobytes()).hexdigest()
        sfile = str(sf)
        keep = int(round(n * ratio / 100))
        idx = np.argsort(-arr, kind="stable")[:keep]  # keep hardest

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    model = train_model(xtr[idx], ytr[idx], dataset, cfg["train"], seed)

    ecfg = cfg["eval"]
    clean = eval_metrics(model, xte, yte, dataset, ecfg["batch"])
    subset = ecfg.get("subset") or len(yte)
    n_cls = N_CLASSES[dataset]
    corr: dict[str, dict[str, float]] = {}
    corr_ece: list[float] = []
    corr_cls_correct = np.zeros(n_cls)
    corr_cls_total = np.zeros(n_cls)
    for cname in ecfg["corruptions"]:
        xc, yc = load_corruption(dataset, cname, cfg["data_dir"], subset)
        corr[cname] = {}
        for sev in ecfg["severities"]:
            em = eval_metrics(model, xc[sev - 1], yc, dataset, ecfg["batch"])
            corr[cname][str(sev)] = em["acc"]
            corr_ece.append(em["ece"])
            corr_cls_correct += em["cls_correct"]
            corr_cls_total += em["cls_total"]
    mca = float(np.mean([a for d in corr.values() for a in d.values()]))
    per_class_corr = np.divide(corr_cls_correct, corr_cls_total,
                               out=np.full(n_cls, np.nan),
                               where=corr_cls_total > 0)

    import importlib.metadata as im
    vers = {p: _ver(im, p) for p in ("torch", "torchvision", "numpy", "timm")}
    result = {
        "dataset": dataset, "method": method, "ratio": ratio, "seed": seed,
        "sseed": None if method == "full" else sseed,
        "scoring_mode": cfg.get("scoring_mode"),
        "score_file": sfile, "score_sha256": sha,
        "n_train": int(len(idx)),
        "class_hist": np.bincount(ytr[idx],
                                  minlength=N_CLASSES[dataset]).tolist(),
        "metrics": {"clean_acc": clean["acc"], "mca": mca,
                    "clean_ece": clean["ece"],
                    "corr_ece": float(np.mean(corr_ece))},
        "per_class": {"clean": np.round(clean["per_class"], 6).tolist(),
                      "corr_mean": np.round(per_class_corr, 6).tolist()},
        "corr": corr,
        "wall_s": round(__import__("time").time() - t0, 1),
        "peak_vram_mb": (round(torch.cuda.max_memory_allocated() / 2**20, 1)
                         if torch.cuda.is_available() else None),
        "fake": FAKE, "versions": vers,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2))
    os.replace(tmp, out_path)


def _ver(im, pkg):
    try:
        return im.version(pkg)
    except Exception:
        return None
