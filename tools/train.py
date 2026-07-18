#!/usr/bin/env python3
"""train.py — config-driven training scaffold for ML research.

Framework-agnostic PyTorch stub. Fill the TODOs with your model, dataset, and
loss. Keeps eval protocol + seeds explicit so results are reproducible and
traceable (see CLAUDE.md conventions).

USAGE
    python tools/train.py --config configs/base.yaml
    python tools/train.py --config configs/base.yaml --opts lr=0.001 epochs=50
"""
import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter


# ------------------------------------------------------------------ config
def load_config(path, opts):
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    for kv in opts or []:               # CLI overrides: key=value
        k, v = kv.split("=", 1)
        cfg[k] = yaml.safe_load(v)      # parse int/float/bool/str
    return cfg


def set_seed(seed):
    """Deterministic-ish. NOTE changing this changes results — log it."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True   # set False for full determinism


# ------------------------------------------------------------------ data
def build_dataloaders(cfg):
    """Dispatch via the datasets/ package (dummy|coco|yolo|voc).

    Falls back to an inline dummy loader if the package import fails, so the
    scaffold runs standalone.
    """
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from datasets import build_dataloaders as _build
        return _build(cfg)
    except Exception as e:  # pragma: no cover - fallback path
        print(f"[warn] datasets package unavailable ({e}); inline dummy loader")
        from torch.utils.data import DataLoader, TensorDataset
        n, d = cfg.get("_dummy_n", 256), cfg.get("_dummy_dim", 32)
        ds = TensorDataset(torch.randn(n, d),
                           torch.randint(0, cfg.get("num_classes", 10), (n,)))
        bs = cfg.get("batch_size", 32)
        return (DataLoader(ds, batch_size=bs, shuffle=True, num_workers=cfg.get("workers", 2)),
                DataLoader(ds, batch_size=bs, shuffle=False, num_workers=cfg.get("workers", 2)))


# ------------------------------------------------------------------ model
def build_model(cfg):
    """Dispatch via the models/ registry. Set `model:` + `model_*` in config.

    Falls back to a tiny inline net only if the registry import fails, so the
    scaffold still runs standalone before you add real architectures.
    """
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from models import build as build_registered
        cfg.setdefault("model", "example_net")
        cfg.setdefault("model_in_dim", cfg.get("_dummy_dim", 32))
        cfg.setdefault("model_num_classes", cfg.get("num_classes", 10))
        return build_registered(cfg)
    except Exception as e:  # pragma: no cover - fallback path
        print(f"[warn] registry unavailable ({e}); using inline stub")
        return torch.nn.Sequential(
            torch.nn.Linear(cfg.get("_dummy_dim", 32), 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, cfg.get("num_classes", 10)),
        )


# ------------------------------------------------------------------ loops
def _to_device(batch, device):
    """Images tensor -> device; detection targets (list of dicts) -> device."""
    x, y = batch
    if torch.is_tensor(x):
        x = x.to(device)
    else:                                   # ragged list of image tensors
        x = [img.to(device) for img in x]
    if isinstance(y, (list, tuple)) and len(y) and isinstance(y[0], dict):
        y = [{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in t.items()}
             for t in y]
    elif torch.is_tensor(y):
        y = y.to(device)
    return x, y


def train_one_epoch(model, loader, optimizer, criterion, device, epoch, writer):
    model.train()
    running = 0.0
    for i, batch in enumerate(loader):
        x, y = _to_device(batch, device)
        optimizer.zero_grad()
        out = model(x, y) if _wants_targets(model) else model(x)
        if isinstance(out, dict):           # detection: model returns loss dict
            loss = sum(out.values())
            log_parts = {k: v.item() for k, v in out.items()}
        else:                                # classification: external criterion
            loss = criterion(out, y)
            log_parts = {"loss": loss.item()}
        loss.backward()
        optimizer.step()
        running += loss.item()
        if i % 20 == 0:
            step = epoch * len(loader) + i
            for k, v in log_parts.items():
                writer.add_scalar(f"train/{k}", v, step)
    return running / max(len(loader), 1)


def _wants_targets(model):
    """True if model.forward accepts a targets arg (detection-style)."""
    import inspect
    try:
        return "targets" in inspect.signature(model.forward).parameters
    except (ValueError, TypeError):
        return False


@torch.no_grad()
def evaluate(model, loader, device, cfg=None):
    """Task metric via tools/metrics.py when cfg['metric'] set; else top-1.

    For coco/semseg you must adapt the model output -> metric.update() call
    below to your prediction format (see metrics.py docstrings).
    """
    model.eval()
    cfg = cfg or {}
    if cfg.get("metric"):
        from metrics import build_metric
        m = build_metric(cfg)
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            m.update(out, y)            # cls/semseg: logits/labels; coco: adapt
        return m.compute()
    # default: top-1 (scaffold / classification)
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return {"top1": 100.0 * correct / max(total, 1)}


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--opts", nargs="*", default=[], help="key=value overrides")
    ap.add_argument("--resume", default=None, help="checkpoint path")
    args = ap.parse_args()

    cfg = load_config(args.config, args.opts)
    set_seed(cfg.get("seed", 42))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    work_dir = Path(cfg.get("work_dir", "work_dirs/exp"))
    work_dir.mkdir(parents=True, exist_ok=True)
    with open(work_dir / "config_used.yaml", "w") as f:
        yaml.safe_dump(cfg, f)          # snapshot exact config with results
    writer = SummaryWriter(work_dir / "tb")

    train_loader, val_loader = build_dataloaders(cfg)
    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg.get("lr", 1e-3),
                                  weight_decay=cfg.get("weight_decay", 1e-4))
    criterion = torch.nn.CrossEntropyLoss()   # TODO: your loss

    start_epoch, best = 0, -1.0
    if args.resume and os.path.isfile(args.resume):
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        start_epoch = ck["epoch"] + 1
        best = ck.get("best", -1.0)
        print(f"resumed from {args.resume} @ epoch {start_epoch}")

    epochs = cfg.get("epochs", 20)
    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, writer)
        metrics = evaluate(model, val_loader, device, cfg)
        primary = metrics[cfg.get("primary_metric", "top1")]
        for k, v in metrics.items():
            writer.add_scalar(f"val/{k}", v, epoch)
        print(f"epoch {epoch:3d} | loss {loss:.4f} | "
              + " ".join(f"{k} {v:.2f}" for k, v in metrics.items())
              + f" | {time.time()-t0:.1f}s")

        ck = {"epoch": epoch, "model": model.state_dict(),
              "optimizer": optimizer.state_dict(), "best": best, "cfg": cfg}
        torch.save(ck, work_dir / "last.pth")
        if primary > best:
            best = primary
            torch.save(ck, work_dir / "best.pth")
            print(f"  new best {cfg.get('primary_metric','top1')}={best:.2f}")

    writer.close()
    print(f"done. best={best:.2f}. logs -> {work_dir}")


if __name__ == "__main__":
    main()
