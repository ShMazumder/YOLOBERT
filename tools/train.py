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
    # TODO: replace with your Dataset. Return (train_loader, val_loader).
    from torch.utils.data import DataLoader, TensorDataset
    n = cfg.get("_dummy_n", 256)
    d = cfg.get("_dummy_dim", 32)
    ds = TensorDataset(torch.randn(n, d), torch.randint(0, cfg.get("num_classes", 10), (n,)))
    bs = cfg.get("batch_size", 32)
    return (DataLoader(ds, batch_size=bs, shuffle=True, num_workers=cfg.get("workers", 2)),
            DataLoader(ds, batch_size=bs, shuffle=False, num_workers=cfg.get("workers", 2)))


# ------------------------------------------------------------------ model
def build_model(cfg):
    # TODO: replace with your architecture.
    return torch.nn.Sequential(
        torch.nn.Linear(cfg.get("_dummy_dim", 32), 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, cfg.get("num_classes", 10)),
    )


# ------------------------------------------------------------------ loops
def train_one_epoch(model, loader, optimizer, criterion, device, epoch, writer):
    model.train()
    running = 0.0
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        running += loss.item()
        if i % 20 == 0:
            step = epoch * len(loader) + i
            writer.add_scalar("train/loss", loss.item(), step)
    return running / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, device):
    """TODO: swap for task metric (AP/mIoU/Top-1/...). Returns dict."""
    model.eval()
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
        metrics = evaluate(model, val_loader, device)
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
