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
        if "=" not in kv:               # stray token (e.g. shell-leaked comment)
            print(f"[warn] ignoring malformed --opts token: {kv!r} (need key=value)")
            continue
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
def _dummy_loaders(cfg):
    from torch.utils.data import DataLoader, TensorDataset
    n, d = cfg.get("_dummy_n", 256), cfg.get("_dummy_dim", 32)
    ds = TensorDataset(torch.randn(n, d),
                       torch.randint(0, cfg.get("num_classes", 10), (n,)))
    bs = cfg.get("batch_size", 32)
    return (DataLoader(ds, batch_size=bs, shuffle=True, num_workers=cfg.get("workers", 2)),
            DataLoader(ds, batch_size=bs, shuffle=False, num_workers=cfg.get("workers", 2)))


def build_dataloaders(cfg):
    """Dispatch via the local datasets/ package (dummy|coco|yolo|voc).

    Import errors fall back to a dummy loader (scaffold still runs). Real dataset
    construction errors (bad path, missing anns) are raised — never silently
    masked by the dummy path.
    """
    try:
        import importlib
        import sys
        repo = str(Path(__file__).resolve().parent.parent)
        if sys.path[0] != repo:
            sys.path.insert(0, repo)
        # Evict a shadowing site-packages 'datasets' (e.g. HuggingFace, preinstalled
        # on Kaggle/Colab) so our local package resolves from repo root.
        cached = sys.modules.get("datasets")
        cached_file = getattr(cached, "__file__", "") or ""
        if cached is not None and repo not in cached_file:
            for k in [m for m in sys.modules if m == "datasets" or m.startswith("datasets.")]:
                del sys.modules[k]
        datasets_pkg = importlib.import_module("datasets")
        if not hasattr(datasets_pkg, "build_dataloaders"):
            raise ImportError(f"resolved wrong 'datasets' at {datasets_pkg.__file__}")
    except Exception as e:
        print(f"[warn] local datasets package unavailable ({e}); inline dummy loader")
        return _dummy_loaders(cfg)
    # import ok -> let real data errors propagate loudly
    return datasets_pkg.build_dataloaders(cfg)


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
            # .mean() collapses DataParallel's per-GPU stacked losses; scalar-safe.
            reduced = {k: (v.mean() if v.dim() > 0 else v) for k, v in out.items()}
            loss = sum(reduced.values())
            log_parts = {k: v.item() for k, v in reduced.items()}
        else:                                # classification: external criterion
            loss = criterion(out, y)
            log_parts = {"loss": loss.item()}
        loss.backward()
        optimizer.step()
        running += loss.item()
        if writer is not None and i % 20 == 0:
            step = epoch * len(loader) + i
            for k, v in log_parts.items():
                writer.add_scalar(f"train/{k}", v, step)
    return running / max(len(loader), 1)


def _unwrap(model):
    """Underlying module, past DataParallel / DistributedDataParallel."""
    return model.module if hasattr(model, "module") else model


def _wants_targets(model):
    """True if model.forward accepts a targets arg (detection-style)."""
    import inspect
    try:
        return "targets" in inspect.signature(_unwrap(model).forward).parameters
    except (ValueError, TypeError):
        return False


# ------------------------------------------------------------------ distributed
def _dist_env():
    """Read torchrun-provided env. Returns (is_ddp, rank, local_rank, world_size)."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        return (True, int(os.environ["RANK"]),
                int(os.environ.get("LOCAL_RANK", 0)), int(os.environ["WORLD_SIZE"]))
    return (False, 0, 0, 1)


def is_main(rank):
    return rank == 0


class DetDataParallel(torch.nn.DataParallel):
    """DataParallel that also splits a detection target LIST across GPUs.

    Stock DataParallel scatters tensors along dim0 but mangles a Python list of
    variable-size per-image target dicts. Here we scatter images the normal way,
    then slice the target list to match each device's image count.
    """
    def scatter(self, inputs, kwargs, device_ids):
        from torch.nn.parallel.scatter_gather import scatter as _scatter
        images, targets = inputs                       # (tensor, list[dict])
        img_parts = _scatter(images, device_ids)       # per-device image chunks
        parts, idx = [], 0
        for dev, imgs in zip(device_ids, img_parts):
            c = imgs.size(0)
            sub = [{k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in d.items()}
                   for d in targets[idx:idx + c]]
            parts.append((imgs, sub))
            idx += c
        return parts, [kwargs] * len(parts)


@torch.no_grad()
def evaluate(model, loader, device, cfg=None):
    """Task metric via tools/metrics.py when cfg['metric'] set; else top-1.

    For coco/semseg you must adapt the model output -> metric.update() call
    below to your prediction format (see metrics.py docstrings).
    """
    net = _unwrap(model)            # eval single-GPU; skip DP/DDP output gather
    net.eval()
    cfg = cfg or {}
    if cfg.get("metric"):
        from metrics import build_metric
        m = build_metric(cfg)
        wants = _wants_targets(net)
        # map contiguous label ids back to original dataset category ids (COCO sparse)
        label2cat = getattr(getattr(loader, "dataset", None), "label2cat", None)
        for batch in loader:
            x, y = _to_device(batch, device)
            # detection models need targets in eval for real image_id/orig_size
            out = net(x, y) if wants else net(x)
            if label2cat and isinstance(out, list):
                for d in out:
                    d["category_id"] = label2cat.get(d["category_id"], d["category_id"])
            m.update(out, y)            # cls/semseg: logits/labels; coco: dicts
        return m.compute()
    # default: top-1 (scaffold / classification)
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = net(x).argmax(1)
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

    # --- parallelism switch: cfg['parallel'] = none | dp | ddp ---
    # none : single GPU.
    # dp   : nn.DataParallel — works with plain `python tools/train.py` (Kaggle 2xT4).
    # ddp  : DistributedDataParallel — MUST be launched via torchrun, e.g.
    #        torchrun --nproc_per_node=2 tools/train.py --config <cfg> --opts parallel=ddp
    mode = str(cfg.get("parallel", "none")).lower()
    env_ddp, rank, local_rank, world = _dist_env()
    if mode == "ddp" and not env_ddp:
        print("[warn] parallel=ddp but not launched via torchrun; falling back to dp")
        mode = "dp"
    use_ddp = mode == "ddp" and env_ddp

    if use_ddp:
        import torch.distributed as dist
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
        cfg["_distributed"] = True          # factory -> DistributedSampler
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    main_proc = is_main(rank)

    set_seed(cfg.get("seed", 42) + rank)     # decorrelate per-rank augmentation

    work_dir = Path(cfg.get("work_dir", "work_dirs/exp"))
    if main_proc:
        work_dir.mkdir(parents=True, exist_ok=True)
        with open(work_dir / "config_used.yaml", "w") as f:
            yaml.safe_dump({k: v for k, v in cfg.items() if not k.startswith("_")}, f)
    writer = SummaryWriter(work_dir / "tb") if main_proc else None

    train_loader, val_loader = build_dataloaders(cfg)
    model = build_model(cfg).to(device)

    if use_ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False)
    elif mode == "dp" and torch.cuda.device_count() > 1:
        # detection returns list targets -> needs list-aware scatter
        dp_cls = DetDataParallel if _wants_targets(model) else torch.nn.DataParallel
        model = dp_cls(model)
        if main_proc:
            print(f"{dp_cls.__name__} across {torch.cuda.device_count()} GPUs")

    optimizer = torch.optim.AdamW(_unwrap(model).parameters(),
                                  lr=cfg.get("lr", 1e-3),
                                  weight_decay=cfg.get("weight_decay", 1e-4))
    criterion = torch.nn.CrossEntropyLoss()   # TODO: your loss (classification path)

    start_epoch, best = 0, -1.0
    if args.resume and os.path.isfile(args.resume):
        ck = torch.load(args.resume, map_location=device)
        _unwrap(model).load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        start_epoch = ck["epoch"] + 1
        best = ck.get("best", -1.0)
        if main_proc:
            print(f"resumed from {args.resume} @ epoch {start_epoch}")

    epochs = cfg.get("epochs", 20)
    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        if use_ddp and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)   # reshuffle shards each epoch
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, writer)

        if main_proc:                        # only rank 0 evals + saves + logs
            metrics = evaluate(model, val_loader, device, cfg)
            primary = metrics[cfg.get("primary_metric", "top1")]
            for k, v in metrics.items():
                writer.add_scalar(f"val/{k}", v, epoch)
            print(f"epoch {epoch:3d} | loss {loss:.4f} | "
                  + " ".join(f"{k} {v:.2f}" for k, v in metrics.items())
                  + f" | {time.time()-t0:.1f}s")

            ck = {"epoch": epoch, "model": _unwrap(model).state_dict(),
                  "optimizer": optimizer.state_dict(), "best": best, "cfg": cfg}
            torch.save(ck, work_dir / "last.pth")
            if primary > best:
                best = primary
                torch.save(ck, work_dir / "best.pth")
                print(f"  new best {cfg.get('primary_metric','top1')}={best:.2f}")

        if use_ddp:                          # resync: rank0 eval takes longer
            import torch.distributed as dist
            dist.barrier()

    if writer is not None:
        writer.close()
    if use_ddp:
        import torch.distributed as dist
        dist.destroy_process_group()
    if main_proc:
        print(f"done. best={best:.2f}. logs -> {work_dir}")


if __name__ == "__main__":
    main()
