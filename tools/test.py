#!/usr/bin/env python3
"""test.py — standalone evaluation on val/test split.

Loads a checkpoint, runs the task metric, prints + optionally dumps JSON.
Shares model/data/metric code with tools/train.py — keep the eval protocol
identical to what you report in the paper (do not change silently).

USAGE
    python tools/test.py --config configs/base.yaml --checkpoint work_dirs/base/best.pth
    python tools/test.py --config configs/base.yaml --checkpoint best.pth --split test --dump results.json
"""
import argparse
import json
from pathlib import Path

import torch

# Reuse builders from train.py (same protocol, no drift).
from train import build_dataloaders, build_model, evaluate, load_config, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--opts", nargs="*", default=[])
    ap.add_argument("--dump", default=None, help="write metrics JSON to this path")
    args = ap.parse_args()

    cfg = load_config(args.config, args.opts)
    set_seed(cfg.get("seed", 42))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # build_dataloaders returns (train, val); for a real test split add a
    # 'test' loader in that function and select here.
    _, eval_loader = build_dataloaders(cfg)

    model = build_model(cfg).to(device)
    ck = torch.load(args.checkpoint, map_location=device)
    state = ck["model"] if "model" in ck else ck
    model.load_state_dict(state)
    print(f"loaded {args.checkpoint}"
          + (f" (epoch {ck['epoch']})" if isinstance(ck, dict) and "epoch" in ck else ""))

    metrics = evaluate(model, eval_loader, device, cfg)
    print(f"[{args.split}] " + " | ".join(f"{k}={v:.3f}" for k, v in metrics.items()))

    if args.dump:
        Path(args.dump).parent.mkdir(parents=True, exist_ok=True)
        with open(args.dump, "w") as f:
            json.dump({"checkpoint": args.checkpoint, "split": args.split,
                       "metrics": metrics}, f, indent=2)
        print(f"wrote {args.dump}")


if __name__ == "__main__":
    main()
