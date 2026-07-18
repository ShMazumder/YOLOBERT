"""build_dataloaders(cfg) — dispatch to the right dataset + wrap in DataLoader.

Detection datasets return variable-size targets, so a collate_fn that keeps
per-image target dicts (list) is used. The dummy path keeps the scaffold
runnable with zero data on disk.
"""
import torch
from torch.utils.data import DataLoader, TensorDataset


def detection_collate(batch):
    """Stack images if same size else keep list; keep targets as a list."""
    imgs, targets = list(zip(*batch))
    try:
        imgs = torch.stack(imgs, 0)          # works if a transform fixed the size
    except Exception:
        imgs = list(imgs)                     # ragged -> list (resize in transform)
    return imgs, list(targets)


def _loader(ds, cfg, shuffle, collate=None):
    return DataLoader(ds, batch_size=cfg.get("batch_size", 16), shuffle=shuffle,
                      num_workers=cfg.get("workers", 4), pin_memory=True,
                      collate_fn=collate)


def build_dataloaders(cfg):
    name = cfg.get("dataset", "dummy")

    if name == "dummy":
        d = cfg.get("_dummy_dim", 32)
        n = cfg.get("_dummy_n", 256)
        ds = TensorDataset(torch.randn(n, d),
                           torch.randint(0, cfg.get("num_classes", 10), (n,)))
        return _loader(ds, cfg, True), _loader(ds, cfg, False)

    root = cfg["data_root"]
    # transforms: explicit cfg['_transforms'] wins; else build defaults from img_size.
    tf_tr = cfg.get("_transforms")
    tf_va = cfg.get("_transforms")
    if tf_tr is None and cfg.get("img_size"):
        from .transforms import default_train, default_val
        tf_tr = default_train(cfg["img_size"])
        tf_va = default_val(cfg["img_size"])

    if name == "coco":
        from .coco import CocoDetection
        import os
        tr = CocoDetection(os.path.join(root, cfg.get("train_img_dir", "train2017")),
                           cfg["train_ann"], transforms=tf_tr)
        va = CocoDetection(os.path.join(root, cfg.get("val_img_dir", "val2017")),
                           cfg["val_ann"], transforms=tf_va)
    elif name == "yolo":
        from .yolo import YoloDetection
        tr = YoloDetection(root, cfg.get("train_split", "train"), transforms=tf_tr)
        va = YoloDetection(root, cfg.get("val_split", "val"), transforms=tf_va)
    elif name == "voc":
        from .voc import VocDetection
        tr = VocDetection(root, cfg.get("train_split", "trainval"), transforms=tf_tr)
        va = VocDetection(root, cfg.get("val_split", "test"), transforms=tf_va)
    else:
        raise KeyError(f"unknown dataset '{name}'. options: dummy|coco|yolo|voc")

    return (_loader(tr, cfg, True, detection_collate),
            _loader(va, cfg, False, detection_collate))
