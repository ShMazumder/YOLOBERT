"""datasets/ — dataset loaders + dataloader factory.

Config-driven. Set in configs/*.yaml:
    dataset: coco | yolo | voc | dummy
    data_root: data/coco
    train_split: train2017
    val_split:   val2017

    from datasets import build_dataloaders
    train_loader, val_loader = build_dataloaders(cfg)

Detection loaders yield (image_tensor, target_dict) where target_dict holds
boxes/labels and image_id — the format tools/metrics.CocoMetric consumes after
your model head converts predictions to COCO result dicts.
"""
from .factory import build_dataloaders  # noqa: F401

__all__ = ["build_dataloaders"]
