"""COCO-format detection/instance-seg dataset.

Reads a COCO annotations json + image dir. Yields (image, target) where
target = {boxes (xyxy), labels, image_id, area, iscrowd}. Keep image_id so
predictions map back for CocoMetric.

Deps: pycocotools, pillow. Bring your own transforms (pass `transforms=`).
"""
import os

import torch
from PIL import Image
from torch.utils.data import Dataset


class CocoDetection(Dataset):
    def __init__(self, img_dir, ann_file, transforms=None):
        from pycocotools.coco import COCO
        self.img_dir = img_dir
        self.coco = COCO(ann_file)
        self.ids = sorted(self.coco.imgs.keys())
        self.transforms = transforms
        # contiguous label remap (COCO cat ids are sparse 1..90)
        cats = sorted(self.coco.getCatIds())
        self.cat2label = {c: i + 1 for i, c in enumerate(cats)}  # 0 = background
        self.label2cat = {v: k for k, v in self.cat2label.items()}  # contig -> COCO id

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        img_id = self.ids[i]
        info = self.coco.loadImgs(img_id)[0]
        img = Image.open(os.path.join(self.img_dir, info["file_name"])).convert("RGB")

        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id, iscrowd=None))
        boxes, labels, areas, iscrowd = [], [], [], []
        for a in anns:
            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])          # xywh -> xyxy
            labels.append(self.cat2label[a["category_id"]])
            areas.append(a.get("area", w * h))
            iscrowd.append(a.get("iscrowd", 0))

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([img_id]),
            "orig_size": torch.tensor([info["width"], info["height"]]),  # (W,H)
            "area": torch.tensor(areas, dtype=torch.float32),
            "iscrowd": torch.tensor(iscrowd, dtype=torch.int64),
        }
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target
