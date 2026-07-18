"""YOLO-format detection dataset.

Layout:
    root/images/<split>/*.jpg
    root/labels/<split>/*.txt      # each line: cls cx cy w h  (normalized 0..1)

Yields (image, target) with boxes in xyxy PIXEL coords to match CocoDetection,
so the same model head + CocoMetric path works. class ids shifted +1 (0=bg).
"""
import glob
import os

import torch
from PIL import Image
from torch.utils.data import Dataset


class YoloDetection(Dataset):
    def __init__(self, root, split="train", transforms=None):
        self.img_dir = os.path.join(root, "images", split)
        self.lbl_dir = os.path.join(root, "labels", split)
        exts = ("*.jpg", "*.jpeg", "*.png")
        self.imgs = sorted(p for e in exts for p in glob.glob(os.path.join(self.img_dir, e)))
        self.transforms = transforms

    def __len__(self):
        return len(self.imgs)

    def _label_path(self, img_path):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        return os.path.join(self.lbl_dir, stem + ".txt")

    def __getitem__(self, i):
        img_path = self.imgs[i]
        img = Image.open(img_path).convert("RGB")
        W, H = img.size

        boxes, labels = [], []
        lp = self._label_path(img_path)
        if os.path.isfile(lp):
            with open(lp) as f:
                for line in f:
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                    c, cx, cy, w, h = map(float, parts)
                    x1 = (cx - w / 2) * W; y1 = (cy - h / 2) * H
                    x2 = (cx + w / 2) * W; y2 = (cy + h / 2) * H
                    boxes.append([x1, y1, x2, y2])
                    labels.append(int(c) + 1)               # 0 = background

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([i]),
        }
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target
