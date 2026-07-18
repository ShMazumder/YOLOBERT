"""Pascal VOC-format detection dataset (XML annotations).

Layout:
    root/JPEGImages/*.jpg
    root/Annotations/*.xml
    root/ImageSets/Main/<split>.txt   # one image id per line

Yields (image, target) with xyxy pixel boxes, matching CocoDetection/YoloDetection.
"""
import os
import xml.etree.ElementTree as ET

import torch
from PIL import Image
from torch.utils.data import Dataset

VOC_CLASSES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)


class VocDetection(Dataset):
    def __init__(self, root, split="trainval", classes=VOC_CLASSES, transforms=None):
        self.root = root
        self.cls2idx = {c: i + 1 for i, c in enumerate(classes)}  # 0 = background
        with open(os.path.join(root, "ImageSets", "Main", f"{split}.txt")) as f:
            self.ids = [ln.strip() for ln in f if ln.strip()]
        self.transforms = transforms

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        img_id = self.ids[i]
        img = Image.open(os.path.join(self.root, "JPEGImages", f"{img_id}.jpg")).convert("RGB")
        tree = ET.parse(os.path.join(self.root, "Annotations", f"{img_id}.xml"))

        boxes, labels = [], []
        for obj in tree.findall("object"):
            name = obj.findtext("name")
            if name not in self.cls2idx:
                continue
            b = obj.find("bndbox")
            boxes.append([float(b.findtext("xmin")), float(b.findtext("ymin")),
                          float(b.findtext("xmax")), float(b.findtext("ymax"))])
            labels.append(self.cls2idx[name])

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([i]),
        }
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target
