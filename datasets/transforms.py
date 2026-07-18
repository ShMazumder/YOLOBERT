"""transforms.py — box-aware detection transforms (no torchvision.transforms.v2 dep).

Each transform is callable (img: PIL.Image, target: dict) -> (img, target),
keeping 'boxes' (xyxy) in sync with geometric ops. Compose them, then pass the
result as cfg['_transforms'] so detection batches share a size and stack.

    from datasets.transforms import Compose, Resize, RandomHFlip, ToTensor, Normalize, default_train, default_val
    cfg['_transforms'] = default_train(size=800)

ImageNet normalization stats by default — override for your data.
"""
import random

import torch
import torch.nn.functional as F
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)
        return img, target


class Resize:
    """Resize PIL image to (size,size) and scale boxes. Square for easy stacking."""
    def __init__(self, size=800):
        self.size = (size, size) if isinstance(size, int) else tuple(size)

    def __call__(self, img, target):
        w0, h0 = img.size
        img = img.resize(self.size, Image.BILINEAR)
        if "boxes" in target and target["boxes"].numel():
            sw = self.size[0] / w0
            sh = self.size[1] / h0
            b = target["boxes"].clone()
            b[:, [0, 2]] *= sw
            b[:, [1, 3]] *= sh
            target["boxes"] = b
        return img, target


class RandomHFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            w, _ = img.size
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if "boxes" in target and target["boxes"].numel():
                b = target["boxes"].clone()
                b[:, [0, 2]] = w - b[:, [2, 0]]        # flip xmin/xmax
                target["boxes"] = b
        return img, target


class ToTensor:
    """PIL -> float tensor CxHxW in [0,1]. numpy path (fast; avoids per-pixel loop)."""
    def __call__(self, img, target):
        import numpy as np
        arr = np.asarray(img, dtype=np.float32)          # HxWxC
        t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        return t / 255.0, target


class Normalize:
    def __init__(self, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def __call__(self, img, target):
        return (img - self.mean) / self.std, target


def default_train(size=800):
    return Compose([Resize(size), RandomHFlip(0.5), ToTensor(), Normalize()])


def default_val(size=800):
    return Compose([Resize(size), ToTensor(), Normalize()])
