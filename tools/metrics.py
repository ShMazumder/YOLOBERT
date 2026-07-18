"""metrics.py — task evaluators for detection, segmentation, pose, classification.

Exact metric names match CLAUDE.md / paper conventions:
  detection/instseg/keypoints : AP, AP50, AP75, AP_S/M/L   (pycocotools)
  semantic segmentation        : mIoU, mAcc, aAcc          (confusion matrix)
  classification               : top1, top5

Dispatch by config:
    metric: coco        # + metric_iou_type: bbox|segm|keypoints
    metric: semseg      # + num_classes, ignore_index
    metric: cls         # + topk: [1,5]

    from metrics import build_metric
    m = build_metric(cfg)
    m.update(preds, targets)      # call per batch
    results = m.compute()         # dict of floats

Never fabricate numbers — every value comes from accumulated predictions.
"""
from collections import defaultdict

import numpy as np


# =====================================================================
# Semantic segmentation — mIoU / mAcc / aAcc via a K×K confusion matrix.
# =====================================================================
class SemSegMetric:
    def __init__(self, num_classes, ignore_index=255, **kw):
        self.n = num_classes
        self.ignore = ignore_index
        self.reset()

    def reset(self):
        self.conf = np.zeros((self.n, self.n), dtype=np.int64)

    def update(self, pred, target):
        """pred,target: int label maps (np or torch), same shape. Class ids."""
        pred = _to_numpy(pred).reshape(-1)
        target = _to_numpy(target).reshape(-1)
        keep = target != self.ignore
        pred, target = pred[keep], target[keep]
        # bincount of true*n+pred -> flat confusion.
        idx = target.astype(np.int64) * self.n + pred.astype(np.int64)
        self.conf += np.bincount(idx, minlength=self.n ** 2).reshape(self.n, self.n)

    def compute(self):
        c = self.conf.astype(np.float64)
        tp = np.diag(c)
        union = c.sum(1) + c.sum(0) - tp
        iou = tp / np.maximum(union, 1e-9)
        acc = tp / np.maximum(c.sum(1), 1e-9)          # per-class recall
        return {
            "mIoU": float(np.nanmean(iou) * 100),
            "mAcc": float(np.nanmean(acc) * 100),
            "aAcc": float(tp.sum() / max(c.sum(), 1e-9) * 100),
        }


# =====================================================================
# Classification — top-k accuracy.
# =====================================================================
class ClsMetric:
    def __init__(self, topk=(1, 5), **kw):
        self.topk = tuple(topk)
        self.reset()

    def reset(self):
        self.correct = defaultdict(int)
        self.total = 0

    def update(self, logits, target):
        """logits: (N,C) scores. target: (N,) int labels."""
        logits = _to_numpy(logits)
        target = _to_numpy(target).reshape(-1)
        maxk = max(self.topk)
        topk_idx = np.argsort(-logits, axis=1)[:, :maxk]     # (N,maxk)
        hit = topk_idx == target[:, None]
        for k in self.topk:
            self.correct[k] += int(hit[:, :k].any(1).sum())
        self.total += target.shape[0]

    def compute(self):
        return {f"top{k}": 100.0 * self.correct[k] / max(self.total, 1)
                for k in self.topk}


# =====================================================================
# COCO — detection / instance-seg / keypoints via pycocotools.
# =====================================================================
class CocoMetric:
    """Accumulate COCO-format results, then score against a COCO gt json.

    iou_type: 'bbox' | 'segm' | 'keypoints'.
    Feed predictions as COCO result dicts:
      det/seg : {image_id, category_id, bbox=[x,y,w,h] or segmentation, score}
      keypts  : {image_id, category_id, keypoints=[...3K], score}
    """
    _KEYS = ["AP", "AP50", "AP75", "AP_S", "AP_M", "AP_L"]

    def __init__(self, ann_file, iou_type="bbox", **kw):
        self.ann_file = ann_file
        self.iou_type = iou_type
        self.reset()

    def reset(self):
        self.results = []

    def update(self, preds, targets=None):
        """preds: list of COCO result dicts (see class doc)."""
        self.results.extend(preds)

    def compute(self):
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        coco_gt = COCO(self.ann_file)
        if not self.results:
            return {k: 0.0 for k in self._KEYS}
        coco_dt = coco_gt.loadRes(self.results)
        ev = COCOeval(coco_gt, coco_dt, self.iou_type)
        ev.evaluate(); ev.accumulate(); ev.summarize()
        s = ev.stats                                   # 12-vector, standard order
        return {
            "AP":   float(s[0] * 100),
            "AP50": float(s[1] * 100),
            "AP75": float(s[2] * 100),
            "AP_S": float(s[3] * 100),
            "AP_M": float(s[4] * 100),
            "AP_L": float(s[5] * 100),
        }


# =====================================================================
# dispatch
# =====================================================================
_REGISTRY = {"semseg": SemSegMetric, "cls": ClsMetric, "coco": CocoMetric}


def build_metric(cfg):
    name = cfg.get("metric", "cls")
    if name not in _REGISTRY:
        raise KeyError(f"unknown metric '{name}'. options: {list(_REGISTRY)}")
    kwargs = {k[len("metric_"):]: v for k, v in cfg.items() if k.startswith("metric_")}
    if name == "semseg":
        kwargs.setdefault("num_classes", cfg.get("num_classes", 2))
    if name == "coco":
        kwargs.setdefault("ann_file", cfg["ann_file"])       # required
    return _REGISTRY[name](**kwargs)


def _to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


if __name__ == "__main__":
    # self-test the pure-numpy metrics (no pycocotools needed).
    m = SemSegMetric(num_classes=3)
    pred = np.array([[0, 1], [2, 1]]); gt = np.array([[0, 1], [2, 2]])
    m.update(pred, gt)
    print("semseg:", m.compute())

    c = ClsMetric(topk=(1, 2))
    logits = np.array([[0.1, 0.9, 0.0], [0.8, 0.1, 0.1]])
    c.update(logits, np.array([1, 2]))
    print("cls:", c.compute())      # top1=50, top2=100
