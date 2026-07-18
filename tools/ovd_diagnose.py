"""ovd_diagnose.py — TIDE-for-OVD: decompose open-vocabulary detector failure into
Localization (L), Semantic-confusion (S), and Calibration (C).

This is the intellectual core of the OVD-Diagnose benchmark (see paper/DESIGN.md).
It is model-agnostic: it consumes a COCO ground-truth json plus COCO-format detection
results produced by ANY OVD model in three prompting modes:

    global   : model prompted with the FULL domain vocabulary (all N classes)
    oracle   : model prompted with ONLY the classes present in each image
    agnostic : class-agnostic / single-"object" prompt (localization capacity)

    from ovd_diagnose import diagnose
    fp = diagnose(gt_json, results_global, results_oracle, results_agnostic,
                  fixed_score_thresh=0.25)
    # -> {'L':.., 'S':.., 'S_norm':.., 'C_ece':.., 'C_thr':.., 'AP_global':.., 'AP_oracle':.., 'AR_agnostic':..}

Axes (paper/DESIGN.md Sec 3):
    L      = 1 - AR_agnostic                         # can it box objects at all?
    S      = AP_oracle - AP_global                   # error caused purely by vocabulary
    S_norm = (AP_oracle - AP_global) / AP_oracle     # fraction of achievable AP lost to confusion
    C_ece  = expected calibration error of detection confidences
    C_thr  = F1_best_threshold - F1_fixed_threshold  # operating-point brittleness

Deps: numpy, pycocotools.
"""
import numpy as np


# ----------------------------------------------------------------- AP / AR (COCO)
def _coco_eval(gt_json, results, use_cats=True):
    """Return the 12-vector of COCOeval stats for bbox. results: list of dicts."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    coco_gt = COCO(gt_json) if isinstance(gt_json, str) else gt_json
    if not results:
        return np.zeros(12)
    coco_dt = coco_gt.loadRes(results)
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    if not use_cats:
        ev.params.useCats = 0                 # class-agnostic: localization only
    ev.evaluate(); ev.accumulate(); ev.summarize()
    return ev.stats                            # [AP, AP50, AP75, AP_S, AP_M, AP_L, AR1, AR10, AR100, AR_S, AR_M, AR_L]


def average_precision(gt_json, results):
    return float(_coco_eval(gt_json, results, use_cats=True)[0])


def agnostic_recall(gt_json, results):
    """Class-agnostic AR@100 — pure localization (ignores category labels)."""
    return float(_coco_eval(gt_json, results, use_cats=False)[8])


# ----------------------------------------------------------------- calibration
def _iou_xywh(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _match_tps(gt_json, results, iou_thr=0.5):
    """Greedy per-image per-category matching. Returns (scores, hits) arrays.

    hits[i] = 1 if detection i is a true positive at iou_thr else 0.
    """
    from pycocotools.coco import COCO
    coco_gt = COCO(gt_json) if isinstance(gt_json, str) else gt_json

    # index gt boxes by (image_id, category_id)
    gt_by = {}
    for ann in coco_gt.loadAnns(coco_gt.getAnnIds()):
        gt_by.setdefault((ann["image_id"], ann["category_id"]), []).append(ann["bbox"])

    # sort detections by descending score, match greedily
    dets = sorted(results, key=lambda d: -d["score"])
    used = {}                                  # (img,cat) -> set of matched gt indices
    scores, hits = [], []
    for d in dets:
        key = (d["image_id"], d["category_id"])
        scores.append(d["score"])
        gts = gt_by.get(key, [])
        best_iou, best_j = 0.0, -1
        for j, gb in enumerate(gts):
            if j in used.get(key, set()):
                continue
            i = _iou_xywh(d["bbox"], gb)
            if i > best_iou:
                best_iou, best_j = i, j
        if best_iou >= iou_thr and best_j >= 0:
            used.setdefault(key, set()).add(best_j)
            hits.append(1)
        else:
            hits.append(0)
    return np.asarray(scores, float), np.asarray(hits, float)


def expected_calibration_error(scores, hits, n_bins=15):
    """ECE: |accuracy - confidence| averaged over confidence bins, weighted by count."""
    if len(scores) == 0:
        return 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece, N = 0.0, len(scores)
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (scores > lo) & (scores <= hi)
        if m.sum() == 0:
            continue
        conf = scores[m].mean()
        acc = hits[m].mean()
        ece += (m.sum() / N) * abs(acc - conf)
    return float(ece)


def _f1_at(scores, hits, n_gt, thr):
    keep = scores >= thr
    tp = hits[keep].sum()
    fp = keep.sum() - tp
    fn = n_gt - tp
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def calibration_errors(gt_json, results_global, fixed_score_thresh=0.25, iou_thr=0.5):
    """C_ece + C_thr from GLOBAL-mode detections (the real operating condition)."""
    from pycocotools.coco import COCO
    coco_gt = COCO(gt_json) if isinstance(gt_json, str) else gt_json
    n_gt = len(coco_gt.getAnnIds())
    scores, hits = _match_tps(coco_gt, results_global, iou_thr)
    ece = expected_calibration_error(scores, hits)
    if len(scores):
        grid = np.linspace(0.05, 0.95, 19)
        f1_best = max(_f1_at(scores, hits, n_gt, t) for t in grid)
        f1_fixed = _f1_at(scores, hits, n_gt, fixed_score_thresh)
        c_thr = float(f1_best - f1_fixed)
    else:
        c_thr = 0.0
    return ece, c_thr


# ----------------------------------------------------------------- top-level
def diagnose(gt_json, results_global, results_oracle, results_agnostic,
             fixed_score_thresh=0.25, eps=1e-6):
    """Compute the (L, S, C) failure fingerprint for one (domain, model)."""
    from pycocotools.coco import COCO
    coco_gt = COCO(gt_json) if isinstance(gt_json, str) else gt_json

    ap_global = average_precision(coco_gt, results_global)
    ap_oracle = average_precision(coco_gt, results_oracle)
    ar_agnostic = agnostic_recall(coco_gt, results_agnostic)
    c_ece, c_thr = calibration_errors(coco_gt, results_global, fixed_score_thresh)

    return {
        "L": 1.0 - ar_agnostic,
        "S": ap_oracle - ap_global,
        "S_norm": (ap_oracle - ap_global) / max(ap_oracle, eps),
        "C_ece": c_ece,
        "C_thr": c_thr,
        "AP_global": ap_global,
        "AP_oracle": ap_oracle,
        "AR_agnostic": ar_agnostic,
    }


# ----------------------------------------------------------------- self-test
def _selftest():
    """Offline construct-validity check on the pure-numpy pieces (no pycocotools)."""
    # ECE: perfectly calibrated -> ~0; overconfident -> >0
    s = np.array([0.9, 0.9, 0.9, 0.9])          # conf 0.9
    h = np.array([1, 1, 1, 0])                   # acc 0.75
    ece = expected_calibration_error(s, h)
    assert abs(ece - 0.15) < 1e-6, ece

    s2 = np.array([0.5, 0.5]); h2 = np.array([1, 0])  # conf .5 acc .5
    assert expected_calibration_error(s2, h2) < 1e-6

    # IoU sanity
    assert abs(_iou_xywh([0, 0, 10, 10], [0, 0, 10, 10]) - 1.0) < 1e-9
    assert _iou_xywh([0, 0, 10, 10], [100, 100, 10, 10]) == 0.0

    # F1 monotonicity: perfect detections -> F1=1 at low thr
    sc = np.array([0.9, 0.8]); hi = np.array([1, 1])
    assert abs(_f1_at(sc, hi, 2, 0.1) - 1.0) < 1e-9
    print("ovd_diagnose self-test passed (ECE, IoU, F1)")


if __name__ == "__main__":
    _selftest()
