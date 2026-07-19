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
def _coco_eval(gt_json, results, use_cats=True, img_ids=None):
    """Return the 12-vector of COCOeval stats for bbox. results: list of dicts.

    img_ids: restrict evaluation to these images. CRITICAL when only a subset of
    the dataset was run (e.g. --limit) — otherwise recall/AP are divided by the
    FULL dataset GT and collapse to ~0.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    coco_gt = COCO(gt_json) if isinstance(gt_json, str) else gt_json
    if not results:
        return np.zeros(12)
    coco_dt = coco_gt.loadRes(results)
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    if img_ids is not None:
        ev.params.imgIds = sorted(img_ids)    # score only the images actually run
    if not use_cats:
        ev.params.useCats = 0                 # class-agnostic: localization only
    ev.evaluate(); ev.accumulate(); ev.summarize()
    return ev.stats                            # [AP, AP50, AP75, AP_S, AP_M, AP_L, AR1, AR10, AR100, AR_S, AR_M, AR_L]


def _eval_img_ids(results):
    """Images actually evaluated = those present in the detections."""
    return sorted({d["image_id"] for d in results})


def average_precision(gt_json, results, img_ids=None):
    return float(_coco_eval(gt_json, results, use_cats=True, img_ids=img_ids)[0])


def agnostic_recall(gt_json, results, img_ids=None):
    """Class-agnostic AR@100 — pure localization (ignores category labels)."""
    return float(_coco_eval(gt_json, results, use_cats=False, img_ids=img_ids)[8])


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
    eval_imgs = set(_eval_img_ids(results_global))          # restrict to run images
    n_gt = len(coco_gt.getAnnIds(imgIds=list(eval_imgs)))
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
def diagnose(gt_json, results_global, results_oracle, results_agnostic=None,
             fixed_score_thresh=0.25, eps=1e-6):
    """Compute the (L, S, C) failure fingerprint for one (domain, model).

    Localization capacity is measured as class-agnostic recall of the ORACLE-mode
    detections (correct class names, labels stripped via useCats=0). This isolates
    "can the model box the objects" from vocabulary confusion WITHOUT relying on a
    vague single-'object' prompt. Pass results_agnostic to override with a dedicated
    class-agnostic proposal source (e.g. SAM).
    """
    from pycocotools.coco import COCO
    coco_gt = COCO(gt_json) if isinstance(gt_json, str) else gt_json

    img_ids = _eval_img_ids(results_global)     # only score the images actually run
    ap_global = average_precision(coco_gt, results_global, img_ids)
    ap_oracle = average_precision(coco_gt, results_oracle, img_ids)
    # L from an external proposer (SAM) if given, else oracle-labelfree.
    ar_source = results_agnostic if results_agnostic else results_oracle
    ar_agnostic = agnostic_recall(coco_gt, ar_source, img_ids)
    # detector-INTRINSIC localizability: the model's OWN oracle boxes, labels stripped.
    # Answers "what if the detector localizes but SAM misses?" -- proposer-independent.
    ar_det = agnostic_recall(coco_gt, results_oracle, img_ids)
    c_ece, c_thr = calibration_errors(coco_gt, results_global, fixed_score_thresh)

    return {
        "L": 1.0 - ar_agnostic,                  # proposer-based (SAM) L
        "L_det": 1.0 - ar_det,                    # detector-intrinsic L (no SAM)
        "S": ap_oracle - ap_global,
        "S_norm": (ap_oracle - ap_global) / max(ap_oracle, eps),
        "C_ece": c_ece,
        "C_thr": c_thr,
        "AP_global": ap_global,
        "AP_oracle": ap_oracle,
        "AR_agnostic": ar_agnostic,
        "AR_det": ar_det,
        "L_source": "agnostic_mode" if results_agnostic else "oracle_labelfree",
    }


# ----------------------------------------------------------------- anchor + sanity
def ioa_f1(gt_json, results, ioa_thr=0.7, score_thr=0.1, per_class=True):
    """Paper-1-style anchor: Intersection-over-(gt)-Area matching + F1.

    Reproduces the metric used by Tsourveloudis'26 so we can validate the harness
    against their published aerial numbers (e.g. ~0.53 F1 on DIOR) before trusting
    the COCO-AP-based axes. IoA is lenient on small/dense aerial objects vs IoU.
    """
    from pycocotools.coco import COCO
    coco = COCO(gt_json) if isinstance(gt_json, str) else gt_json
    eval_imgs = set(_eval_img_ids(results))              # only images actually run
    gt_by = {}
    for a in coco.loadAnns(coco.getAnnIds(imgIds=list(eval_imgs))):
        key = (a["image_id"], a["category_id"]) if per_class else a["image_id"]
        gt_by.setdefault(key, []).append(a["bbox"])
    n_gt = sum(len(v) for v in gt_by.values())

    dets = sorted((d for d in results if d["score"] >= score_thr), key=lambda d: -d["score"])
    used, tp, fp = {}, 0, 0
    for d in dets:
        key = (d["image_id"], d["category_id"]) if per_class else d["image_id"]
        gts = gt_by.get(key, [])
        best, bj = 0.0, -1
        dx, dy, dw, dh = d["bbox"]
        for j, g in enumerate(gts):
            if j in used.get(key, set()):
                continue
            gx, gy, gw, gh = g
            ix = max(0.0, min(dx + dw, gx + gw) - max(dx, gx))
            iy = max(0.0, min(dy + dh, gy + gh) - max(dy, gy))
            inter = ix * iy
            ioa = inter / (gw * gh) if gw * gh > 0 else 0.0
            if ioa > best:
                best, bj = ioa, j
        if best >= ioa_thr and bj >= 0:
            used.setdefault(key, set()).add(bj); tp += 1
        else:
            fp += 1
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / n_gt if n_gt else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"F1": f1, "precision": prec, "recall": rec, "TP": tp, "FP": fp, "n_gt": n_gt}


def bootstrap_ci(gt_json, results, metric="ioa_f1", n_boot=1000, seed=0,
                 ioa_thr=0.7, score_thr=0.25):
    """Image-level bootstrap CI for a per-image metric. Resamples the evaluated
    images with replacement n_boot times. metric: 'ioa_f1' or 'recall_agnostic'.

    Returns {'mean','lo','hi'} (95% percentile interval). Cheap: uses the greedy
    matcher, not COCOeval, so 1000 resamples run in seconds.
    """
    from pycocotools.coco import COCO
    coco = COCO(gt_json) if isinstance(gt_json, str) else gt_json
    rng = np.random.default_rng(seed)
    img_ids = _eval_img_ids(results)

    # pre-index per image for fast resampling
    res_by_img = {}
    for d in results:
        res_by_img.setdefault(d["image_id"], []).append(d)
    gt_by_img = {}
    for a in coco.loadAnns(coco.getAnnIds(imgIds=img_ids)):
        gt_by_img.setdefault(a["image_id"], []).append(a)

    def _score(sample_ids):
        # class-agnostic (recall) or class-aware IoA-F1 over the sampled images
        per_class = metric == "ioa_f1"
        gt_by, used = {}, {}
        n_gt = 0
        for iid in sample_ids:
            for a in gt_by_img.get(iid, []):
                key = (iid, a["category_id"]) if per_class else iid
                gt_by.setdefault(key, []).append(a["bbox"]); n_gt += 1
        dets = []
        for iid in sample_ids:
            dets.extend(res_by_img.get(iid, []))
        dets = sorted((d for d in dets if d["score"] >= score_thr), key=lambda d: -d["score"])
        tp = fp = 0
        for d in dets:
            key = (d["image_id"], d["category_id"]) if per_class else d["image_id"]
            best, bj = 0.0, -1
            for j, g in enumerate(gt_by.get(key, [])):
                if j in used.get(key, set()):
                    continue
                dx, dy, dw, dh = d["bbox"]; gx, gy, gw, gh = g
                ix = max(0.0, min(dx+dw, gx+gw)-max(dx, gx))
                iy = max(0.0, min(dy+dh, gy+gh)-max(dy, gy))
                inter = ix*iy; ioa = inter/(gw*gh) if gw*gh > 0 else 0.0
                if ioa > best:
                    best, bj = ioa, j
            if best >= ioa_thr and bj >= 0:
                used.setdefault(key, set()).add(bj); tp += 1
            else:
                fp += 1
        if metric == "recall_agnostic":
            return tp / n_gt if n_gt else 0.0
        prec = tp/(tp+fp) if (tp+fp) else 0.0
        rec = tp/n_gt if n_gt else 0.0
        return 2*prec*rec/(prec+rec) if (prec+rec) else 0.0

    vals = np.array([_score(rng.choice(img_ids, size=len(img_ids), replace=True))
                     for _ in range(n_boot)])
    return {"mean": float(vals.mean()),
            "lo": float(np.percentile(vals, 2.5)),
            "hi": float(np.percentile(vals, 97.5))}


def sanity_check(gt_json, results, n_images=5):
    """Detect coordinate/format bugs: per image, report #gt, #pred, best IoU stats.

    If many predictions exist but best-IoU is ~0, boxes are misaligned (wrong
    coordinate space, xywh/xyxy mixup, or resized-vs-original mismatch) — NOT a
    domain-gap problem.
    """
    from pycocotools.coco import COCO
    coco = COCO(gt_json) if isinstance(gt_json, str) else gt_json
    by_img = {}
    for d in results:
        by_img.setdefault(d["image_id"], []).append(d)
    img_ids = sorted(coco.imgs.keys())[:n_images]
    print(f"{'img':>12} {'#gt':>5} {'#pred':>6} {'meanBestIoU':>12} {'%gt IoU>0.5':>12}")
    for iid in img_ids:
        gts = [a["bbox"] for a in coco.loadAnns(coco.getAnnIds(imgIds=iid))]
        preds = by_img.get(iid, [])
        best_ious = []
        for g in gts:
            bi = max((_iou_xywh(p["bbox"], g) for p in preds), default=0.0)
            best_ious.append(bi)
        mean_bi = np.mean(best_ious) if best_ious else 0.0
        frac = np.mean([b > 0.5 for b in best_ious]) if best_ious else 0.0
        print(f"{iid:>12} {len(gts):>5} {len(preds):>6} {mean_bi:>12.3f} {frac:>12.2%}")


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
