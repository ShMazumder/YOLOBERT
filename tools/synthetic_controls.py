"""synthetic_controls.py — construct-validity experiments for OVD-Diagnose.

Each control perturbs one factor and checks that only its intended axis moves:

  temperature : rescale detection confidences -> C_ece should change while AP
                (hence S) and L are invariant. POST-HOC, no model/GPU needed.
  vocab       : enlarge the prompt vocabulary with distractor classes -> S should
                rise monotonically while L is untouched. Needs an OVD adapter.
  blur        : Gaussian-blur inputs -> L should rise (SAM localizes less) while
                the perturbation is text-independent. Needs the SAM adapter.

    # instant, from saved results:
    python tools/synthetic_controls.py --control temperature \
        --ann data/aerial/annotations/instances_val.json \
        --results runs/diag/aerial/owlv2/results_global.json \
        --out runs/controls/aerial_owlv2_temperature.csv

    # re-inference controls:
    python tools/synthetic_controls.py --control vocab  --ann ... --imgs ... \
        --model owlv2 --weights ... --out ... [--limit N]
    python tools/synthetic_controls.py --control blur   --ann ... --imgs ... \
        --sam_weights mobile_sam.pt --out ... [--limit N]
"""
import argparse
import csv
import json
import math
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ----------------------------------------------------------------- temperature
def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def _scale_scores(results, T):
    """Temperature-scale confidences via logits: p' = sigmoid(logit(p)/T).
    Monotonic in p, so detection RANKING (and thus AP) is preserved."""
    out = []
    eps = 1e-6
    for d in results:
        p = min(max(d["score"], eps), 1 - eps)
        logit = math.log(p / (1 - p))
        d2 = dict(d)
        d2["score"] = _sigmoid(logit / T)
        out.append(d2)
    return out


def temperature_control(ann, results, temps, out_csv):
    from tools.ovd_diagnose import average_precision, calibration_errors, _eval_img_ids
    img_ids = _eval_img_ids(results)
    rows = []
    for T in temps:
        scaled = _scale_scores(results, T)
        ce, _ = calibration_errors(ann, scaled)
        ap = average_precision(ann, scaled, img_ids)
        rows.append({"T": T, "C_ece": round(ce, 4), "AP": round(ap, 4)})
        print(f"  T={T:>4}: C_ece={ce:.4f}  AP={ap:.4f}")
    _write(out_csv, rows, ["T", "C_ece", "AP"])
    ap_span = max(r["AP"] for r in rows) - min(r["AP"] for r in rows)
    ce_span = max(r["C_ece"] for r in rows) - min(r["C_ece"] for r in rows)
    print(f"\nPASS if C_ece varies ({ce_span:.4f}) while AP is ~flat ({ap_span:.4f}).")


# ----------------------------------------------------------------- vocab
def vocab_control(ann, imgs, model, extra_sizes, out_csv, limit=0, device=None):
    from pycocotools.coco import COCO
    from tools.ovd_diagnose import average_precision
    coco = COCO(ann)
    all_names = [c["name"] for c in coco.loadCats(sorted(coco.getCatIds()))]
    name2catid = {c["name"]: c["id"] for c in coco.loadCats(sorted(coco.getCatIds()))}
    img_ids = [i for i in sorted(coco.imgs.keys())
               if os.path.exists(os.path.join(imgs, coco.loadImgs(i)[0]["file_name"]))]
    if limit:
        img_ids = img_ids[:limit]

    # oracle AP (present-only) is the fixed reference; S = AP_oracle - AP(|V|)
    rng = random.Random(0)
    rows = []
    for k in extra_sizes:
        results, oracle_results = [], []
        for img_id in img_ids:
            info = coco.loadImgs(img_id)[0]
            path = os.path.join(imgs, info["file_name"])
            present = sorted({a["category_id"] for a in coco.loadAnns(coco.getAnnIds(imgIds=img_id))})
            pnames = [coco.loadCats([c])[0]["name"] for c in present]
            if not pnames:
                continue
            others = [n for n in all_names if n not in pnames]
            distract = rng.sample(others, min(k, len(others)))
            vocab = pnames + distract
            for box, sc, ci in model.predict(path, vocab):
                nm = vocab[ci]
                if nm in name2catid:
                    results.append({"image_id": img_id, "category_id": name2catid[nm],
                                    "bbox": box, "score": sc})
            if k == extra_sizes[0]:                     # oracle once (k-independent)
                for box, sc, ci in model.predict(path, pnames):
                    oracle_results.append({"image_id": img_id,
                                           "category_id": name2catid[pnames[ci]],
                                           "bbox": box, "score": sc})
        if k == extra_sizes[0]:
            ap_oracle = average_precision(ann, oracle_results, img_ids)
        ap = average_precision(ann, results, img_ids)
        s = ap_oracle - ap
        rows.append({"n_distractors": k, "AP": round(ap, 4), "S": round(s, 4)})
        print(f"  +{k:>3} distractors: AP={ap:.4f}  S={s:.4f}")
    _write(out_csv, rows, ["n_distractors", "AP", "S"])
    print("\nPASS if S rises monotonically with distractor count.")


# ----------------------------------------------------------------- blur
def blur_control(ann, imgs, sam, sigmas, out_csv, limit=0):
    import numpy as np
    from PIL import Image, ImageFilter
    from pycocotools.coco import COCO
    from tools.ovd_diagnose import agnostic_recall
    coco = COCO(ann)
    agn_catid = sorted(coco.getCatIds())[0]
    img_ids = [i for i in sorted(coco.imgs.keys())
               if os.path.exists(os.path.join(imgs, coco.loadImgs(i)[0]["file_name"]))]
    if limit:
        img_ids = img_ids[:limit]

    rows = []
    tmp = "/tmp/_blur.png"
    for sig in sigmas:
        results = []
        for img_id in img_ids:
            info = coco.loadImgs(img_id)[0]
            path = os.path.join(imgs, info["file_name"])
            if sig > 0:
                Image.open(path).convert("RGB").filter(
                    ImageFilter.GaussianBlur(sig)).save(tmp)
                src = tmp
            else:
                src = path
            for box, sc, _ in sam.predict(src):
                results.append({"image_id": img_id, "category_id": agn_catid,
                                "bbox": box, "score": sc})
        ar = agnostic_recall(ann, results, img_ids)
        rows.append({"blur_sigma": sig, "AR_SAM": round(ar, 4), "L": round(1 - ar, 4)})
        print(f"  sigma={sig:>4}: AR_SAM={ar:.4f}  L={1-ar:.4f}")
    _write(out_csv, rows, ["blur_sigma", "AR_SAM", "L"])
    print("\nPASS if L rises monotonically with blur.")


# ----------------------------------------------------------------- io + cli
def _write(path, rows, keys):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"wrote {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", required=True, choices=["temperature", "vocab", "blur"])
    ap.add_argument("--ann", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--results", help="saved results_global.json (temperature)")
    ap.add_argument("--imgs", help="image dir (vocab/blur)")
    ap.add_argument("--model", default="owlv2")
    ap.add_argument("--weights", default="google/owlv2-base-patch16-ensemble")
    ap.add_argument("--sam_weights", default="mobile_sam.pt")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    if args.control == "temperature":
        results = json.load(open(args.results))
        temperature_control(args.ann, results, [0.5, 1.0, 1.5, 2.0, 3.0, 5.0], args.out)
    elif args.control == "vocab":
        from models.ovd import build_adapter
        model = build_adapter(args.model, weights=args.weights, device=args.device)
        vocab_control(args.ann, args.imgs, model, [0, 5, 20, 40, 79], args.out,
                      args.limit, args.device)
    elif args.control == "blur":
        from models.ovd import build_adapter
        sam = build_adapter("sam", weights=args.sam_weights, device=args.device)
        blur_control(args.ann, args.imgs, sam, [0, 1, 2, 4, 8], args.out, args.limit)


if __name__ == "__main__":
    main()
