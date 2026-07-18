"""run_all.py — headless multi-model OVD-Diagnose driver for one domain.

Runs SAM once (shared class-agnostic L source), then every OVD model in Global +
Oracle modes, computes each (L,S,C) fingerprint, and writes a consolidated
fingerprints.{json,csv} to the out dir. Per-model try/except so one failure
(OOM, bad weights) does not abort the whole run — ideal for Kaggle Commit.

    python tools/run_all.py \
        --ann data/aerial/annotations/instances_val.json \
        --imgs data/aerial/images \
        --out runs/diag/aerial --limit 200 --device cuda:0

Models default to a T4-friendly set; override with --models "name:weights,...".
"""
import argparse
import csv
import json
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_MODELS = [
    ("yoloworld", "yolov8s-world.pt"),
    ("owlv2", "google/owlv2-base-patch16-ensemble"),
    ("groundingdino", "IDEA-Research/grounding-dino-tiny"),
]


def _run_modes(model, coco, img_ids, imgs_dir, names, name2catid, agnostic=False, agn_catid=None):
    """Return {'global':[...], 'oracle':[...]} (or {'agnostic':[...]} if agnostic)."""
    out = {"agnostic": []} if agnostic else {"global": [], "oracle": []}
    for k, img_id in enumerate(img_ids):
        info = coco.loadImgs(img_id)[0]
        path = os.path.join(imgs_dir, info["file_name"])
        if agnostic:
            for box, sc, _ in model.predict(path):
                out["agnostic"].append({"image_id": img_id, "category_id": agn_catid,
                                        "bbox": box, "score": sc})
        else:
            for box, sc, ci in model.predict(path, names):
                out["global"].append({"image_id": img_id,
                                      "category_id": name2catid[names[ci]],
                                      "bbox": box, "score": sc})
            present = sorted({a["category_id"] for a in coco.loadAnns(coco.getAnnIds(imgIds=img_id))})
            pnames = [coco.loadCats([c])[0]["name"] for c in present]
            if pnames:
                for box, sc, ci in model.predict(path, pnames):
                    out["oracle"].append({"image_id": img_id,
                                         "category_id": name2catid[pnames[ci]],
                                         "bbox": box, "score": sc})
        if (k + 1) % 100 == 0:
            print(f"    {k+1}/{len(img_ids)}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True)
    ap.add_argument("--imgs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--models", default=None, help='comma list "name:weights,..."')
    ap.add_argument("--sam_weights", default="mobile_sam.pt", help='"none" to skip SAM')
    ap.add_argument("--fixed_thresh", type=float, default=0.25)
    args = ap.parse_args()

    from pycocotools.coco import COCO
    from models.ovd import build_adapter
    from tools.ovd_diagnose import diagnose

    coco = COCO(args.ann)
    cat_ids = sorted(coco.getCatIds())
    names = [c["name"] for c in coco.loadCats(cat_ids)]
    name2catid = {c["name"]: c["id"] for c in coco.loadCats(cat_ids)}
    img_ids = sorted(coco.imgs.keys())
    if args.limit:
        img_ids = img_ids[:args.limit]
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    models = DEFAULT_MODELS
    if args.models:
        models = [tuple(m.split(":", 1)) for m in args.models.split(",")]

    # --- SAM once (shared L source) ---
    agn = None
    if args.sam_weights.lower() != "none":
        try:
            print("[SAM] proposals ...")
            sam = build_adapter("sam", weights=args.sam_weights, device=args.device)
            agn = _run_modes(sam, coco, img_ids, args.imgs, names, name2catid,
                             agnostic=True, agn_catid=cat_ids[0])["agnostic"]
            json.dump(agn, open(out_dir / "results_agnostic.json", "w"))
            del sam
        except Exception:
            print("[SAM] failed:\n" + traceback.format_exc())

    # --- each OVD model ---
    rows = []
    for name, weights in models:
        print(f"[{name}] {weights}")
        try:
            m = build_adapter(name, weights=weights, device=args.device)
            res = _run_modes(m, coco, img_ids, args.imgs, names, name2catid)
            mdir = out_dir / name; mdir.mkdir(exist_ok=True)
            for mode, r in res.items():
                json.dump(r, open(mdir / f"results_{mode}.json", "w"))
            fp = diagnose(args.ann, res["global"], res["oracle"],
                          results_agnostic=agn, fixed_score_thresh=args.fixed_thresh)
            fp["model"] = name
            json.dump(fp, open(mdir / "fingerprint.json", "w"), indent=2)
            rows.append(fp)
            print(f"  -> {name}: L={fp['L']:.3f} S_norm={fp['S_norm']:.3f} "
                  f"C_ece={fp['C_ece']:.3f} AP_g={fp['AP_global']:.3f} AP_o={fp['AP_oracle']:.3f}")
            del m
        except Exception:
            print(f"[{name}] FAILED:\n" + traceback.format_exc())

    # --- consolidated table ---
    if rows:
        keys = ["model", "L", "S", "S_norm", "C_ece", "C_thr",
                "AP_global", "AP_oracle", "AR_agnostic", "L_source"]
        with open(out_dir / "fingerprints.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)
        json.dump(rows, open(out_dir / "fingerprints.json", "w"), indent=2)
        print(f"\nwrote {len(rows)} fingerprints -> {out_dir}/fingerprints.csv")


if __name__ == "__main__":
    main()
