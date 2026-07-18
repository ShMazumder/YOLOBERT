"""run_agnostic.py — generate class-agnostic proposals (SAM) for the L axis.

Runs a frozen proposal model over the SAME images as run_ovd, writes
results_agnostic.json into the run dir. Then recompute the fingerprint with a
true localization signal:

    python tools/run_agnostic.py \
        --ann data/aerial/annotations/instances_val.json \
        --imgs data/aerial/images \
        --model sam --weights mobile_sam.pt \
        --out runs/diag/aerial_yoloworld --limit 200

    # then, in python:
    from tools.ovd_diagnose import diagnose
    A = json.load(open('runs/diag/aerial_yoloworld/results_agnostic.json'))
    diagnose(ann, G, O, results_agnostic=A)   # L now from SAM
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True)
    ap.add_argument("--imgs", required=True)
    ap.add_argument("--model", default="sam")
    ap.add_argument("--weights", default="mobile_sam.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--imgz", type=int, default=1024)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    from pycocotools.coco import COCO
    from models.ovd import build_adapter

    coco = COCO(args.ann)
    agnostic_catid = sorted(coco.getCatIds())[0]      # class-agnostic eval -> any id
    img_ids = sorted(coco.imgs.keys())
    if args.limit:
        img_ids = img_ids[:args.limit]

    model = build_adapter(args.model, weights=args.weights, device=args.device, imgsz=args.imgz)

    results = []
    for k, img_id in enumerate(img_ids):
        info = coco.loadImgs(img_id)[0]
        path = os.path.join(args.imgs, info["file_name"])
        for box, sc, _ in model.predict(path):
            results.append({"image_id": img_id, "category_id": agnostic_catid,
                            "bbox": box, "score": sc})
        if (k + 1) % 50 == 0:
            print(f"  {k+1}/{len(img_ids)} images")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    with open(out / "results_agnostic.json", "w") as f:
        json.dump(results, f)
    print(f"wrote {len(results)} proposals -> {out}/results_agnostic.json")


if __name__ == "__main__":
    main()
