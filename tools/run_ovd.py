"""run_ovd.py — run a frozen OVD model over a COCO-format domain in the three
diagnostic modes (Global / Oracle / Agnostic), then emit the (L,S,C) fingerprint.

    python tools/run_ovd.py \
        --ann  data/aerial/annotations/instances_val.json \
        --imgs data/aerial/images \
        --model yoloworld --weights yolov8s-world.pt \
        --out runs/diag/aerial_yoloworld

Writes results_{global,oracle,agnostic}.json + fingerprint.json to --out.
Reuses tools/ovd_diagnose.diagnose for the metrics.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True, help="COCO gt json")
    ap.add_argument("--imgs", required=True, help="image dir")
    ap.add_argument("--model", default="yoloworld")
    ap.add_argument("--weights", default="yolov8s-world.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="cap #images (debug)")
    ap.add_argument("--fixed_thresh", type=float, default=0.25)
    ap.add_argument("--imgz", type=int, default=800, help="image size for model inference")
    ap.add_argument("--device", default=None, help="cuda device or cpu")
    args = ap.parse_args()

    from pycocotools.coco import COCO
    from models.ovd.base import ADAPTERS, build_adapter
    from models.ovd import yoloworld  # Force parsing of registration decorators
    from tools.ovd_diagnose import diagnose

    coco = COCO(args.ann)
    cat_ids = sorted(coco.getCatIds())
    cats = coco.loadCats(cat_ids)
    names = [c["name"] for c in cats]
    name2catid = {c["name"]: c["id"] for c in cats}
    agnostic_catid = cat_ids                     # arbitrary; eval is class-agnostic here

    img_ids = sorted(coco.getImgs().keys()) if hasattr(coco, "getImgs") else sorted(coco.imgs.keys())
    if args.limit:
        img_ids = img_ids[:args.limit]

    print(f"Verified registry targets: {list(ADAPTERS.keys())}")
    model = build_adapter(args.model, weights=args.weights, device=args.device, imgsz=args.imgz)

    res = {"global": [], "oracle": [], "agnostic": []}
    for k, img_id in enumerate(img_ids):
        info = coco.loadImgs(img_id)[0]
        path = os.path.join(args.imgs, info["file_name"])

        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        present = sorted({a["category_id"] for a in anns})
        present_names = [coco.loadCats([c])[0]["name"] for c in present]

        # --- Global: full vocabulary ---
        for box, sc, ci in model.predict(path, names):
            res["global"].append({"image_id": img_id, "category_id": name2catid[names[ci]],
                                   "bbox": box, "score": sc})
        # --- Oracle: only present classes ---
        if present_names:
            for box, sc, ci in model.predict(path, present_names):
                res["oracle"].append({"image_id": img_id,
                                      "category_id": name2catid[present_names[ci]],
                                      "bbox": box, "score": sc})
        # --- Agnostic: single 'object' prompt (localization only) ---
        for box, sc, _ in model.predict(path, ["object"]):
            res["agnostic"].append({"image_id": img_id, "category_id": agnostic_catid,
                                    "bbox": box, "score": sc})

        if (k + 1) % 50 == 0:
            print(f"  {k+1}/{len(img_ids)} images")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    for mode, r in res.items():
        with open(out / f"results_{mode}.json", "w") as f:
            json.dump(r, f)

    fp = diagnose(args.ann, res["global"], res["oracle"], res["agnostic"],
                  fixed_score_thresh=args.fixed_thresh)
    with open(out / "fingerprint.json", "w") as f:
        json.dump(fp, f, indent=2)
    print("fingerprint:", json.dumps(fp, indent=2))
    print(f"wrote -> {out}")


if __name__ == "__main__":
    main()
