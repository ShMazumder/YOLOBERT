"""vindr_to_coco.py — convert VinBigData/VinDr chest X-ray train.csv to COCO json.

Domain 2 (medical) for OVD-Diagnose. VinDr is one of the few public, multi-class,
box-annotated *specialized* detection sets (14 findings), so it exercises the
semantic-confusion axis (S) unlike single-class fish/fruit detection.

VinBigData train.csv schema:
    image_id, class_name, class_id, rad_id, x_min, y_min, x_max, y_max
    class_id 14 == "No finding" (no box) -> skipped.

    python tools/vindr_to_coco.py \
        --csv   data/medical/train.csv \
        --imgs  data/medical/images \
        --out   data/medical/annotations/instances_val.json \
        [--img_ext .png] [--scale_from_dims dims.csv]

Box coords are in the ORIGINAL DICOM pixel space. If you use resized PNGs, pass
--scale_from_dims (csv with image_id,orig_w,orig_h) OR pre-resize boxes; otherwise
boxes assume the image files match the annotation coordinate space.
"""
import argparse
import csv
import json
import os
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--imgs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--img_ext", default=".png")
    ap.add_argument("--scale_from_dims", default=None,
                    help="csv image_id,orig_w,orig_h,new_w,new_h to rescale boxes")
    args = ap.parse_args()

    from PIL import Image

    # optional rescale map: image_id -> (sx, sy)
    scale = {}
    if args.scale_from_dims:
        with open(args.scale_from_dims) as f:
            for r in csv.DictReader(f):
                sx = float(r["new_w"]) / float(r["orig_w"])
                sy = float(r["new_h"]) / float(r["orig_h"])
                scale[r["image_id"]] = (sx, sy)

    rows = list(csv.DictReader(open(args.csv)))
    # collect boxes per image, and the class name<->id map
    cats = {}
    boxes_by_img = defaultdict(list)
    orig_dims = {}                               # image_id -> (orig_w, orig_h) from csv
    for r in rows:
        cid = int(r["class_id"])
        if cid == 14 or r.get("x_min", "") in ("", None):   # "No finding" / no box
            continue
        cats[cid] = r["class_name"]
        boxes_by_img[r["image_id"]].append(
            (cid, float(r["x_min"]), float(r["y_min"]),
             float(r["x_max"]), float(r["y_max"])))
        # VinBigData train.csv carries original DICOM width/height per row
        if "width" in r and "height" in r and r["width"] and r["height"]:
            orig_dims[r["image_id"]] = (float(r["width"]), float(r["height"]))

    images, annotations = [], []
    ann_id = 1
    for img_id_str, boxes in boxes_by_img.items():
        fn = img_id_str + args.img_ext
        path = os.path.join(args.imgs, fn)
        if not os.path.exists(path):
            continue
        w, h = Image.open(path).size
        img_int_id = len(images) + 1
        images.append({"id": img_int_id, "file_name": fn, "width": w, "height": h})
        # scale boxes from original DICOM coords (csv width/height) to the PNG size.
        if img_id_str in scale:
            sx, sy = scale[img_id_str]
        elif img_id_str in orig_dims:
            ow, oh = orig_dims[img_id_str]
            sx, sy = w / ow, h / oh
        else:
            sx, sy = 1.0, 1.0
        for cid, x1, y1, x2, y2 in boxes:
            x1, y1, x2, y2 = x1 * sx, y1 * sy, x2 * sx, y2 * sy
            bw, bh = x2 - x1, y2 - y1
            if bw <= 0 or bh <= 0:
                continue
            annotations.append({
                "id": ann_id, "image_id": img_int_id, "category_id": cid,
                "bbox": [x1, y1, bw, bh], "area": bw * bh, "iscrowd": 0,
            })
            ann_id += 1

    categories = [{"id": cid, "name": name} for cid, name in sorted(cats.items())]
    coco = {"images": images, "annotations": annotations, "categories": categories}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(coco, open(args.out, "w"))
    print(f"images: {len(images)} | annotations: {len(annotations)} | "
          f"classes: {len(categories)} -> {args.out}")
    print("classes:", [c["name"] for c in categories])


if __name__ == "__main__":
    main()
