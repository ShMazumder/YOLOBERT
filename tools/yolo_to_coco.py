"""yolo_to_coco.py — convert a YOLO-format detection dataset to COCO json.

Written for adding new OVD-Diagnose domains (Brackish underwater, agriculture, ...)
without hand-writing a converter each time. Handles the layouts these datasets
actually ship in:

  root/images/*.jpg + root/labels/*.txt          (parallel dirs)
  root/train/images/*.jpg + root/train/labels/   (split subdirs, use --split)
  root/*.jpg + root/*.txt                        (flat, same dir)

Class names come from --names (comma list), a classes.txt / obj.names (one per
line), or a data.yaml with a `names:` list. YOLO boxes are normalized
cx,cy,w,h in [0,1]; COCO wants absolute x,y,w,h, so real image sizes are read
from the files themselves -- no assumed resolution.

    python tools/yolo_to_coco.py \
        --root /kaggle/input/brackish-dataset/dataset \
        --split valid \
        --out data/underwater/annotations/instances_val.json

Prints a box-scale sanity line at the end; if max coord exceeds image bounds the
source was likely already absolute, and the script says so rather than writing
silently-wrong boxes.
"""
import argparse
import json
import os
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _load_names(root: Path, names_arg):
    if names_arg:
        return [n.strip() for n in names_arg.split(",") if n.strip()]
    for cand in ("classes.txt", "obj.names", "names.txt"):
        p = root / cand
        if p.exists():
            return [l.strip() for l in p.read_text().splitlines() if l.strip()]
    for cand in ("data.yaml", "data.yml", "dataset.yaml"):
        p = root / cand
        if p.exists():
            import yaml
            y = yaml.safe_load(p.read_text())
            nm = y.get("names")
            if isinstance(nm, dict):                  # {0: 'fish', 1: 'crab'}
                return [nm[k] for k in sorted(nm, key=int)]
            if isinstance(nm, list):
                return list(nm)
    raise SystemExit(
        "no class names found: pass --names 'a,b,c' or add classes.txt / data.yaml under root")


def _find_pairs(root: Path, split: str):
    """Return [(image_path, label_path)], tolerating the common layouts."""
    base = root / split if split and (root / split).is_dir() else root
    img_dir = base / "images" if (base / "images").is_dir() else base
    lbl_dir = base / "labels" if (base / "labels").is_dir() else base

    pairs = []
    for img in sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXT):
        # mirror any nested structure when images/ and labels/ are parallel trees
        rel = img.relative_to(img_dir).with_suffix(".txt")
        lbl = lbl_dir / rel
        if not lbl.exists():
            lbl = lbl_dir / (img.stem + ".txt")       # flat fallback
        pairs.append((img, lbl if lbl.exists() else None))
    return pairs, img_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dataset root")
    ap.add_argument("--split", default="", help="subdir e.g. train/valid/test")
    ap.add_argument("--out", required=True, help="output COCO json")
    ap.add_argument("--names", default=None, help="comma list overriding classes.txt")
    ap.add_argument("--limit", type=int, default=0, help="cap #images (debug)")
    args = ap.parse_args()

    from PIL import Image

    root = Path(args.root)
    names = _load_names(root, args.names)
    pairs, img_dir = _find_pairs(root, args.split)
    if args.limit:
        pairs = pairs[:args.limit]
    if not pairs:
        raise SystemExit(f"no images found under {img_dir}")

    categories = [{"id": i + 1, "name": n} for i, n in enumerate(names)]
    images, annotations = [], []
    ann_id = 1
    n_unlabeled = 0
    max_rel = 0.0                       # tracks whether coords really were normalized

    for img_id, (img_path, lbl_path) in enumerate(pairs, start=1):
        try:
            with Image.open(img_path) as im:
                W, H = im.size
        except Exception as e:
            print(f"[skip unreadable] {img_path.name}: {e}")
            continue

        images.append({
            "id": img_id,
            # store path relative to the image dir so --imgs points at img_dir
            "file_name": str(img_path.relative_to(img_dir)),
            "width": W, "height": H,
        })

        if lbl_path is None:
            n_unlabeled += 1
            continue

        for line in lbl_path.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cid = int(float(parts[0]))
            cx, cy, bw, bh = (float(v) for v in parts[1:5])
            max_rel = max(max_rel, cx, cy, bw, bh)
            x, y = (cx - bw / 2) * W, (cy - bh / 2) * H
            w, h = bw * W, bh * H
            # clip to frame; some exports run boxes slightly outside
            x, y = max(0.0, x), max(0.0, y)
            w, h = min(w, W - x), min(h, H - y)
            if w <= 1 or h <= 1:
                continue
            if not 0 <= cid < len(names):
                print(f"[skip] class id {cid} outside 0..{len(names)-1} in {lbl_path.name}")
                continue
            annotations.append({
                "id": ann_id, "image_id": img_id, "category_id": cid + 1,
                "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0,
            })
            ann_id += 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"images": images, "annotations": annotations,
               "categories": categories}, open(out, "w"))

    print(f"images: {len(images)} | annotations: {len(annotations)} | "
          f"classes: {len(names)} -> {out}")
    print(f"classes: {names}")
    if n_unlabeled:
        print(f"[note] {n_unlabeled} images had no label file (kept as negatives)")
    if max_rel > 1.5:
        print(f"[WARN] max normalized coord was {max_rel:.1f} > 1 — source boxes were "
              f"probably ALREADY absolute, so this conversion is wrong. Inspect a label "
              f"file before trusting these annotations.")
    else:
        mx = max((max(a['bbox'][0] + a['bbox'][2], a['bbox'][1] + a['bbox'][3])
                  for a in annotations), default=0)
        print(f"max box extent: {mx:.0f}px (image dims up to "
              f"{max(i['width'] for i in images)}x{max(i['height'] for i in images)}) -> OK")


if __name__ == "__main__":
    main()
