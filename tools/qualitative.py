"""qualitative.py — render GT vs OVD detections to illustrate failure modes.

Produces a grid of example images with ground-truth boxes (green) and a model's
global-mode detections (red, labelled with class + score). Picks images that
illustrate the paper's failure taxonomy:

  localized_wrong : a detection overlaps a GT box (IoU>0.5) but the class is wrong
                    -> semantic confusion (aerial story)
  missed          : GT objects with no overlapping detection at all
                    -> localization failure (medical story)

    python tools/qualitative.py \
        --ann data/aerial/annotations/instances_val.json --imgs data/aerial/images \
        --results runs/diag/aerial/owlv2/results_global.json \
        --mode localized_wrong --n 6 --out paper/figures/qual_aerial.png
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    return inter / (aw * ah + bw * bh - inter + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True)
    ap.add_argument("--imgs", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--mode", default="localized_wrong",
                    choices=["localized_wrong", "missed"])
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--score_thr", type=float, default=0.2)
    ap.add_argument("--out", default="paper/figures/qualitative.png")
    args = ap.parse_args()

    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from PIL import Image
    from pycocotools.coco import COCO

    coco = COCO(args.ann)
    catname = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}
    dets_by_img = {}
    for d in json.load(open(args.results)):
        if d["score"] >= args.score_thr:
            dets_by_img.setdefault(d["image_id"], []).append(d)

    # score images by how well they illustrate the chosen failure mode
    scored = []
    for img_id in coco.getImgs().keys() if hasattr(coco, "getImgs") else coco.imgs.keys():
        gts = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        if not gts:
            continue
        dets = dets_by_img.get(img_id, [])
        if args.mode == "localized_wrong":
            n_hit = sum(1 for g in gts for d in dets
                        if _iou(d["bbox"], g["bbox"]) > 0.5 and d["category_id"] != g["category_id"])
            scored.append((n_hit, img_id))
        else:  # missed
            n_miss = sum(1 for g in gts
                         if all(_iou(d["bbox"], g["bbox"]) < 0.5 for d in dets) if dets or True)
            # only count as illustrative if there ARE detections elsewhere or none at all
            scored.append((n_miss, img_id))
    scored.sort(reverse=True)
    picks = [iid for _, iid in scored[:args.n]]

    cols = min(3, args.n); rows = (args.n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.ravel() if hasattr(axes, "ravel") else [axes]
    for ax, img_id in zip(axes, picks):
        info = coco.loadImgs(img_id)[0]
        ax.imshow(Image.open(os.path.join(args.imgs, info["file_name"])).convert("RGB"))
        for g in coco.loadAnns(coco.getAnnIds(imgIds=img_id)):
            x, y, w, h = g["bbox"]
            ax.add_patch(mpatches.Rectangle((x, y), w, h, fill=False, ec="lime", lw=1.5))
        for d in dets_by_img.get(img_id, []):
            x, y, w, h = d["bbox"]
            ax.add_patch(mpatches.Rectangle((x, y), w, h, fill=False, ec="red", lw=1.0))
            ax.text(x, max(0, y - 2), f"{catname.get(d['category_id'], '?')}:{d['score']:.2f}",
                    color="red", fontsize=5, va="bottom")
        ax.set_title(info["file_name"], fontsize=6); ax.axis("off")
    for ax in axes[len(picks):]:
        ax.axis("off")
    fig.suptitle(f"GT (green) vs detections (red) — {args.mode}", fontsize=9)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
