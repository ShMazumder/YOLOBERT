"""plot_reliability.py — reliability diagrams for OVD detections (calibration axis).

Turns the C_ece number into an actual result: for each model, bin detection
confidences and plot empirical precision (accuracy) vs mean confidence. A curve
below the diagonal = overconfident; above = underconfident. Annotates ECE.

    python tools/plot_reliability.py \
        --ann data/aerial/annotations/instances_val.json \
        --models yoloworld:runs/diag/aerial/yoloworld/results_global.json \
                 owlv2:runs/diag/aerial/owlv2/results_global.json \
                 groundingdino:runs/diag/aerial/groundingdino/results_global.json \
        --out paper/figures/reliability_aerial
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _bin_stats(scores, hits, n_bins=10):
    import numpy as np
    edges = np.linspace(0, 1, n_bins + 1)
    conf, acc, frac = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (scores > lo) & (scores <= hi)
        if m.sum() == 0:
            conf.append(np.nan); acc.append(np.nan); frac.append(0.0); continue
        conf.append(scores[m].mean()); acc.append(hits[m].mean())
        frac.append(m.sum() / len(scores))
    return np.array(conf), np.array(acc), np.array(frac)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True)
    ap.add_argument("--models", nargs="+", required=True, help="name:results_global.json")
    ap.add_argument("--out", default="paper/figures/reliability")
    ap.add_argument("--iou_thr", type=float, default=0.5)
    args = ap.parse_args()

    import numpy as np
    from pycocotools.coco import COCO
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "paper"))
    from paper_style import PALETTE, save, set_paper_style
    import matplotlib.pyplot as plt
    from tools.ovd_diagnose import _match_tps, expected_calibration_error

    set_paper_style(column="single")
    coco = COCO(args.ann)
    colors = [PALETTE["blue"], PALETTE["vermil"], PALETTE["green"], PALETTE["purple"]]

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], "--", color="0.6", lw=0.8, label="perfect")
    for i, spec in enumerate(args.models):
        name, path = spec.split(":", 1)
        results = json.load(open(path))
        scores, hits = _match_tps(coco, results, args.iou_thr)
        ece = expected_calibration_error(scores, hits)
        conf, acc, _ = _bin_stats(scores, hits)
        ok = ~np.isnan(conf)
        ax.plot(conf[ok], acc[ok], "o-", ms=3, color=colors[i % len(colors)],
                label=f"{name} (ECE={ece:.3f})")
    ax.set_xlabel("mean confidence"); ax.set_ylabel("empirical precision")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=6, loc="upper left")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save(fig, args.out)
    print(f"wrote {args.out}.pdf")


if __name__ == "__main__":
    main()
