"""plot_fingerprint.py — cross-domain (L, S, C) fingerprint figure for the paper.

Grouped bars: for each domain, the three failure axes (averaged over models),
showing which axis dominates. Uses paper/paper_style.py (colorblind-safe, PDF).

    python tools/plot_fingerprint.py \
        --domains aerial:runs/diag/aerial/fingerprints.csv \
                  medical:runs/diag/medical/fingerprints.csv \
        --out paper/figures/fingerprint
"""
import argparse
import csv
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "paper"))


def _mean(rows, key):
    return st.mean(float(r[key]) for r in rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="+", required=True)
    ap.add_argument("--out", default="paper/figures/fingerprint")
    args = ap.parse_args()

    import numpy as np
    from paper_style import PALETTE, save, set_paper_style
    import matplotlib.pyplot as plt

    set_paper_style(column="single")
    domains = [d.split(":", 1) for d in args.domains]

    # axes: L (localization), S_norm (semantic), C_ece scaled x5 for visibility
    axis_keys = [("L", "L", 1.0), ("S_norm", r"$S_{norm}$", 1.0),
                 ("C_ece", r"$C_{ece}\!\times\!5$", 5.0)]
    colors = [PALETTE["blue"], PALETTE["vermil"], PALETTE["green"]]

    fig, ax = plt.subplots()
    x = np.arange(len(domains))
    w = 0.25
    for j, (key, label, scale) in enumerate(axis_keys):
        vals = []
        for _, cpath in domains:
            with open(cpath) as f:
                rows = list(csv.DictReader(f))
            vals.append(_mean(rows, key) * scale)
        ax.bar(x + (j - 1) * w, vals, w, label=label, color=colors[j])

    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d, _ in domains])
    ax.set_ylabel("failure magnitude")
    ax.set_ylim(0, 1.05)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    save(fig, args.out)
    print(f"wrote {args.out}.pdf")


if __name__ == "__main__":
    main()
