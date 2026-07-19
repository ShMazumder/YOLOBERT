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
    S_UNSTABLE_AP = 0.01     # S_norm unreliable when AP_oracle ~ 0

    # Honest discriminators: localizability (AR_SAM), vocabulary cost (S_norm, hatched
    # where unstable), calibration (C_ece x5). AR_SAM is the clean cross-domain axis.
    axis_keys = [("AR_agnostic", r"AR$_{SAM}$", 1.0),
                 ("S_norm", r"$S_{norm}$", 1.0),
                 ("C_ece", r"$C_{ece}\!\times\!5$", 5.0)]
    colors = [PALETTE["blue"], PALETTE["vermil"], PALETTE["green"]]

    fig, ax = plt.subplots()
    x = np.arange(len(domains))
    w = 0.25
    for j, (key, label, scale) in enumerate(axis_keys):
        vals, hatched = [], []
        for _, cpath in domains:
            with open(cpath) as f:
                rows = list(csv.DictReader(f))
            vals.append(_mean(rows, key) * scale)
            # flag S_norm as unstable when mean AP_oracle ~ 0 for this domain
            hatched.append(key == "S_norm" and _mean(rows, "AP_oracle") < S_UNSTABLE_AP)
        bars = ax.bar(x + (j - 1) * w, vals, w, label=label, color=colors[j])
        for b, h in zip(bars, hatched):     # hatch + fade unreliable S bars
            if h:
                b.set_hatch("///"); b.set_alpha(0.35); b.set_edgecolor("white")

    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d, _ in domains])
    ax.set_ylabel("failure / signal magnitude")
    ax.set_ylim(0, 1.05)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    ax.text(0.99, 0.02, "hatched = unstable (AP$_o$≈0)", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6, color="0.4")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)   # ensure figures/ exists
    save(fig, args.out)
    print(f"wrote {args.out}.pdf")


if __name__ == "__main__":
    main()
