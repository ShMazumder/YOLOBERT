"""paper_style.py — camera-ready matplotlib defaults for ML papers.

Vector PDF output, >=8pt fonts, colorblind-safe palette (Wong 2011),
type-42 (TrueType) fonts so CVPR/NeurIPS PDF checks pass (no Type-3).

USAGE
    from paper_style import set_paper_style, PALETTE, save
    set_paper_style(column="single")     # or "double" for full-width figs
    fig, ax = plt.subplots()
    ax.plot(x, y, color=PALETTE["blue"], label="Ours")
    save(fig, "figures/ablation")        # writes figures/ablation.pdf

Set column width to your venue's \columnwidth in inches:
    CVPR/ICCV single col ~3.25in, double ~6.875in
    NeurIPS/ICLR single col ~5.5in (one-column layout)
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

# Wong 2011 colorblind-safe palette (Nature Methods). Distinct in deuteranopia.
PALETTE = {
    "black":   "#000000",
    "orange":  "#E69F00",
    "skyblue": "#56B4E9",
    "green":   "#009E73",
    "yellow":  "#F0E442",
    "blue":    "#0072B2",
    "vermil":  "#D55E00",
    "purple":  "#CC79A7",
}
# Ordered cycle for automatic multi-series plots.
CYCLE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
         "#E69F00", "#56B4E9", "#F0E442", "#000000"]

# Figure widths in inches by column type.
_WIDTHS = {"single": 3.25, "double": 6.875, "neurips": 5.5}


def set_paper_style(column="single", base_font=9, golden=True):
    """Apply rcParams. base_font in pt (>=8 required by most venues)."""
    assert base_font >= 8, "venues require font size >= 8pt"
    width = _WIDTHS.get(column, 3.25)
    height = width / 1.618 if golden else width * 0.75
    mpl.rcParams.update({
        # --- vector, embeddable fonts (no Type-3) ---
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "text.usetex": False,          # set True if you have a LaTeX install
        "font.family": "serif",        # match paper body; use "sans-serif" for slides
        "mathtext.fontset": "cm",
        # --- sizes ---
        "figure.figsize": (width, height),
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": base_font,
        "axes.titlesize": base_font,
        "axes.labelsize": base_font,
        "xtick.labelsize": base_font - 1,
        "ytick.labelsize": base_font - 1,
        "legend.fontsize": base_font - 1,
        # --- clean look ---
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.5,
        "lines.markersize": 4,
        "legend.frameon": False,
        "figure.autolayout": True,
        # --- colorblind-safe default cycle ---
        "axes.prop_cycle": mpl.cycler(color=CYCLE),
    })


def save(fig, path_no_ext, formats=("pdf",)):
    """Save tight vector figure(s). Default PDF; add 'png' for previews."""
    for fmt in formats:
        fig.savefig(f"{path_no_ext}.{fmt}", bbox_inches="tight",
                    pad_inches=0.01, transparent=True)


if __name__ == "__main__":
    # Self-test: render a demo figure with the palette.
    import numpy as np
    set_paper_style(column="single")
    x = np.linspace(0, 10, 100)
    fig, ax = plt.subplots()
    for i, name in enumerate(["Baseline", "+module A", "Ours"]):
        ax.plot(x, np.sin(x + i) + i * 0.3, label=name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AP$_{50}$")
    ax.legend()
    save(fig, "demo_figure")
    print("wrote demo_figure.pdf")
