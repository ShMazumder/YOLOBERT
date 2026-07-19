"""make_paper_table.py — build the cross-domain OVD-Diagnose results table (booktabs).

Reads one fingerprints.csv per domain and emits a grouped LaTeX table:
domain x model rows with AP_global, AP_oracle, L, S_norm, C_ece, IoA-F1.
Per domain, the dominant failure axis is highlighted.

    python tools/make_paper_table.py \
        --domains aerial:runs/diag/aerial/fingerprints.csv \
                  medical:runs/diag/medical/fingerprints.csv \
        --out paper/tables/ovd_fingerprint.tex

Note: S_norm is unstable when AP_oracle ~ 0 (ratio on a near-zero denominator);
such cells are marked with a dagger and should not be over-interpreted.
"""
import argparse
import csv
from pathlib import Path

MODEL_LABEL = {"yoloworld": "YOLO-World", "owlv2": "OWLv2",
               "groundingdino": "Grounding DINO"}
S_UNSTABLE_AP = 0.01     # AP_oracle below this -> S_norm unreliable


def _rows(csv_path):
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="+", required=True,
                    help='list of name:csv (e.g. aerial:runs/diag/aerial/fingerprints.csv)')
    ap.add_argument("--out", default="paper/tables/ovd_fingerprint.tex")
    args = ap.parse_args()

    domains = [d.split(":", 1) for d in args.domains]

    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Cross-domain OVD failure fingerprint. AP is COCO mAP@[.5:.95]. "
        r"$L{=}1{-}\mathrm{AR}_{\mathrm{SAM}}$ (domain-level localizability). "
        r"$S_{\mathrm{norm}}$ is the fraction of achievable AP lost to vocabulary confusion; "
        r"$^{\dagger}$ unstable when AP$_{\mathrm{oracle}}{\approx}0$. "
        r"$C_{\mathrm{ece}}$ is expected calibration error.}",
        r"  \label{tab:fingerprint}",
        r"  \begin{tabular}{ll cc c c c c}",
        r"    \toprule",
        r"    Domain & Model & AP$_{\mathrm{g}}$ & AP$_{\mathrm{o}}$ & $L$ & "
        r"$S_{\mathrm{norm}}$ & $C_{\mathrm{ece}}$ & IoA-F1 \\",
        r"    \midrule",
    ]

    for di, (dname, cpath) in enumerate(domains):
        rows = _rows(cpath)
        n = len(rows)
        for ri, r in enumerate(rows):
            model = MODEL_LABEL.get(r["model"], r["model"])
            apg = float(r["AP_global"]); apo = float(r["AP_oracle"])
            L = float(r["L"]); sn = float(r["S_norm"])
            ce = float(r["C_ece"]); f1 = float(r.get("IoA_F1", 0) or 0)
            sn_str = f"{sn:.2f}" + (r"$^{\dagger}$" if apo < S_UNSTABLE_AP else "")
            dom_cell = (rf"\multirow{{{n}}}{{*}}{{{dname.capitalize()}}}"
                        if ri == 0 else "")
            lines.append(
                f"    {dom_cell} & {model} & {apg:.3f} & {apo:.3f} & "
                f"{L:.2f} & {sn_str} & {ce:.3f} & {f1:.3f} \\\\")
        if di < len(domains) - 1:
            lines.append(r"    \midrule")

    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"wrote {out}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
