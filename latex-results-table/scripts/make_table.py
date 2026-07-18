#!/usr/bin/env python3
"""make_table.py — format ML metrics into a publish-ready booktabs LaTeX table.

Auto-bolds the best value and underlines the second-best per column, respecting
whether higher or lower is better for each metric.

USAGE (CLI):
    python make_table.py results.csv \
        --caption "Comparison on COCO val2017." \
        --label tab:main \
        --decimals 1 \
        --lower FLOPs,Params,Latency

CSV format: first column = Method name, remaining columns = metrics.
Cells may be "78.2" or "78.2±0.3" or "78.2 0.3" (value std).

USAGE (import):
    from make_table import format_table
    rows = [
        {"Method": "Baseline", "AP": 45.2, "AP50": 63.1},
        {"Method": "Ours",     "AP": 48.7, "AP50": 66.4},
    ]
    print(format_table(rows, caption="...", label="tab:main",
                       lower_better={"Latency"}, ours="Ours"))
"""
import argparse
import csv
import re
import sys

# Metrics where LOWER is better (extend as needed).
DEFAULT_LOWER = {
    "flops", "params", "latency", "mem", "memory", "perplexity", "ppl",
    "mpjpe", "error", "err", "mae", "rmse", "fid",
}

_CELL = re.compile(r"^\s*([-+]?\d*\.?\d+)\s*(?:[±]|\+/-|\s)\s*([-+]?\d*\.?\d+)\s*$")


def _parse_cell(raw):
    """Return (value, std_or_None). Non-numeric -> (None, None)."""
    if raw is None:
        return None, None
    s = str(raw).strip()
    if s == "" or s == "-":
        return None, None
    m = _CELL.match(s)
    if m:
        return float(m.group(1)), float(m.group(2))
    try:
        return float(s), None
    except ValueError:
        return None, None


def _escape(text):
    for a, b in [("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")]:
        text = text.replace(a, b)
    return text


def format_table(rows, caption="CAPTION", label="tab:main", decimals=1,
                 lower_better=None, ours=None, position="t"):
    """rows: list of dicts, first key is the method-name column.

    Returns a LaTeX string (booktabs). Bold=best, underline=2nd best per column.
    """
    if not rows:
        raise ValueError("no rows given")
    lower_better = {c.lower() for c in (lower_better or set())}
    cols = list(rows[0].keys())
    name_col, metric_cols = cols[0], cols[1:]

    # Parse all cells.
    parsed = {}  # (row_idx, col) -> (val, std)
    for i, r in enumerate(rows):
        for c in metric_cols:
            parsed[(i, c)] = _parse_cell(r.get(c))

    # Rank per column to find best / second best.
    fmt = {}  # (row_idx, col) -> latex cell string
    for c in metric_cols:
        lower = c.lower() in (lower_better or set()) or c.lower() in DEFAULT_LOWER
        vals = [(i, parsed[(i, c)][0]) for i in range(len(rows))
                if parsed[(i, c)][0] is not None]
        vals.sort(key=lambda t: t[1], reverse=not lower)
        best_i = vals[0][0] if vals else None
        second_i = vals[1][0] if len(vals) > 1 else None
        for i in range(len(rows)):
            v, std = parsed[(i, c)]
            if v is None:
                fmt[(i, c)] = "-"
                continue
            num = f"{v:.{decimals}f}"
            if std is not None:
                num = num + rf"{{\scriptsize$\pm${std:.{decimals}f}}}"
            if i == best_i:
                cell = rf"\textbf{{{num}}}"
            elif i == second_i:
                cell = rf"\underline{{{num}}}"
            else:
                cell = num
            fmt[(i, c)] = cell

    # Build LaTeX.
    align = "l" + " " + " ".join("c" for _ in metric_cols)
    header = " & ".join([_escape(name_col)] + [_escape(c) for c in metric_cols])
    body_lines = []
    for i, r in enumerate(rows):
        name = _escape(str(r[name_col]))
        if ours is not None and str(r[name_col]) == ours:
            name = rf"\textbf{{{name}}}"
        cells = [name] + [fmt[(i, c)] for c in metric_cols]
        body_lines.append("    " + " & ".join(cells) + r" \\")
    body = "\n".join(body_lines)

    return (
        f"\\begin{{table}}[{position}]\n"
        f"  \\centering\n"
        f"  \\caption{{{caption}}}\n"
        f"  \\label{{{label}}}\n"
        f"  \\begin{{tabular}}{{{align}}}\n"
        f"    \\toprule\n"
        f"    {header} \\\\\n"
        f"    \\midrule\n"
        f"{body}\n"
        f"    \\bottomrule\n"
        f"  \\end{{tabular}}\n"
        f"\\end{{table}}\n"
    )


def _read_csv(path):
    with open(path, newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="CSV: col1=Method, rest=metrics")
    ap.add_argument("--caption", default="CAPTION")
    ap.add_argument("--label", default="tab:main")
    ap.add_argument("--decimals", type=int, default=1)
    ap.add_argument("--lower", default="",
                    help="comma-separated metric cols where lower is better")
    ap.add_argument("--ours", default=None, help="method name to bold in row")
    ap.add_argument("--position", default="t")
    args = ap.parse_args(argv)

    rows = _read_csv(args.csv)
    lower = {c.strip() for c in args.lower.split(",") if c.strip()}
    print(format_table(rows, caption=args.caption, label=args.label,
                       decimals=args.decimals, lower_better=lower,
                       ours=args.ours, position=args.position))
    return 0


if __name__ == "__main__":
    sys.exit(main())
