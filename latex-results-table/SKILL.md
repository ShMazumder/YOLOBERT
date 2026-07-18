---
name: latex-results-table
description: >-
  Format ML experiment metrics into publish-ready LaTeX tables (booktabs style,
  bold-best / underline-second-best, mean±std). Use whenever the user wants a
  results table, ablation table, comparison table, or SOTA leaderboard for a
  paper — for detection (AP/AP50/mAP/OBB), segmentation (mIoU/PQ), pose
  (AP/PCK), classification (Top-1/F1), LLM (accuracy/perplexity), or multimodal
  (VQA/CIDEr/R@k). Triggers: "results table", "ablation table", "latex table",
  "comparison table", "make a table for the paper", "booktabs".
---

# LaTeX Results Table

Turn raw metrics into a clean, publish-ready LaTeX table. Output compiles with
`\usepackage{booktabs}` (add `graphicx`, `multirow` only if used).

## When to use

User has numbers (from logs, a dict, CSV, or typed inline) and wants a paper
table. Detection, segmentation, pose, OBB, recognition, LLM, or multimodal.

## Workflow

1. **Collect the data.** Accept a CSV path, a Python dict, or inline numbers.
   Columns = metrics, rows = methods. Note which row is *your* method.
2. **Confirm format choices** (ask only if unclear):
   - decimals (default 1),
   - highlight rule: bold = best, underline = 2nd best (default),
   - higher-is-better per metric (default True; set False for perplexity,
     FLOPs, latency, params, MPJPE, error rates).
3. **Run the helper** to auto-bold/underline and emit LaTeX:
   ```bash
   python scripts/make_table.py data.csv --decimals 1 --caption "..." --label "tab:main"
   ```
   Or import `format_table(rows, ...)` for programmatic use. See script header.
4. **Return the `.tex` snippet** in chat AND, if the user wants, write it to a
   `.tex` file in their repo. Never fabricate numbers — every value comes from
   the user's data.

## Rules

- `booktabs` only: `\toprule \midrule \bottomrule`. No vertical rules.
- Bold best, `\underline{}` second best, per column, respecting direction.
- Report `mean±std` when std given: `\num{78.2}\std{0.3}` or `78.2\pm0.3`.
- Right-align numeric columns (`S` column from `siunitx` if available, else `r`).
- Mark your method row with `\rowcolor` or a `\textbf` method name — ask which.
- Escape `%`, `_`, `&` in method names.
- Keep caption above the table (journal/CVPR convention): `\caption` before `\begin{tabular}`... actually caption goes right after `\begin{table}`.

## Output skeleton

```latex
\begin{table}[t]
  \centering
  \caption{CAPTION}
  \label{tab:LABEL}
  \begin{tabular}{l cccc}
    \toprule
    Method & AP & AP$_{50}$ & AP$_{75}$ & FPS \\
    \midrule
    Baseline      & 45.2 & 63.1 & 48.9 & \textbf{120} \\
    \textbf{Ours} & \textbf{48.7} & \textbf{66.4} & \underline{51.2} & 95 \\
    \bottomrule
  \end{tabular}
\end{table}
```
