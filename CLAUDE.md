# CLAUDE.md — ML Research Context

> Standing context for this repo. Auto-loaded every session. Keeps Claude from
> re-deriving project facts. Fill bracketed `[...]` fields once, delete unused rows.
> Compress with `/caveman-compress CLAUDE.md` after filling to cut input tokens.

## Project

- **Title**: [paper working title]
- **Goal**: [one line — e.g. "improve small-object detection on aerial imagery"]
- **Task type**: [ ] detection [ ] recognition [ ] segmentation (sem/inst/panoptic) [ ] pose [ ] OBB [ ] LLM [ ] multimodal (VLM) [ ] other: ___
- **Target venue**: [CVPR / ICCV / ECCV / NeurIPS / ICLR / ACL / EMNLP / journal]
- **Deadline**: [YYYY-MM-DD]
- **Novelty claim**: [what is new — architecture / loss / data / training recipe]

## Repo Layout

```
[configs/]      # experiment configs
[datasets/]     # data loaders + splits
[models/]       # architecture code
[tools/]        # train.py, test.py, eval scripts
[work_dirs/]    # checkpoints + logs (gitignored)
```

- Framework: [PyTorch / MMDetection / Detectron2 / Ultralytics / HF transformers / timm]
- Entry: `python [tools/train.py] [config]`
- Env: [conda env name / requirements.txt / Dockerfile path]

## Datasets

| Name | Split sizes | Classes | Notes |
|------|-------------|---------|-------|
| [COCO] | train 118k / val 5k | 80 | primary benchmark |
| [custom] | [n] | [k] | [annotation format: COCO json / YOLO txt / Pascal VOC] |

- Image size / resolution: [e.g. 1024×1024]
- Preprocessing: [normalization stats, augmentation policy]

## Metrics (report these, exact names)

- **Detection/OBB**: AP, AP50, AP75, AP_S/M/L, mAP. (OBB uses rotated IoU.)
- **Segmentation**: mIoU, mAcc, aAcc; PQ/SQ/RQ (panoptic); AP^mask (instance).
- **Pose**: AP (OKS), PCK, MPJPE.
- **Recognition/Classification**: Top-1, Top-5, F1, precision/recall.
- **LLM**: perplexity, task accuracy, [benchmark: MMLU/GSM8K/...], win-rate.
- **Multimodal/VLM**: [VQA acc, CIDEr, BLEU, retrieval R@1/5/10, CLIPScore].
- **Efficiency**: params (M), FLOPs (G), FPS, latency (ms), GPU mem.
- Primary metric to optimize: [e.g. AP50]

## Notation (keep consistent across code + manuscript)

- [ $x \in \mathbb{R}^{H\times W\times 3}$ input image ]
- [ $\hat{y}$ prediction, $y$ ground truth ]
- [ $\mathcal{L}$ total loss = ... ]
- Symbol table lives in: [paper/notation.tex]

## Baselines to compare (SOTA)

| Method | Venue/Year | Primary metric | Source |
|--------|-----------|----------------|--------|
| [YOLOv8] | [2023] | [AP] | [github / paper] |
| [DINO] | [ICLR23] | [AP] | [arXiv id] |

## Conventions

- Results tables: **booktabs LaTeX**, bold = best, underline = 2nd best. Use the
  `latex-results-table` skill.
- Numbers: [N] decimal places; report mean ± std over [N] seeds.
- Figures: [PDF vector, matplotlib, font size ≥ 8pt, colorblind-safe palette].
- Citations: [BibTeX file paper/refs.bib]; cite arXiv + published version.
- Never invent numbers. Every reported metric traces to a log in `work_dirs/`.

## Current status / TODO

- [ ] [what's running now]
- [ ] [next ablation]
- [ ] [sections left to write]

## Do / Don't for Claude

- DO point-read specific files; don't ask me to paste whole logs.
- DO give diffs/patches, not full-file reprints, on edits.
- DO escalate to Opus-high only for method design, proofs, rebuttals.
- DON'T fabricate results, citations, or SOTA numbers — flag if unknown.
- DON'T change random seeds or eval protocol without noting it.
