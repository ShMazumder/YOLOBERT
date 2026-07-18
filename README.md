# YOLOBERT

> [One-line goal — e.g. "unified detector + language model for multimodal grounding".]
> Fill the `[...]` fields. See `CLAUDE.md` for the full research-context spec.

[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)

## Highlights

- **Novelty**: [architecture / loss / training recipe].
- **Result**: [primary metric] = **[X.X]** on [benchmark], [+Δ vs prior SOTA].
- Task: [detection / segmentation / pose / OBB / LLM / multimodal].

## Install

```bash
# Option A — Docker (reproducible)
docker build -t yolobert:latest .
docker run --gpus all -it --rm -v $(pwd):/workspace -v /data:/data yolobert:latest

# Option B — conda / pip
conda create -n yolobert python=3.10 -y && conda activate yolobert
pip install -r requirements.txt
```

## Data prep

```
data/
  [dataset]/
    annotations/   # COCO json / YOLO txt / Pascal VOC
    images/
```

[Download links + expected structure. Note preprocessing / normalization stats.]
Small split files are force-added despite `.gitignore`: `git add -f splits/*.json`.

## Train

```bash
python tools/train.py --config configs/base.yaml
# override any field:
python tools/train.py --config configs/base.yaml --opts lr=0.001 epochs=50
# resume:
python tools/train.py --config configs/base.yaml --resume work_dirs/base/last.pth
```

Checkpoints + TensorBoard logs land in `work_dir` (default `work_dirs/base/`).
`config_used.yaml` is snapshotted alongside results for traceability.

```bash
tensorboard --logdir work_dirs/
```

## Evaluate

```bash
python tools/train.py --config configs/base.yaml --opts epochs=0   # eval-only
# TODO: add tools/test.py for standalone eval on the test split.
```

## Results

| Method | [AP] | [AP50] | [FPS] |
|--------|------|--------|-------|
| [Baseline] | [xx.x] | [xx.x] | [xxx] |
| **Ours** | **[xx.x]** | **[xx.x]** | [xxx] |

Tables generated with the `latex-results-table` skill (booktabs, bold=best,
underline=2nd). Figures use `paper/paper_style.py` (colorblind-safe, ≥8pt, PDF).
Every number traces to a log in `work_dirs/` — no fabricated metrics.

## Repo layout

```
configs/     experiment configs (yaml)
datasets/    data loaders + splits
models/      architecture code
tools/       train.py, test.py, eval scripts
paper/       refs.bib, paper_style.py, manuscript, digests
work_dirs/   checkpoints + logs (gitignored)
```

## Citation

```bibtex
@inproceedings{[key]YYYY,
  title     = {[Title]},
  author    = {[Last, First and ...]},
  booktitle = {[CVPR / NeurIPS / ...]},
  year      = {[YYYY]},
  note      = {arXiv:[XXXX.XXXXX]},
}
```

## License

[MIT / Apache-2.0 / ...]
