# OVD-Diagnose

**Decomposing open-vocabulary detector failure across specialized domains.**

Aggregate mAP tells you *that* an open-vocabulary detector (OVD) fails outside natural
images, not *why*. OVD-Diagnose is a training-free, inference-only protocol that separates
OVD failure into three interpretable axes — **localization (L)**, **semantic confusion (S)**,
and **calibration (C)** — using controlled prompting modes. Any frozen detector, any
COCO-format domain.

> Status: research preview. Numbers in `paper/` are from evaluation subsets and should be
> read as preliminary (see *Limitations* in the manuscript).

## The protocol

For a frozen detector `M`, image `x`, domain vocabulary `V`, and the classes actually
present `P(x)`:

| Mode | Prompt | Isolates |
|------|--------|----------|
| **Global** | `V` (all classes) | deployment setting (all axes jointly) |
| **Oracle** | `P(x)` (present only) | removes vocabulary confusion |
| **Agnostic** | class-agnostic proposer | localizability, text-independent |

Axes:

- `L = 1 − AR_proposer` — external proposer (SAM by default, or any domain detector).
- `L_det = 1 − AR_oracle-labelfree` — **detector-intrinsic**, proposer-free (guards against
  blaming a natural-image proposer for a domain it wasn't built for).
- `S = AP_oracle − AP_global`, `S_norm = S / AP_oracle` — AP lost purely to the vocabulary.
  *Unstable when `AP_oracle ≈ 0`; flagged, not interpreted, in that regime.*
- `C_ece` (expected calibration error) and `C_thr = F1* − F1_fixed` (operating-point brittleness).

## Install

```bash
pip install ultralytics transformers pycocotools pyyaml sentence-transformers
python tools/ovd_diagnose.py     # metric self-test (no model/data needed)
```

## Quickstart

```bash
# all models on one domain -> fingerprints.csv + per-model results_{global,oracle}.json
python tools/run_all.py \
  --ann  data/aerial/annotations/instances_val.json \
  --imgs data/aerial/images \
  --out  runs/diag/aerial --device cuda:0 \
  --models "yoloworld:yolov8s-world.pt,owlv2:google/owlv2-base-patch16-ensemble,groundingdino:IDEA-Research/grounding-dino-tiny" \
  --sam_weights mobile_sam.pt
```

Ready-to-run notebooks (Kaggle, GPU): `notebooks/ovd_diagnose.ipynb` (aerial),
`notebooks/ovd_diagnose_medical.ipynb` (medical).

## Evaluation protocol (exact settings)

| Setting | Value |
|---|---|
| Detection metric | COCO mAP@[.50:.95], `pycocotools`, restricted to evaluated images |
| Anchor metric | IoA-F1, IoA threshold **0.7**, score threshold **0.25** (matches prior aerial benchmarks) |
| Inference score threshold | **0.001** (low, so calibration sees the full score range) |
| Fixed operating point | **0.25** (used for `C_thr`) |
| Prompt format | raw category names from the benchmark taxonomy; Grounding DINO uses lowercase period-separated (`"class a. class b."`) |
| Localization proposer | Ultralytics SAM `mobile_sam.pt`, "everything" mode, `imgsz=1024` |
| Detection input size | `imgsz=800` (YOLO-World); processor default (OWLv2, Grounding DINO) |
| Class mapping | contiguous ids internally; mapped back to dataset category ids before scoring |
| Splits | benchmark-provided val split; `--limit N` evaluates the first N images (reported per run) |

Missing/unreadable images are skipped and excluded from the denominator; the number of
evaluated images is printed by every run and should be reported with results.

## Domains

| Domain | Dataset | Classes | Source |
|---|---|---|---|
| Aerial | LAE-80C (DOTA/DIOR/FAIR1M/xView) | 80 | HuggingFace `jaychempan/LAE-1M` |
| Medical | VinDr-CXR (via VinBigData 512px) | 14 | Kaggle `awsaf49/vinbigdata-512-image-dataset` |

Add a domain: provide a COCO-format json + image dir. `tools/vindr_to_coco.py` shows a
converter example (handles box rescaling to resized images).

## Validation (construct validity)

```bash
# C responds to confidence rescaling; AP/S must not (post-hoc, no GPU)
python tools/synthetic_controls.py --control temperature --ann ... --results ... --out ...
# S responds to vocabulary; semantic distractors should hurt more than random
python tools/synthetic_controls.py --control vocab --distractor semantic --ann ... --imgs ... --out ...
python tools/synthetic_controls.py --control vocab --distractor random   --ann ... --imgs ... --out ...
# L responds to input degradation
python tools/synthetic_controls.py --control blur --ann ... --imgs ... --out ...
```

Uncertainty: `tools.ovd_diagnose.bootstrap_ci(ann, results, metric="ioa_f1", n_boot=1000)`
gives image-level bootstrap 95% intervals.

## Tools

| Script | Purpose |
|---|---|
| `tools/ovd_diagnose.py` | metrics: L, L_det, S, C, IoA-F1 anchor, bootstrap CIs, sanity checks |
| `tools/run_all.py` | multi-model driver (SAM once + each OVD), fault-isolated, writes fingerprints.csv |
| `tools/run_ovd.py` / `run_agnostic.py` | single-model / proposer-only runs |
| `tools/synthetic_controls.py` | temperature / vocab (random\|semantic) / blur controls |
| `tools/plot_reliability.py` | reliability diagrams (calibration) |
| `tools/qualitative.py` | GT-vs-detection failure examples |
| `tools/make_paper_table.py`, `tools/plot_fingerprint.py` | paper table + figure |
| `models/ovd/` | adapters: yoloworld, owlv2, groundingdino, sam, generic `proposer` |

Add a detector: subclass `OVDAdapter` in `models/ovd/`, implement
`predict(image_path, class_names) -> [(bbox_xywh, score, class_index)]`, decorate with
`@register_adapter("name")`.

## Reproducing the paper

```bash
python tools/make_paper_table.py --domains aerial:runs/diag/aerial/fingerprints.csv \
                                            medical:runs/diag/medical/fingerprints.csv \
                                 --out paper/tables/ovd_fingerprint.tex
python tools/plot_fingerprint.py --domains aerial:... medical:... --out paper/figures/fingerprint
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## Citation

```bibtex
@misc{mazumder2026ovddiagnose,
  title  = {OVD-Diagnose: Decomposing Open-Vocabulary Detector Failure Across Specialized Domains},
  author = {Mazumder, Shazzad Hossain and Hasan, Kazi Nazmul and Chowdhury, Muntasir Karim and Abid, Shahriar Zaman},
  year   = {2026},
  note   = {Department of Computer Science \& Engineering, Feni University},
}
```

## License

[MIT / Apache-2.0 — choose before release]
