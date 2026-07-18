# OVD-Diagnose: A Cross-Domain Failure-Decomposition Benchmark for Open-Vocabulary Detection

> Working design doc. Target: NeurIPS / ICLR **Datasets & Benchmarks** track (or CVPR
> main as analysis). Contribution is a benchmark + diagnostic protocol, NOT a new detector.
> Pivoted from the FCOS baseline scaffold after literature pressure-testing ruled out
> method-novelty directions (AutoAssign, OBB, small-object, OVD-adaptation all saturated).

## 1. One-line contribution

A cross-domain benchmark and diagnostic protocol that **decomposes open-vocabulary
detector (OVD) failure into three interpretable axes — localization, semantic
confusion, and calibration — measured uniformly across specialized domains (aerial,
agriculture, medical)**, revealing *which* failure mode dominates *where*, and why
generic OVD does not transfer out of natural images.

## 2. Why this is defensible (gap analysis)

- **Method-novelty in mainstream detection/OVD is saturated** (AutoAssign owns dynamic
  assignment; OBB/small-object/OVD-adaptation crowded; training-free OVD TTA + class
  pruning + calibration all shipping monthly in 2026). A 2×T4 lab cannot win that race.
- **Single-domain OVD benchmarks exist** (aerial: Tsourveloudis 2026; underwater; OV-RS-seg)
  but **no multi-specialized-domain OVD benchmark**, and **no OVD-specific failure
  decomposition**. TIDE / DnD decompose *closed-set* errors and cannot see the axis unique
  to OVD: semantic confusion from the open vocabulary itself.
- Benchmarks are **compute-light** (inference only → T4-friendly), **ownable**, and a
  recognized top-tier artifact when well-designed and adopted.

## 3. The diagnostic decomposition (intellectual core)

Let `M` be a frozen OVD model, image `x`, full domain vocabulary `V` (|V| = N), and the
per-image present classes `P(x) ⊆ V` (from ground truth). `M(x, T)` runs detection with
prompt set `T`. Three controlled inference modes:

| Mode | Prompt set | Isolates |
|------|-----------|----------|
| **Global** | `V` (all N classes) | real-world setting (loc + confusion + calib together) |
| **Oracle** | `P(x)` (present classes only) | removes vocabulary confusion |
| **Agnostic** | single "object" / class-agnostic | pure localization capacity |

### Three error axes (per domain, aggregated over models or per model)

**(a) Localization error `L`** — can the model box the objects at all, labels aside.
Use class-agnostic recall in Agnostic/Oracle mode:
```
L = 1 − AR_agnostic         # AR = average recall of boxes ignoring category
```
High `L` ⇒ visual encoder fails on this domain's imagery (viewpoint, scale, texture).

**(b) Semantic-confusion error `S`** — error caused *purely* by enlarging the vocabulary.
The Global↔Oracle gap (Tsourveloudis'26 measured a 15× F1 swing here on aerial):
```
S      = AP_oracle − AP_global
S_norm = (AP_oracle − AP_global) / max(AP_oracle, ε)   # fraction of achievable AP lost to vocabulary
```
High `S` ⇒ overlapping taxonomy / weak text-side discrimination, not vision.

**(c) Calibration error `C`** — score-vs-correctness mismatch (threshold brittleness;
Tsourveloudis'26 saw 69% FP at a fixed threshold). Expected Calibration Error over
detection confidences, plus a threshold-sensitivity term:
```
C_ece = Σ_b (|B_b|/D) · | acc(B_b) − conf(B_b) |          # ECE over confidence bins B_b
C_thr = F1_best_threshold − F1_fixed_threshold           # brittleness to a single operating point
```

### Failure fingerprint

Each `(domain, model)` → vector `(L, S_norm, C_ece)`. The paper's payload is the
**pattern across domains**, e.g. (hypotheses to test, not claims):
- aerial → **S-bound** (nadir view localizes OK, taxonomy confuses),
- medical → **L-bound** (subtle low-contrast findings hard to localize),
- agriculture → **C-bound** (dense repeated instances, threshold collapse).

### Validation (makes it a metric, not a description)

Synthetic controls where the true failure mode is known:
- inject near-synonym distractor classes into `V` ⇒ `S` must rise, `L`,`C` stable;
- blur / downscale inputs ⇒ `L` must rise;
- temperature-scale logits ⇒ `C` must move, `L`,`S` stable.
Passing these shows each axis measures what it claims (construct validity).

## 4. Benchmark composition

### Domains + datasets

| Domain | Dataset(s) | Boxes | Access | Notes |
|--------|-----------|-------|--------|-------|
| Aerial | LAE-80C (DOTA/DIOR/FAIR1M/xView) | yes | public | Tsourveloudis'26 used it; 80 cls, taxonomy overlap |
| Agriculture | public crop/weed/fruit detection sets | yes | public/Kaggle | dense repeated instances |
| Medical | VinDr-CXR (14 findings) | yes | PhysioNet (credentialed) | strong contrast; fallback below |
| *Fallback* | FishDet-M (underwater) | yes | public | drop-in if medical licensing stalls |

### Models (all frozen, zero-shot inference)

| Model | Backbone | Params | Family |
|-------|----------|--------|--------|
| Grounding DINO | Swin-L | 218M | region + grounding |
| OWLv2 | ViT-L/14 | 428M | dense image-text |
| YOLO-World | CSPDarknet | 60M | real-time |
| YOLOE | C3k2 | 26M | real-time |
| LLMDet | Swin-L | 435M | LLM-enhanced |
| (+1–2 newer 2026 OVD) | — | — | keep current |

## 5. Experiment matrix

1. **Main table** — `(L, S_norm, C_ece)` for every domain × model; standard AP/AR too.
2. **Fingerprint figure** — 3-axis radar / ternary per domain (paper_style.py palette).
3. **Validation** — synthetic-control curves confirming each axis (Section 3).
4. **Vocabulary-scaling curve** — AP vs |V| (N → present-only), per domain: quantifies S.
5. **Cross-dataset brittleness** — within-aerial DIOR→FAIR1M→xView degradation (extend Paper 1).
6. **Actionable coda (optional, analysis-framed)** — one *training-free* remedy per dominant
   axis (e.g. context vocabulary prior for S; temperature scaling for C) → % of gap recovered.
   Framed as "diagnosis is actionable", not as a method contribution.

## 6. Deliverables

- Curated multi-domain OVD eval split + unified label spaces.
- **OVD-Diagnose toolkit** (TIDE-for-OVD): given detections + GT + vocabulary, emits
  `(L, S, C)` and plots. This adoptable artifact is what earns a benchmark-track accept.
- Leaderboard + reproducible configs (reuse the existing datasets/ + metrics/ harness).

## 7. Feasibility on 2×T4

Inference only — no OVD pretraining. Cost = forward passes × (models × domains × modes).
Estimate: 5 models × 3 domains × 3 modes × few-k images = large but embarrassingly
parallel and checkpointable. Kaggle Commit runs per (model,domain). No training arms race.

## 8. Risks / honesty

- **Fast-moving neighborhood** (domain-generalized OVOD, ExDet, ProCal, FACTOR — all 2026).
  Mitigation: be first with the *multi-domain decomposition* framing; benchmarks defend once
  released + adopted.
- **Descriptive-not-diagnostic risk.** Mitigation: the synthetic-control validation (Sec 3).
- **Medical data access.** Mitigation: underwater fallback keeps it fully public.
- **Adoption bar.** A clean toolkit + leaderboard is the difference between tech-report and
  top-tier benchmark.

## 9. Paper outline (maps to paper/main.tex)

1. Intro — OVD fails out-of-domain; *why* is unmeasured; we decompose it.
2. Related — OVD, OVD benchmarks (single-domain), error-diagnosis (TIDE/DnD), OVD TTA/calib.
3. The OVD-Diagnose protocol — modes, the three axes (math), validation design.
4. Benchmark — domains, datasets, models, unified vocabularies.
5. Experiments — main fingerprints, validation, scaling, brittleness, actionable coda.
6. Findings & discussion — which axis dominates where, implications for OVD design.
7. Limitations, broader impact, conclusion.

## 10. Immediate next actions

- [ ] Lock 3rd domain (confirm VinDr-CXR access; else underwater FishDet-M).
- [ ] Formalize `L/S/C` in code: extend tools/metrics.py with an `ovd_diagnose` module.
- [ ] Wire one frozen OVD model (start: YOLO-World, cheapest) through datasets/ in the 3 modes.
- [ ] Reproduce Paper 1's aerial numbers as a correctness anchor before scaling out.
- [ ] Draft Sec 3 math into paper/main.tex.
