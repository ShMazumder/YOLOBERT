# Weekly arXiv SOTA Digest — 2026-07-18

Source: alphaXiv `discover_papers` (prioritize=recency). Metrics taken only from stated text; abstracts were truncated, so unstated numbers = **n/a** (never invented).

---

## 1. Computer Vision — detection / segmentation / pose / OBB

| # | Paper | Authors | arXiv | Contribution | Headline metric | SOTA? |
|---|-------|---------|-------|--------------|-----------------|-------|
| 1 | OBBSeg: Irregular Lesion Segmentation under Oriented Bounding Box Annotations | — et al. | 2607.06007 | Weakly-supervised medical segmentation using OBB annotations as intermediate supervision. | n/a | — |
| 2 | A Turbo-Inference Strategy for Object Detection and Instance Segmentation | — et al. | 2606.12371 | Faster detect-then-segment inference sharing detection features into segmentation. | n/a | — |
| 3 | DroneDAR: Long-Range Drone Distance Estimation Using Monocular Vision + Bounding-Box Features | — et al. | 2606.07756 | Monocular long-range drone distance estimation from bbox features. | n/a | — |
| 4 | Fast Online 3D Multi-Camera Multi-Object Tracking and Pose Estimation | — et al. | 2604.16522 | Online joint 3D MOT + pose from multi monocular cameras, 2D bbox+pose input only. | n/a | — |
| 5 | RiO-DETR: DETR for Real-time Oriented Object Detection | — et al. | 2603.09411 | Claims first real-time oriented-detection transformer (DETR for OBBs). | n/a | ⚠ claims "first real-time OBB DETR" |
| 6 | ER-Pose: Rethinking Keypoint-Driven Representation Learning for Real-Time Human Pose Estimation | — et al. | 2603.08681 | Single-stage real-time multi-person pose via keypoint representation learning. | n/a | — |

## 2. Large Language Models — reasoning / efficient training

| # | Paper | Authors | arXiv | Contribution | Headline metric | SOTA? |
|---|-------|---------|-------|--------------|-----------------|-------|
| 1 | Latent Thought Flow: Efficient Latent Reasoning in LLMs | — et al. | 2606.16222 | Latent-space reasoning avoiding token-decode bottleneck of explicit CoT. | n/a | — |
| 2 | EPTS: Elastic Post-Training Sparsity for Efficient LLM Compression | — et al. | 2606.25285 | Elastic post-training sparsity for on-device LLM compression. | n/a | — |
| 3 | Reasoning-preserved Efficient Distillation via Activation-aware Initialization | — et al. | 2605.29327 | Structured-pruning distillation that preserves reasoning via activation-aware init. | n/a | — |
| 4 | HyperGuide: Hyperbolic Guidance for Efficient Multi-Step Reasoning | — et al. | 2605.24140 | Hyperbolic guidance balancing single-pass speed vs tree-search accuracy. | n/a | — |
| 5 | LEAD: Length-Efficient Adaptive and Dynamic Reasoning | — et al. | 2605.09806 | Cuts verbose CoT length adaptively while keeping accuracy. | n/a | — |
| 6 | Tandem: Riding Together with Large and Small LMs for Efficient Reasoning | — et al. | 2604.23623 | Large+small model collaboration for cheaper reasoning inference. | n/a | — |

## 3. Multimodal / Vision-Language Models (VLM)

| # | Paper | Authors | arXiv | Contribution | Headline metric | SOTA? |
|---|-------|---------|-------|--------------|-----------------|-------|
| 1 | Hy-Embodied-VLM-1.0: Efficient Physical-World Agents | Tencent et al. | 2607.12894 | Embodied VLM adding agentic action-reasoning to multimodal perception. | n/a | — |
| 2 | FabriVLA: Lightweight Vision-Language-Action Model for Precise Multi-Task Manipulation | — et al. | 2607.08575 | Lightweight VLA (InternVL3.5 backbone + flow-matching action head). | n/a | — |
| 3 | AVA-VLM: Adaptive Visual Attention VLM for In-the-Wild Construction Site Monitoring | — et al. | 2607.05859 | Construction-site-tailored VLM with adaptive visual attention. | n/a | — |
| 4 | MonoIR-RS: Infrared Remote Sensing Vision-Language Learning with CLIP + VLM Adaptation | — et al. | 2607.06552 | CLIP/VLM adaptation for infrared remote-sensing imagery. | n/a | — |
| 5 | DataComp-VLM: Improved Open Datasets for Vision-Language Models | — et al. | 2606.28551 | Benchmark + curation strategies for open VLM training data. | n/a | — |
| 6 | VLM3: Vision Language Models Are Native 3D Learners | — et al. | 2605.30561 | Extends VLM prompting to 3D understanding tasks. | n/a | — |

---

*Notes:* First-author names not returned by connector metadata (listed "—"). No headline metrics appeared in returned (truncated) abstracts, so all marked n/a per no-invention rule. Only RiO-DETR made an explicit primacy/SOTA-style claim. To pull confirmed numbers, fetch full text of any flagged paper.
