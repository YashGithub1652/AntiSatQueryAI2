# SatQuery AI — 100% PS Implementation Roadmap

This document is the execution contract for SIH PS 26167. It separates
**implemented software**, **trained model artifacts**, and **measured
evaluation evidence**. A configured model is not considered complete until
its real checkpoint loads and its benchmark result is measured.

## Phase 0 — Platform freeze
- GUI/input state
- GeoTIFF/TIFF validation
- query parser and agent graph
- model registry
- audit trace
- evidence rendering

Status: implemented; continue regression testing.

## Phase 1 — Mandatory remote-sensing adaptation
**Target:** GeoChat-7B adapted with BigEarthNet.txt.

Pipeline:
1. BigEarthNet.txt annotations
2. real Sentinel-2 image resolution
3. GeoChat instruction JSON
4. native GeoChat multimodal QLoRA
5. adapter validation
6. before/after evaluation on prescribed RS benchmarks
7. register only validated adapter as the production VLM

Files:
- `satquery/backend/training/bigearthnet_prepare.py`
- `satquery/backend/training/lora_finetune.py`
- `satquery/backend/training/validate_adaptation.py`

## Phase 2 — Single-image mandatory baseline
- VQA → GeoChat
- caption/scene description → GeoChat
- grounding → RSVG
- optional box → SAM mask
- RemoteCLIP auxiliary semantic evidence

Exit criteria:
- real inference
- no fake fallback in production
- benchmark evidence
- trace contains model/checkpoint/task.

## Phase 3 — Bi-temporal change
- input compatibility and co-registration
- ChangeFormer pixel change map
- quantitative area/cluster statistics
- Change-VQA specialist for language interpretation
- CDVQA evaluation
- LEVIR-CD quantitative evaluation

Exit criteria:
- trained/verified ChangeFormer checkpoint
- measured IoU/F1
- measured CDVQA result
- spatial evidence returned.

## Phase 4 — Optical + SAR
- Sentinel-2 optical preprocessing
- Sentinel-1 VV/VH preprocessing
- real co-registered paired dataset
- SAR encoder
- optical encoder
- bi-directional cross-attention
- fusion head
- multimodal explanation

Important: synthetic data may be used for unit tests only. It cannot be
used as the final scientific training/evidence claim.

Exit criteria:
- real paired training
- verified checkpoint
- held-out evaluation
- optical-only vs SAR-only vs fused ablation.

## Phase 5 — Agentic orchestration
The controller must:
1. classify query;
2. validate number/modality/format;
3. check co-registration when required;
4. select specialist models from registry;
5. execute in permitted sequence;
6. aggregate textual + spatial outputs;
7. compute calibrated confidence;
8. emit observable audit trace.

## Phase 6 — Evaluation and provenance
Every reported metric must carry:
- model/checkpoint
- dataset
- split
- sample count
- metric definition
- measured/published provenance
- run timestamp/id.

Published baseline values must never be presented as SatQuery measurements.

## Phase 7 — ISRO/SAC readiness
Test the complete pipeline against:
- pre-georeferenced optical input
- SAR input
- paired optical/SAR
- bi-temporal pair
- task-specific reference outputs where available
- hidden-evaluation-like conditions.

## Final 100% definition

SatQuery is PS-complete only when all mandatory branches have real
specialist checkpoints, real inference, evaluation evidence, agent routing,
visual evidence, confidence/provenance, and an auditable execution trace.

A UI badge, registry entry, configuration file, synthetic training run, or
published benchmark number alone does not satisfy completion.
