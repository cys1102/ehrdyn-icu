---
title: Credentialed User Reconstruction Instructions
created: 2026-07-14
updated: 2026-07-14
type: writing
tags: [mimic-iv, credentialed-access, reconstruction]
status: active
confidence: high
---

# Credentialed-user reconstruction

## Access boundary

These instructions assume authorized MIMIC-IV access under the PhysioNet data-use agreement. They do not redistribute or reconstruct patient records in the manuscript repository. Keep all raw tables, task arrays, membership, row predictions, and checkpoints in the authorized experiment backend.

## Local preparation

1. Check out the exact `world-ehr` commit in `provenance-manifest.csv`.
2. Create or activate the environment recorded in `environment-manifest.md`.
3. Place MIMIC-IV 3.1 CSV tables under the authorized local data root expected by the backend, or pass `--mimiciv-root` explicitly.
4. Verify repository status and preserve existing outputs. Every command below must use a new output path.

## Construction and component sequence

The authoritative internal stage names are retained in `provenance-manifest.csv`. The corresponding executable entry points are:

```bash
# AI-Clinician-aligned K25 sepsis materialization (legacy module name retained)
python -m kdd_benchmark_discovery.run_kdd_s02_canonical_sepsis_materialization \
  --mimiciv-root /authorized/path/to/mimiciv/3.1 \
  --output kdd_benchmark_discovery/results/reproduction_sepsis_materialization

# Converged world-model components
python -m kdd_benchmark_discovery.run_kdd098r_world_models \
  --config configs/kdd098r_world_model.json \
  --mimiciv-root /authorized/path/to/mimiciv/3.1 \
  --output results/reproduction_world_models \
  --checkpoint-root results/reproduction_world_model_checkpoints

# Model-free development diagnostics
python -m kdd_benchmark_discovery.run_kdd101_model_free_diagnostics \
  --config configs/kdd101_model_free_diagnostics_v5.json \
  --mimiciv-root /authorized/path/to/mimiciv/3.1 \
  --output kdd_benchmark_discovery/results/reproduction_model_free
```

Known-value evaluator and cross-cohort aggregate stages do not require patient-level exports:

```bash
python -m kdd_benchmark_discovery.run_kdd_e01_evaluator_repair_preflight \
  --output kdd_benchmark_discovery/results/reproduction_evaluator_preflight

python -m kdd_benchmark_discovery.run_kdd_e02_known_value_full \
  --output kdd_benchmark_discovery/results/reproduction_known_value

python -m kdd_benchmark_discovery.run_kdd_x01_cross_cohort_evaluability \
  --output kdd_benchmark_discovery/results/reproduction_cross_cohort_contracts

python -m kdd_benchmark_discovery.run_kdd_x02_cross_cohort_policy_benchmark \
  --config configs/kdd_x02_cross_cohort_policy_benchmark_v1.json \
  --output kdd_benchmark_discovery/results/reproduction_cross_cohort_policy

python -m kdd_benchmark_discovery.run_kdd_adapt01_adaptive_known_value \
  --config configs/kdd_adapt01_adaptive_known_value_v1.json \
  --output kdd_benchmark_discovery/results/reproduction_adaptive_known_value
```

## Validation and release boundary

- Compare only aggregate tables and artifact hashes.
- Scan outputs for identifiers, exact timestamps, row-level trajectories, probabilities, tensors, checkpoints, credentials, and free text before transfer.
- Do not transfer membership or patient-level arrays to ResearchWiki or a public artifact.
- Do not label a fixture execution as credentialed scientific parity.
- Do not compute retrospective EHR OPE unless the exact task-specific estimator tuple, trained target-policy probability surface, and post-training overlap/ESS receipt all pass together.
