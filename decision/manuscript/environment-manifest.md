---
title: Environment Manifest
created: 2026-07-14
updated: 2026-07-14
type: writing
tags: [reproducibility, environment, manifest]
status: active
confidence: high
---

# Environment manifest

## Frozen source

- Experiment repository: `world-ehr`
- Source commit: `aeccdde6fb8e413caed64908a70c980e7b39c109`
- Current aggregate synthesis: `kdd_benchmark_discovery/results/kdd_syn_final_benchmark_20260714_213941/`
- Manuscript repository: `ResearchWiki`
- Python executable used for synthesis: `<world-model-python>`

## Core versions

- Python 3.11.14
- NumPy 2.4.1
- pandas 2.3.3
- Matplotlib 3.10.8
- scikit-learn 1.8.0
- PyTorch 2.10.0+cu130
- LaTeX engine: Tectonic from the frozen local manuscript toolchain

## Seeds and planner constants

- Training seeds: 3408, 3411, 3414.
- Planner horizons: 1, 4, 8.
- H4/H8 candidates: 64 sequences per iteration.
- H4/H8 iterations: 3.
- OPE horizons: 1, 2, 4, 8, 12, 17.

Exact source and aggregate artifact hashes are recorded in `provenance-manifest.csv` and `artifact-hashes.json`.
