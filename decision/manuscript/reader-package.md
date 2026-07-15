---
title: Final EHR Decision-Evaluation Benchmark Manuscript Package
created: 2026-07-14
updated: 2026-07-15
type: writing
tags: [kdd, benchmark, ehr, world-model, offline-rl, reproducibility]
status: active
confidence: high
---

# Final manuscript package

This directory is the canonical reader-facing manuscript package. It combines the frozen six-task P/R/T component synthesis, adaptive exact-finite policy benchmark, historical monotone sensitivities, and task-matched OPE audits through 2026-07-15. Internal experiment identifiers are excluded from `manuscript.tex` and retained in `provenance-manifest.csv` and `number-audit.csv`.

Contents:

- `manuscript.tex` and `manuscript.pdf`: final source and compiled manuscript.
- `benchmark-card.md`: scope, tasks, roles, metrics, and use boundaries.
- `contracts.md`: cohort, feature, action, reward, timing, and split contracts.
- `estimator-planner-cards.md`: planner, policy, probability, and OPE contracts.
- `tables/` and `figures/`: sanitized aggregate publication surfaces, including all 36 cohort--transition-method rows, all 680 adaptive non-oracle method--seed rows, all 2,448 historical monotone sensitivity rows, and all reference/task-matched OPE tuple metrics and dispositions.
- `number-audit.csv`: every reported manuscript number with source, estimator, split, and uncertainty method.
- `provenance-manifest.csv`: internal source IDs, immutable commits, and hashes.
- `reproducibility.md`, `environment-manifest.md`, and `credentialed-reconstruction.md`: rebuild instructions and data-access boundary.
- `non-claims-and-limitations.md`: explicit interpretation limits.

The current evidence authorizes no retrospective EHR policy-value comparison or method winner. The anonymous release candidate is at <https://anonymous.4open.science/r/ehrdyn-icu-65FB>. Independent credentialed reconstruction remains pending; it means aggregate-result regeneration by another MIMIC-credentialed user from a clean clone, not external or clinical validation.
