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

This directory is the reader-facing manuscript package. It freezes six cohort targets above 10,000 subjects and 10,000 episodes, reports the complete current P/R/T component synthesis for the three unchanged lineages, and marks sepsis, AF/flutter, and heart-failure small-lineage results as superseded pending full reruns. It also contains the current exact-finite policy benchmark, historical sensitivities, repeated-dataset OPE audit, and EHR-to-known-value diagnostic bridge through 2026-07-15. Internal experiment identifiers are excluded from `manuscript.tex` and retained in `provenance-manifest.csv` and `number-audit.csv`.

Contents:

- `manuscript.tex` and `manuscript.pdf`: final source and compiled manuscript.
- `benchmark-card.md`: scope, tasks, roles, metrics, and use boundaries.
- `contracts.md`: cohort, feature, action, reward, timing, and split contracts.
- `estimator-planner-cards.md`: planner, policy, probability, and OPE contracts.
- `tables/` and `figures/`: sanitized aggregate publication surfaces. Files prefixed `current_scale_qualified_` contain the primary three-cohort view: all 18 transition method--cohort rows, 3,060 heterogeneous policy--seed rows, 2,160 component-model--planner rows, 15,552 heterogeneous repeated-coverage rows, and 2,592 heterogeneous OPE dispositions. The null/composite-adaptive grid and complete prior six-task ledger remain available as explicitly separated sensitivities.
- `number-audit.csv`: every reported manuscript number with source, estimator, split, and uncertainty method.
- `provenance-manifest.csv`: internal source IDs, immutable commits, and hashes.
- `reproducibility.md`, `environment-manifest.md`, and `credentialed-reconstruction.md`: rebuild instructions and data-access boundary.
- `non-claims-and-limitations.md`: explicit interpretation limits.

The current evidence authorizes no retrospective EHR policy-value comparison or method winner. The anonymous release candidate is at <https://anonymous.4open.science/r/ehrdyn-icu-65FB>. Independent credentialed reconstruction remains pending; it means aggregate-result regeneration by another MIMIC-credentialed user from a clean clone, not external or clinical validation.
