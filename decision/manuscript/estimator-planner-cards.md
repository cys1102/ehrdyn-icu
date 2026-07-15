---
title: Estimator and Planner Cards
created: 2026-07-14
updated: 2026-07-15
type: writing
tags: [planner, ope, offline-rl, benchmark-card]
status: active
confidence: high
---

# Estimator and planner cards

## Planner cards

### H1 exhaustive

- Exact search over currently supported actions for one step.
- Not labeled MPC.
- Executes the selected first action.

### H4 categorical CEM

- Horizon: four decisions.
- Candidates: 64 action sequences per iteration.
- Elites: eight action sequences per iteration.
- Iterations: three.
- Categorical probability smoothing: 0.2.
- Support masks applied at every simulated step.
- Receding horizon: execute the first action only, then replan.

### H8 categorical CEM

- Same contract as H4 with an eight-decision planning horizon.
- Query budgets are matched across comparable variants.

### Planner sensitivities

- Support constrained.
- Uncertainty penalized with coefficient 0.1.
- Support and uncertainty constrained.
- Fixed-horizon or empirical-hazard termination; learned termination is sensitivity when calibration is worse.

## Model-free policy cards

| Method | Fidelity label | Probability object | Key boundary |
| --- | --- | --- | --- |
| Behavior cloning | independent reimplementation | full categorical | primary non-value baseline |
| Discrete BCQ | independent reimplementation | full categorical where emitted | supported Bellman target |
| Discrete CQL | independent reimplementation | full categorical where emitted | conservative Q penalty |
| Soft-SPIBB | conceptual adapter | categorical | not an official exact implementation |
| Decision Transformer | contract adapter | categorical only when decoding supplies it | frozen target-return/decoding contract |
| Supported controls | local controls | categorical / prespecified epsilon-soft | random, minimum, maximum, severity rule |

Deterministic retrospective controls use a prespecified epsilon-soft probability object over the supported set with `epsilon=0.001`. It is part of the frozen diagnostic contract, not selected after seeing ESS, and no retrospective OPE score is authorized.

## Behavior denominator cards

- Exact behavior denominator in known-value environments.
- Paper-compatible one-layer LSTM, hidden size 16.
- Stronger cross-fitted denominator.
- Deliberately misspecified denominator stress contract.

Each denominator is evaluated separately for NLL, calibration/ECE, action-wise fit, overlap, ratio tails, and horizon-specific ESS. Denominators are never pooled.

## OPE estimator cards

- Trajectory IS and WIS.
- Per-decision IS, WPDIS, and CWPDIS.
- DR and WDR when nuisance prerequisites are met.
- FQE.
- Prespecified support-restricted variants.

Repeated-dataset approval applies only to the exact tuple:

`estimator × denominator × clipping × support rule × horizon × task regime`.

Each task uses its frozen decision horizons. For every task--regime cell, the evaluator generates 200 independent logged datasets for respiratory, shock, and heart failure and 500 for AKI--RRT; behavior denominators and nuisance models are refit within each dataset. Approval requires every gate below:

- minimum policy-specific empirical coverage across independent logged datasets at least `0.90`; a 95% Wilson interval is reported for precision;
- Spearman rank recovery at least `0.80`;
- pairwise policy-order recovery at least `0.80`;
- behavior-relative improvement-sign recovery at least `0.90`;
- false-improvement rate at most `0.05`;
- median ESS at least `max(100, 0.05 * logged episodes)`;
- maximum unsupported target-policy mass equal to zero;
- all required values finite.

The former policy-set interval inclusion rate within one fixed logged-data realization is retained as diagnostic-only. Known-value approval does not transport automatically to retrospective EHR evaluation.

## Current disposition

- Repeated-dataset known-value grid: 40 of 1,728 tuples approved.
- Adaptive response: 0 of 864 tuples approved.
- Null response: 40 of 864 tuples approved, all in AKI--RRT sanity cells.
- Retrospective EHR policy value: not executed and unavailable.

Complete repeated-dataset metrics and dispositions are in `repeated_dataset_ope_coverage.csv`, `repeated_dataset_ope_rank_and_sign.csv`, and `repeated_dataset_ope_authorization.csv`. Historical fixed-dataset surfaces remain available with diagnostic-only labels. No tuple is transferred by estimator family name alone.
