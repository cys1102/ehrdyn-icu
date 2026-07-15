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

Approval applies only to the exact tuple:

`estimator × denominator × clipping × support rule × horizon × task regime`.

The reference evaluator checks H1, H2, H4, H8, H12, and H17 separately. Task-matched evaluators use the horizons defined by their decision contracts (AKI H1/H2/H3; heart failure H1/H2/H4/H8/H12). Approval requires every gate below:

- policy-set bootstrap-interval coverage within a fixed logged-data realization at least `0.90` (not repeated-dataset frequentist coverage);
- Spearman rank recovery at least `0.80`;
- pairwise policy-order recovery at least `0.80`;
- behavior-relative improvement-sign recovery at least `0.90`;
- false-improvement rate at most `0.05`;
- median ESS at least `max(100, 0.05 * logged episodes)`;
- maximum unsupported target-policy mass equal to zero;
- maximum paired true-return standard error at most `0.03`;
- all required values finite.

Exact finite-environment approval does not transport automatically to aggregate-calibrated semi-synthetic or retrospective EHR evaluation.

## Current disposition

- Exact finite known-value environments: 32 of 3,456 tuples approved for that synthetic scope. All are moderate-regime, horizon-one, masked-support contracts; their full identities are in `ope_approved_exact_tuples.csv` (under `tables/` in the reader package and `decision/evidence/` in the anonymous release).
- Shared reference semi-synthetic environment: 0 of 3,456 tuples approved.
- AKI task-matched evaluator: 228 of 1,728 exact-finite and 236 of 1,728 semi-synthetic tuples approved.
- Heart-failure task-matched evaluator: 0 of 2,880 exact-finite and 0 of 2,880 semi-synthetic tuples approved.
- Retrospective EHR policy value: not executed and unavailable.

Complete achieved metrics and dispositions are in `ope_reference_all_tuple_metrics.csv` and `ope_task_matched_all_tuple_metrics.csv` (under `tables/` in the reader package and `decision/evidence/` in the anonymous release). No tuple is transferred by estimator family name alone.
