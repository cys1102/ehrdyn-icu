---
title: EHR Decision-Evaluation Benchmark Card
created: 2026-07-14
updated: 2026-07-15
type: writing
tags: [benchmark-card, ehr, world-model, offline-rl]
status: active
confidence: high
---

# Benchmark card

## Purpose

Standardize comparison and failure decomposition across EHR P/R/T component learning, component model × planner combinations, model-based and model-free offline RL, known-value truth, retrospective support diagnostics, and heterogeneous task evaluability.

Transition accuracy is a validity layer, not the benchmark endpoint. Retrospective EHR policy value is unavailable because no task completes the exact estimator, trained target-policy, probability, support, and post-training overlap contract.

## Data and access

- Source: MIMIC-IV under credentialed PhysioNet access.
- Publicly redistributable: code, schemas, synthetic fixtures, aggregate tables, figures, contracts, hashes, and documentation.
- Restricted: patient rows, identifiers, exact timestamps, trajectories, role membership, row-level predictions/probabilities, tensors, checkpoints, credentials, and clinical text.

## Evidence status

- EHR surface: subject-disjoint development train/validation roles.
- Retrospective action semantics: recorded four-hour exposure/setting abstraction; not guaranteed assignable at interval start.
- Confirmatory status: no genuinely untouched role remains.
- Primary known-value surface: four adaptive exact-finite environments with state-dependent optima and exact dynamic-programming values.
- Evaluator sensitivities: historical monotone exact/semi-synthetic environments retained for OPE and implementation stress tests.
- Real-EHR OPE: not executed for comparison; AKI has task-matched semi-synthetic candidates but incomplete policy-specific gates.
- Independent credentialed reconstruction: pending aggregate-result regeneration from a clean clone; not external validation.

## Tasks

- Reference scaffold: AI-Clinician-aligned sepsis, four-hour fluid-5 × vasopressor-5 action, K=25; not an exact cohort/reward reproduction.
- Known-value policy extensions: respiratory support, shock, severe-AKI RRT initiation, and heart failure.
- World-model-only extension: AF/flutter.
- AKI and heart failure retain their original P/T transition surfaces and use separate K2/K8 decision contracts for policy evaluation.

## Model and policy coverage

- P/R/T component families: recurrent/GRU-D, causal Transformer, categorical recurrent state-space model, Gaussian recurrent ensemble.
- Sanity controls: persistence and histogram gradient boosting.
- Planners: supported H1 exhaustive, H4 categorical CEM, H8 categorical CEM; support-constrained and uncertainty-penalized variants.
- Model-free: behavior cloning, discrete BCQ, discrete CQL, Soft-SPIBB adapter, Decision Transformer adapter, and supported controls.

## Primary evaluation surfaces

- State: normalized observed-cell RMSE and MAE, feature-group and support-stratified error.
- Rollout: logged-action recursive error and horizon degradation.
- Reward and termination: prediction/propagation error, Brier score, ECE, calibration, survival sanity.
- Uncertainty: NLL/CRPS, coverage, interval score, width, and risk–coverage.
- Policy: exact adaptive return/regret, behavior-normalized exact regret, support violations, divergence, collapse, model-exploitation gap, and planner disagreement.
- OPE: exact estimator-contract accuracy, policy-set interval inclusion, rank/sign recovery, false improvement, and ESS.

## Machine-readable audit surfaces

- Complete layer-specific inventory: six transition methods, five uncertainty-capable methods, 34 policy/planner labels, and nine OPE estimators.
- Complete six-method task-balanced table plus all 36 cohort--method rows; winner-only leader tables are secondary views.
- Fixed primary factual-action estimand and per-row episode-bootstrap interval counts.
- Task-balanced one-step and recursive uncertainty summary.
- All 680 adaptive non-oracle method--seed rows with exact return/regret, plus 2,448 historical monotone sensitivity rows.
- All 16,128 reference and task-matched OPE tuple metrics, gate dispositions, and approved identities.

## Intended use

- Development-stage benchmarking of computational pipelines.
- Evaluator, support, uncertainty, and exploitation audits.
- Task-contract comparison and reproducibility testing.

## Scope boundary

- No causal effect, retrospective policy-value, deployment, or external-site claim.
- Known-value results characterize constructed mechanisms; retrospective EHR rows characterize prediction, support, collapse, and observability only.
