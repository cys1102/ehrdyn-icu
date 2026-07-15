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

Standardize comparison and failure decomposition across EHR world-model learning, world model × planner combinations, model-based and model-free offline RL, known-value truth, retrospective support diagnostics, and heterogeneous task evaluability.

Transition accuracy is a validity layer, not the benchmark endpoint. Retrospective EHR policy value is unavailable because no task completes the exact estimator, trained target-policy, probability, support, and post-training overlap contract.

## Data and access

- Source: MIMIC-IV under credentialed PhysioNet access.
- Publicly redistributable: code, schemas, synthetic fixtures, aggregate tables, figures, contracts, hashes, and documentation.
- Restricted: patient rows, identifiers, exact timestamps, trajectories, role membership, row-level predictions/probabilities, tensors, checkpoints, credentials, and clinical text.

## Evidence status

- EHR surface: subject-disjoint development train/validation roles.
- Retrospective action semantics: recorded four-hour exposure/setting abstraction; not guaranteed assignable at interval start.
- Confirmatory status: no genuinely untouched role remains.
- Known-value surface: exact finite and aggregate-calibrated semi-synthetic environments with known mechanisms.
- Real-EHR OPE: not executed for comparison; AKI has task-matched semi-synthetic candidates but incomplete policy-specific gates.
- Independent credentialed reproduction: pending.

## Tasks

- Canonical reference: sepsis, four-hour fluid-5 × vasopressor-5 action, K=25.
- Known-value policy extensions: respiratory support, shock, severe-AKI RRT initiation, and heart failure.
- World-model-only extension: AF/flutter.
- AKI and heart failure retain their original P/T transition surfaces and use separate K2/K8 decision contracts for policy evaluation.

## Model and policy coverage

- World models: recurrent/GRU-D, causal Transformer, categorical recurrent state-space model, Gaussian recurrent ensemble.
- Sanity controls: persistence and histogram gradient boosting.
- Planners: supported H1 exhaustive, H4 categorical CEM, H8 categorical CEM; support-constrained and uncertainty-penalized variants.
- Model-free: behavior cloning, discrete BCQ, discrete CQL, Soft-SPIBB adapter, Decision Transformer adapter, and supported controls.

## Primary evaluation surfaces

- State: normalized observed-cell RMSE and MAE, feature-group and support-stratified error.
- Rollout: logged-action recursive error and horizon degradation.
- Reward and termination: prediction/propagation error, Brier score, ECE, calibration, survival sanity.
- Uncertainty: NLL/CRPS, coverage, interval score, width, and risk–coverage.
- Policy: true known-value return, behavior-relative paired differences, support violations, divergence, collapse, model-exploitation gap, and planner disagreement.
- OPE: exact estimator-contract accuracy, coverage, rank/sign recovery, false improvement, and ESS.

## Machine-readable audit surfaces

- Fixed primary factual-action estimand and per-row episode-bootstrap interval counts.
- Task-balanced one-step and recursive uncertainty summary.
- All 2,448 known-value policy rows with paired Monte Carlo SE.
- All 16,128 reference and task-matched OPE tuple metrics, gate dispositions, and approved identities.

## Intended use

- Development-stage benchmarking of computational pipelines.
- Evaluator, support, uncertainty, and exploitation audits.
- Task-contract comparison and reproducibility testing.

## Scope boundary

- No causal effect, retrospective policy-value, deployment, or external-site claim.
- Known-value results characterize constructed mechanisms; retrospective EHR rows characterize prediction, support, collapse, and observability only.
