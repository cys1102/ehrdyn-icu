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
- Frozen cohort-scale gate: at least 10,000 subjects and 10,000 ICU-stay episodes; all six final target lineages pass.
- Current primary known-value surface: 18 exact-finite task--mechanism environments, including 12 prespecified interior-optimum, state-dependent, heterogeneous-response, and delayed-trade-off cells. Heart-failure results are withheld until its large-lineage rerun.
- Evaluator sensitivities: null-response, historical monotone, and the earlier composite-adaptive family.
- Primary repeated-dataset OPE: 3,600 independent logged datasets over the four heterogeneous mechanisms and six fixed target policies; 0/2,592 current known-value tuple approvals.
- Secondary null/composite-adaptive repeated grid: 40/1,296 approvals, all AKI null-response sanity cells.
- Real-EHR OPE: not executed for comparison; no heterogeneous repeated-dataset contract and policy-specific EHR gate both pass.
- Independent credentialed reconstruction: pending aggregate-result regeneration from a clean clone; not external validation.

## Tasks

- Reference scaffold: AI-Clinician-aligned sepsis, four-hour fluid-5 × vasopressor-5 action, K=25; not an exact cohort/reward reproduction.
- Frozen sepsis target: 22,437 subjects and 27,236 episodes; its prior 3,440-episode result is superseded and not primary evidence.
- Known-value policy extensions: respiratory support, shock, severe-AKI RRT initiation, and heart failure.
- World-model-only extension: AF/flutter.
- AKI and heart failure retain their original P/T transition surfaces and use separate K2/K8 decision contracts for policy evaluation.
- AF/flutter and heart-failure targets contain 11,820/14,580 and 27,611/32,552 subjects/episodes, respectively; both require complete large-lineage reruns.

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
- Policy: exact heterogeneous-mechanism return/regret, fixed-control-relative return, support violations, divergence, collapse, model-exploitation gap, and planner disagreement.
- OPE: repeated-dataset empirical coverage with Wilson precision intervals, rank/order/sign recovery, false-improvement probability, ESS, support, and finiteness. Fixed-dataset interval inclusion is diagnostic-only.

## Machine-readable audit surfaces

- Complete layer-specific inventory: six transition methods, five uncertainty-capable methods, 34 policy/planner labels, and nine OPE estimators.
- Complete current six-method task-balanced table plus all 18 primary method--cohort rows; the 18 superseded small-lineage rows are historical only.
- Fixed primary factual-action estimand and per-row episode-bootstrap interval counts.
- Task-balanced one-step and recursive uncertainty summary.
- All 3,060 current heterogeneous policy--seed rows and 2,160 component-model--planner rows, plus separately labeled superseded/historical sensitivities.
- All 15,552 current heterogeneous repeated-dataset policy coverage rows and 2,592 OPE tuple dispositions; null/composite-adaptive and fixed-dataset diagnostics remain separately labeled.

## Intended use

- Development-stage benchmarking of computational pipelines.
- Evaluator, support, uncertainty, and exploitation audits.
- Task-contract comparison and reproducibility testing.

## Scope boundary

- No causal effect, retrospective policy-value, deployment, or external-site claim.
- Known-value results characterize constructed mechanisms; retrospective EHR rows characterize prediction, support, collapse, and observability only.
