# KDD-RV successor contract

`KDD-RV-SUCCESSOR-RC1` is a separate, local release-candidate surface for five
cohorts: sepsis, respiratory support, AKI, AF/flutter, and heart failure. It
does not alter the historical KDD089 task identities or evidence.

## Frozen evaluation contract

- Subject roles use SHA-256 over the frozen salt plus the decimal local entity
  key, with train/validation/sealed-test ranges 70/15/15.
- State and target cells use 33 permitted features. Mean and population
  standard deviation are fitted on observed train pre-action cells only.
- Primary point scoring is cell-micro normalized RMSE over observed target
  cells. MAE on the same normalized cells is reported alongside it using the
  frozen RV02R field name `mae`.
- Modes are `one_step` and `conditional_recursive`. Recursive segments start
  once from logged pre-action history, feed prior means into later states, keep
  logged masks, recencies, and actions, and reset only after a sequence or
  relative-index gap.
- Gaussian diagnostics use native scales after normalization. Three-seed
  Gaussian backends use moment-matched total variance before row export.
- Inference uses 1,000 paired subject-cluster bootstrap replicates with seed
  8,813,408. The release exports aggregate intervals and practical leader sets,
  not cluster keys or replicate membership. Fixed-test-reference leader sets
  are descriptive; the evaluator always sets
  `unique_winner_claim_allowed=false`.

## Restricted prediction columns

The local CSV requires a subject-cluster key, sequence key, role, task, mode,
method, transition index, horizon, feature, target, observation mask,
prediction mean, optional positive prediction standard deviation, and recorded
action class. Key-column names are configurable for local evaluation. Neither
keys nor split membership appears in aggregate output.

For credentialed evaluation, every row must have role `sealed_test`. Synthetic
fixture rows use separate synthetic key names and role `synthetic`.

Every invocation also requires a frozen evaluation contract created by the
benchmark operator before prediction scoring. It binds the task version,
task/mode target-cell digest, exact 33-feature set, horizons, realized action
classes, subject/transition counts, and normalization receipt. One-step rows
must have horizon one; conditional-recursive horizons increment only within
consecutive segments. The realized respiratory action classes are `0`, `1`,
and `2` after duplicate train-quantile edges collapse.

Credentialed normalization receipts must bind the task/version, source commit,
split-contract digest, and construction receipt. The evaluator emits a
cryptographic receipt over the input files, target-cell contract, and aggregate
payload. `rv-validate-submission` accepts only that output; arbitrary
self-reported aggregate metrics are rejected. Non-synthetic public aggregates
must include at least 100 subject clusters and 1,000 observed cells per primary
metric row.

Example restricted invocation (paths remain inside the authorized environment):

```bash
ehrdyn-icu rv-evaluate-local \
  --predictions /restricted/local_predictions.csv \
  --normalization /restricted/train_normalization_receipt.json \
  --evaluation-contract /restricted/frozen_target_cell_contract.json \
  --output /restricted/aggregate_evaluation.json
ehrdyn-icu rv-validate-submission \
  --submission /restricted/aggregate_evaluation.json \
  --config-dir successor/configs/tasks
```

## Claim and release boundary

This surface supports recorded-trajectory forecasting and benchmark-validity
auditing. It does not support treatment recommendation, causal action effects,
counterfactual treatment simulation, policy selection, clinical utility,
deployment, or autonomous decisions.

Clinical review remains pending and is not simulated. Independent
credentialed reproduction, final-paper parity, institutional determination,
fairness auditing, and public release remain external gates.

The frozen task table also has known contract/code discrepancies that block
promotion without a versioned scientific repair. See
[`CONTRACT_DISCREPANCIES.md`](CONTRACT_DISCREPANCIES.md).

The five blank external-review rows and reviewer template are under
`clinical_review/`; no response or decision is prefilled.

Validated aggregate-only RV01R/RV02R tables are packaged under `evidence/` with
byte-level source hashes. Restricted rows, bootstrap replicate membership,
preprocessing values, checkpoints, and test-opening material are excluded.
