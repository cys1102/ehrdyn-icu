# Changelog

## KDD187 public reconstruction update

- Added a Rocky Linux conda setup backed by the frozen `uv.lock` dependency
  graph and one documented public/credentialed reconstruction workflow.
- Added an additive five-task current-lineage manifest, including the KDD184
  corrected sepsis DBP lineage while preserving historical E060 manifests.
- Added explicit aggregate inventory, parity tolerances, privacy boundaries,
  and author-side evidence labels. KDD182 remains externally blocked.
- Registered the completed KDD186 uncertainty-penalty sensitivity as optional
  aggregate evidence rather than a core reconstruction prerequisite.

## Action-construction and parity update

- Prorated medication amounts by event-window overlap and represented RRT/CRRT
  as four-hour procedure-overlap fraction, closing the binary-to-three-level
  construction mismatch.
- Added occupied action cardinality and class histograms to credentialed build
  receipts and the exact parity gate.
- Added direct tests for empirical respiratory `K=4` and the complete AKI
  factorized `3 x 3` encoder.
- Clarified that drug-family values are recorded-exposure proxies, not
  cross-drug dose-equivalent treatment intensities, and that the respiratory
  task has a historical nominal five-bin name but empirical `K=4`.

## Credentialed reconstruction update

- Published credentialed MIMIC-IV SQL and preprocessing for all seven compact
  tasks, including the five primary paper tasks, with exact aggregate parity
  targets and restricted-output protections.
- Added one-to-one paper task and 41-contract manifests with executable
  validation against the 533-row public transition evidence.
- Added a local aggregate evaluator and stricter submission governance fields.
- Removed the policy track and CWPDIS from the public leaderboard validator;
  historical OPE files now live under `evidence/quarantine/policy/`.
- Added compact-task clinician packets, a response template, and a
  machine-readable status table. Independent clinical review remains pending.

## Provenance and clinical-review update

- Added KDD091 headline candidate-set reconciliation.
- Added KDD093 subject-cluster feasibility and KDD094 OPE provenance audits.
- Added seven aggregate-safe KDD092 clinical review packets with blank responses.
- Corrected OPE documentation and explicitly blocked unsupported CWPDIS, exact
  KDD078 probability-surface replay, and subject-cluster robustness claims.

## Rich-action and portability update

- Added aggregate rich-action role and fixed-family transition evidence.
- Added same-family probabilistic uncertainty evidence.
- Added bounded eICU evaluator-portability summaries with nonidentity limits.
- Preserved the immutable scientific task contract and all clinical/causal claim blocks.

## Initial frozen public benchmark update

- Added exact E060 sepsis K=25 reference task.
- Added KDD087B/KDD087R corrected action roles.
- Added separate sepsis-relative and absolute gates.
- Added aggregate submission validation and uncertainty evaluator metrics.
- Preserved all clinical and causal claim blocks.
