# Canonical Evidence Surfaces

EHRDyn-ICU keeps task discovery, headline dynamics, confirmation, portability,
and policy diagnostics separate. Results from different rows below must not be
pooled or silently substituted for one another.

| Surface | Frozen content | Purpose | Selection status |
| --- | --- | --- | --- |
| Core dynamics | Five primary compact-action tasks, 41 controlled contracts, 11 forecasting methods | One-step versus conditional-recursive method ranking | Headline frozen benchmark surface |
| Rich-action task audit | Exact sepsis 25-action reference plus promoted non-sepsis 25-action views and exclusions | Action support, task-role, and sepsis-relative diagnostics | Separate audit; never replaces core rankings |
| Temporal lockbox | Five primary tasks and six representative methods on a later-period patient-disjoint split | No-retuning rank confirmation | Confirmation only; cannot select or retune models |
| eICU portability pilot | Two nonidentically mapped tasks and 26 baseline-seed fits | Evaluator portability | Not phenotype-identical external validation |
| Compact policy diagnostics | Five eligible compact-action tasks, 13--14 completed policy families, kNN and neural behavior denominators | Support, ESS, WIS/WPDIS, clipping, and FQE fragility | Diagnostic only; no policy winner |

The seven paper task IDs map one-to-one to `configs/tasks/` through
`contracts/paper_task_manifest.csv`. All 41 headline contracts map to the 533
public method rows through `contracts/paper_contract_manifest.csv`. The separate
K25 and exclusion configs are retained under `configs/rich_action/` and cannot
replace a compact paper task.

## Method Eligibility

The headline ranking compares **forecasting methods**, including persistence and
previous-window controls where they are eligible. It is not restricted to
learned architectures. `implementation_label` and baseline fidelity must be
reported for every row.

## Recursive Forecast Contract

The recursive evaluator starts from the first recorded state, feeds predicted
values into subsequent transitions, and advances recency causally. Logged future
actions are held fixed. The result is therefore a conditional recursive
logged-action forecast, not an autonomous rollout under a candidate policy.

## Aggregate Evidence

- `evidence/core/contract_transition_leaderboard.csv`: 533 controlled method-contract rows.
- `evidence/core/horizon_rank_stability.csv`: one-step versus recursive ranks and seed comparison.
- `evidence/rich_action/`: exact-reference task roles and fixed-family transition evidence.
- `evidence/uncertainty/`: same-family NLL, Cov90, and Width90 by horizon.
- `evidence/temporal/`: no-retuning temporal rank confirmation.
- `evidence/external/`: bounded eICU mapping and evaluator-portability evidence.
- `evidence/quarantine/policy/`: behavior, overlap, OPE, and FQE diagnostics.
- `evidence/audits/`: post-freeze headline, inference-unit, and OPE provenance audits.

## Post-Freeze Audit Boundary

KDD091 confirms the all-method headline and the learned-only sensitivity.
KDD093 does not establish subject-cluster robustness. KDD094 supports WIS and
WPDIS but blocks CWPDIS availability, exact KDD078 probability-surface replay,
and claims of one universal bootstrap/FQE-selection convention.
