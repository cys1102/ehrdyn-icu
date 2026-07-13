# Offline-Policy Diagnostic Documentation

This document describes the frozen aggregate KDD078/KDD079 diagnostics after the
KDD094 code-to-document audit. It is explanatory documentation, not an exact
target-policy replay package and not evidence of causal treatment-policy value.

## Quarantine Status

Offline-policy rows are not accepted by the public leaderboard submission
validator in v1.1.0. They remain frozen aggregate stress tests because exact
KDD078 target-policy probability surfaces are unavailable and no public
task-policy registry can reproduce every reported value. A future policy track
requires a new version with policy probabilities, task-policy-denominator row
keys, a frozen FQE protocol, and subject-cluster inference. It must not silently
reuse the current diagnostics.

## Population, Horizon, And Reward

- Population: held-out test episodes from each support-eligible frozen task.
- Initial-state distribution: the first test transition of each episode.
- Horizon: 17 transitions from 18 four-hour bins.
- Discount: `gamma = 0.99`.
- Termination: the final transition is terminal; no separate censoring model is
  implemented.
- Canonical KDD079 reward: `terminal_plus_intermediate`.
- Terminal reward: `+1` for 90-day survival and `-1` for 90-day mortality.
- Composite reward: terminal reward plus `0.25` times the clipped task-specific
  intermediate physiology change. Component definitions remain observable
  proxies, not validated clinical utilities.

## Policies And Behavior Denominators

`evidence/quarantine/policy/baseline_catalog.csv` is a policy-family catalog, not an exact
registry of KDD078 probability surfaces. KDD078 reconstructs policy IDs,
configurations, and seeds, but the original probability surfaces were not
retained. KDD079 fits and freezes its own rows. Learned policies use seeds
`3408`, `3411`, and `3414`; fidelity labels are never pooled.

Two behavior denominators are fit on training data and evaluated separately:

- kNN frequency: `k=64`, Euclidean distance in a train-standardized fixed random
  projection, additive smoothing 1;
- neural classifier: MLP `32-64-K`, five epochs, validation-only temperature
  scaling.

Behavior features are projected current values, masks, `log1p` recency/deltas,
previous action, and normalized step. Aggregate calibration is reported by task
and split. The two denominators are never averaged. Naive deterministic controls
use `epsilon=0.001`, not exact point masses.

## Importance Sampling

For logged action `a_it`, evaluation policy `pi_e`, and behavior policy `pi_b`,

```text
rho_it = pi_e(a_it | h_it) / max(pi_b(a_it | h_it), 1e-12)
w_i,t  = product_{j=0}^t rho_ij
```

WIS normalizes final trajectory weights across episodes and weights each
discounted return. WPDIS normalizes cumulative weights separately at each step.
ESS uses final trajectory weights:

```text
ESS = (sum_i w_i,H)^2 / sum_i w_i,H^2
```

Unclipped results are primary diagnostics. Ratio clipping is evaluated at `1`,
`2`, `5`, `10`, `20`, and `50`. ESS is reported at horizons `1`, `2`, `4`, `8`,
`12`, and `17`; this is not per-decision ESS at every step. CWPDIS is not
implemented or exported in KDD078/KDD079.

## Inference And FQE

- KDD079 WIS/WPDIS intervals use 100 episode-bootstrap replicates; KDD078 uses
  200. Neither establishes subject-cluster robustness.
- Linear and neural FQE use training transitions and evaluate the target policy
  on initial held-out test states.
- Neural FQE uses three seeds, two 64-unit hidden layers, eight epochs, batch
  size 2048, AdamW learning rate `5e-4`, target updates once per epoch, and
  `gamma=0.99`.
- Linear and neural FQE are both reported. No frozen selector chooses one as the
  preferred estimand. Finite FQE is a software diagnostic, not proof of correct
  clinical value.

## Required Reporting

Every row should identify task, action contract, reward, policy ID, policy seed,
behavior denominator, horizon, clipping, estimator, bootstrap unit, ESS, weight
concentration, and fidelity label. Exact KDD078 policy-specific attribution is
blocked where probability provenance is unavailable. Denominator or estimator
rank disagreement blocks a policy-winner declaration. See
`evidence/audits/kdd094/` for the authoritative mismatch matrix.

The machine-readable status is in `contracts/ope_provenance_registry.csv`.
Missing probability provenance is an availability block, not a negative
policy-performance result.
