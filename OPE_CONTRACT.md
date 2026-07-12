# Offline-Policy Diagnostic Contract

This contract defines an observational diagnostic estimand. It does not identify
the causal value of a treatment policy.

## Population, Horizon, And Reward

- Population: held-out test episodes from each support-eligible frozen task.
- Initial-state distribution: the first test transition of each episode.
- Horizon: 17 transitions from 18 four-hour bins.
- Discount: `gamma = 0.99`.
- Termination: the final transition is terminal; no post-terminal continuation.
- Canonical KDD079 reward: `terminal_plus_intermediate`.
- Terminal reward: `+1` for 90-day survival and `-1` for 90-day mortality.
- Composite reward: terminal reward plus `0.25` times the clipped task-specific
  intermediate physiology change. Component definitions remain observable
  proxies, not validated clinical utilities.

## Policies And Behavior Denominators

The target-policy registry is `evidence/policy/baseline_catalog.csv`. Learned
policies use seeds `3408`, `3411`, and `3414`; library, adapted, independent,
conceptual, and control fidelity labels are never pooled.

Two behavior denominators are fit on training data and evaluated separately:

- kNN frequency: `k=64`, Euclidean distance in a fixed-projection latent state,
  additive smoothing 1, validation-selected temperature;
- neural classifier: MLP `32-64-K`, five epochs, validation-only temperature
  scaling.

The state contains current values, masks, log-recency, previous action, and
normalized step. Aggregate calibration is reported by task and split. The two
denominators are never averaged.

## Importance Sampling

For logged action `a_it`, evaluation policy `pi_e`, and behavior policy `pi_b`,

```text
rho_it = pi_e(a_it | h_it) / max(pi_b(a_it | h_it), 1e-6)
w_i,t  = product_{j=0}^t rho_ij
```

WIS normalizes final trajectory weights across episodes and weights each
discounted return. WPDIS normalizes cumulative weights separately at each step.
ESS uses final trajectory weights:

```text
ESS = (sum_i w_i,H)^2 / sum_i w_i,H^2
```

Unclipped results are primary diagnostics. Per-step ratio clipping is evaluated
at `1`, `2`, `5`, and `10`. Report ESS fraction, maximum normalized weight,
top-1% and top-5% concentration, cumulative log-ratio quantiles, and zero/low
support diagnostics with every estimate.

## Inference And FQE

- WIS/WPDIS confidence intervals use 100 episode-bootstrap replicates.
- Linear and neural FQE use training transitions and evaluate the target policy
  on initial held-out test states.
- Neural FQE uses three seeds, two 64-unit hidden layers, eight epochs, batch
  size 2048, AdamW learning rate `5e-4`, target updates once per epoch, and
  `gamma=0.99`.
- Finite FQE is a software diagnostic, not proof of correct clinical value.

## Required Reporting

Every row must identify task, action contract, reward, policy ID, policy seed,
behavior denominator, horizon, clipping, estimator, bootstrap unit, ESS, weight
concentration, and fidelity label. Denominator or estimator rank disagreement
blocks a policy-winner declaration.

