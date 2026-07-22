# Frozen KDD235A recursive component entrant contract

KDD235A is a constructed-workflow interface and smoke validation. It adds no EHR truth,
clinical recommendation, causal claim, or arbitrary-code correctness claim.

## Frozen lineage

- Public base: `f030c5793e4519a2b961eda1085d0bbeef29dbd0`.
- Public policy entrant: KDD215R capability receipt plus KDD216R clean-clone receipt.
- Generator/direct return: repaired KDD198-v2 and KDD199 contracts imported by the public
  full-suite manifest.
- Prediction-to-planning: KDD224/KDD232 validation definitions and
  `H4_support_only_sequence_categorical_CEM`.
- H4: horizon 4, 64 sequences, 3 categorical-CEM iterations, 8 elites, smoothing 0.2,
  support masking at every step, first-action execution, stable tie ordering, and planner
  seed `1903408 + 1009 * environment_seed`.

## Entrant boundary

The typed `kdd235a.runtime.v1` subprocess JSONL interface supplies public environment metadata,
observable histories, observation masks, recency, previous discrete actions, proposed supported
action sequences, and an injected deterministic seed. It never supplies latent state, subtype,
future observations, true values, or final returns. Fit/load is entrant-owned. Only checkpoint ID,
entrant metadata, and source commit enter aggregate results; model weights do not.

Transition is entrant-supplied. Reward and termination are each declared independently as entrant
or benchmark supplied, and mismatched payloads fail. Point, independent-Gaussian, and Gaussian-
ensemble predictions are versioned. Ensemble total variance is exactly within-model plus
between-model variance. Floating semantic comparisons use `rtol=atol=1e-12`; policy sums use
absolute tolerance `1e-8`. Canonical JSON quantizes to 12 decimal places and is frozen for Python
3.11, 3.12, and 3.13.

## Evaluation definitions

Observed cells alone enter one-step and common-origin recursive RMSE. Probabilistic entrants add
Gaussian CRPS, empirical 50/80/90/95% coverage, interval widths, MACE, and risk-coverage area over
retention fractions 0.1 through 1.0. Point entrants receive structural NA for these metrics, never
invented dispersion. Planning produces complete supported policy probabilities for the existing
direct-return and repeated-dataset OPE surfaces. The smoke executes the frozen H4 recursive
entrant policy for direct return under common random numbers, reports learned-model return and
their absolute gap, and does not select a winner.

## Smoke scope

The development smoke uses sepsis (K=25) and AKI (K=4), environment seed 171901, a deterministic
point entrant and an independent-Gaussian entrant. It is deliberately bounded and is not the full
40-environment validation authorized as a later stage.
