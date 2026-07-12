# shock_fluid_vasopressor_k25

- Cohort: `shock_fluid_bolus_support`
- Frozen role: `hard_transition_stress_test`
- Clinical anchor: sustained hypotension or shock-support anchor
- Episode: adult ICU stay, 72h anchor window, 18 x 4h bins, window-overlaps-stay
- State: canonical 40 feature values, masks, and log-recency channels
- Action families: `fluid;vasopressor`
- Action count: 25
- Allowed tracks: `point_transition;uncertainty;policy_diagnostic`
- Dynamics NI: False
- Uncertainty parity: False
- Frozen reward available: True
- OPE available: True

## Claim Boundary

Benchmark construction and recorded-trajectory forecasting, uncertainty, action-support, and offline-policy diagnostics only. Treatment, causal, counterfactual, policy-improvement, clinical-utility, deployment, and autonomous-decision claims are blocked.
