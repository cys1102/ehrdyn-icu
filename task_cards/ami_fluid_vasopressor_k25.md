# ami_fluid_vasopressor_k25

- Cohort: `acute_myocardial_infarction_hemodynamic_support`
- Frozen role: `axis_limited_transition_plus_policy_diagnostic_not_full_core`
- Clinical anchor: AMI with hemodynamic-support anchor
- Episode: adult ICU stay, 72h anchor window, 18 x 4h bins, window-overlaps-stay
- State: canonical 40 feature values, masks, and log-recency channels
- Action families: `fluid;vasopressor`
- Action count: 25
- Allowed tracks: `point_transition;uncertainty;policy_diagnostic`
- Dynamics NI: True
- Uncertainty parity: False
- Frozen reward available: True
- OPE available: True

## Claim Boundary

Benchmark construction and recorded-trajectory forecasting, uncertainty, action-support, and offline-policy diagnostics only. Treatment, causal, counterfactual, policy-improvement, clinical-utility, deployment, and autonomous-decision claims are blocked.
