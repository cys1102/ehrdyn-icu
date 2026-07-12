# sepsis_original25_reference

- Cohort: `sepsis_reference`
- Frozen role: `frozen_reference`
- Clinical anchor: suspected infection plus SOFA>=2
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
