# af_rate_rhythm_k25

- Cohort: `af_flutter_rate_control`
- Frozen role: `axis_limited_transition_candidate_not_full_core`
- Clinical anchor: AF/flutter with rate or rhythm-control need
- Episode: adult ICU stay, 72h anchor window, 18 x 4h bins, window-overlaps-stay
- State: canonical 40 feature values, masks, and log-recency channels
- Action families: `rate_control;rhythm_control`
- Action count: 25
- Allowed tracks: `point_transition`
- Dynamics NI: True
- Uncertainty parity: False
- Frozen reward available: False
- OPE available: False

## Claim Boundary

Benchmark construction and recorded-trajectory forecasting, uncertainty, action-support, and offline-policy diagnostics only. Treatment, causal, counterfactual, policy-improvement, clinical-utility, deployment, and autonomous-decision claims are blocked.
