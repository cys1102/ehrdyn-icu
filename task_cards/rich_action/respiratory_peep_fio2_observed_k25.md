# respiratory_peep_fio2_observed_k25

- Cohort: `respiratory_ventilator_settings`
- Frozen role: `repaired_action_transition_observation_stress_test`
- Clinical anchor: invasive ventilation with jointly observed pre-action PEEP and FiO2
- Episode: adult ICU stay, 72h anchor window, 18 x 4h bins, window-overlaps-stay
- State: canonical 40 feature values, masks, and log-recency channels
- Action families: `peep;fio2`
- Action count: 25
- Allowed tracks: `point_transition`
- Dynamics NI: False
- Uncertainty parity: False
- Frozen reward available: True
- OPE available: False

## Claim Boundary

Benchmark construction and recorded-trajectory forecasting, uncertainty, action-support, and offline-policy diagnostics only. Treatment, causal, counterfactual, policy-improvement, clinical-utility, deployment, and autonomous-decision claims are blocked.
