# aki_no_rich_action_exclusion

- Cohort: `aki_diuretic_fluid_balance`
- Frozen role: `no_rich_action_exclusion`
- Clinical anchor: AKI or renal-deterioration anchor
- Episode: adult ICU stay, 72h anchor window, 18 x 4h bins, window-overlaps-stay
- State: canonical 40 feature values, masks, and log-recency channels
- Action families: `none`
- Action count: 0
- Allowed tracks: `documentation`
- Dynamics NI: False
- Uncertainty parity: False
- Frozen reward available: False
- OPE available: False

## Claim Boundary

Benchmark construction and recorded-trajectory forecasting, uncertainty, action-support, and offline-policy diagnostics only. Treatment, causal, counterfactual, policy-improvement, clinical-utility, deployment, and autonomous-decision claims are blocked.
