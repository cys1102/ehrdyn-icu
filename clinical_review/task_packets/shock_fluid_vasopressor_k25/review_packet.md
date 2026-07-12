# Independent Review Packet: shock_fluid_vasopressor_k25

> This packet supports independent ICU/cardiac construct review. It is not clinical approval.

## Cohort And Episode

- Cohort: `shock_fluid_bolus_support`
- Frozen role: `hard_transition_stress_test`
- Anchor and direction: First sustained hypotension or vasopressor/shock-support anchor; 72h window `[-24h,+48h]` around the anchor, 18 x 4h bins, window-overlaps-stay.
- Inclusion: Adult non-sepsis E060-eligible ICU stay with hypotension or shock-support evidence.
- Exclusion: Exclude E060 sepsis-reference stays and non-overlapping anchor windows.
- Aggregate flow: eligible episodes=30563; subjects=25970; windows=550134. Stage-specific excluded counts are unavailable.
- Multiple-anchor rule: one frozen clinical anchor per ICU stay; exact competing-anchor/tie-break counts are not exported and require reviewer resolution.
- Cross-label overlap: unavailable from aggregate artifacts; no zero-overlap claim is made.

## Observations And Missingness

- Vitals and labs use source-specific canonical 4h aggregation, generally median for chart/lab values; treatments retain their documented sum/max/overlap aggregation.
- State channels are value, observation mask, and log-recency. Preprocessing uses training-only statistics, train median then within-episode past-only LOCF, and no future/backward fill.
- Missing measurement is distinct from physiological zero. For treatment exposures, zero means no qualifying recorded exposure only where the action mapping explicitly says so.

## Actions

- Released primary action families: `fluid;vasopressor`; K=25.
- Timing: current 4h exposure predicts next 4h state. Within-bin ordering is unresolved; current-window association is not a treatment effect.
- `fluid`: source=ICU inputevents E060 fluid item/rule set; unit=mL per 4h; aggregation=sum bolus starts within bin; 5-bin edges=75.0976810455;240;994.200012207; counts=bin0=283648;bin1=24929;bin2=24919;bin3=24938;bin4=24930; start=0.1076157385670016; stop=0.10201731176083918; change=0.36891892638358753; disposition=promote_to_kdd087_repeated_action.
- `vasopressor`: source=ICU inputevents E060 vasopressor item set; unit=raw rate-or-amount proxy; aggregation=maximum overlapping value per bin; 5-bin edges=0.180357463658;0.599776208401;1.50506296754; counts=bin0=333142;bin1=12556;bin2=12554;bin3=12556;bin4=12556; start=0.02944214590709981; stop=0.024227627007230725; change=0.12533350273154617; disposition=promote_to_kdd087_repeated_action.
- `inotrope`: source=prescriptions+inputevents inotrope regex; unit=overlap fraction per 4h bin; aggregation=summed interval overlap; 5-bin edges=1;1;1; counts=bin0=330144;bin1=0;bin2=0;bin3=0;bin4=53220; start=0.01568222368297493; stop=0.008153209635812255; change=0.023835433318787184; disposition=keep_as_context_or_few_step_action.
- families=fluid;vasopressor; theoretical_K=25; empirical_K=25; occupied=1.0; top1=0.7081937792802663; minimum_nonzero_train_count=1589; median_nonzero_train_count=2655.0

## Reward And Terminal Handling

- `terminal_only` (available): final transition receives +1 survival or -1 90d mortality
- `intermediate_physiology` (available): MAP/lactate/urine-output/shock-index change
- `terminal_plus_intermediate` (available): terminal + 0.25 * clipped intermediate physiology
- Terminal handling: when available, the final transition receives +1 for 90d survival or -1 for 90d mortality; prior transitions receive no terminal reward.
- Composite reward, when available, is terminal + 0.25 x clipped intermediate physiology. Components must be reported separately.

## Subgroup Warning

- KDD073 label: `benchmark_row_with_subgroup_warning`; material gap count=2; warning axes=behavior_calibration.
- Subgroup evidence is aggregate and confounded; it does not establish equitable treatment effects.

## Unresolved Construct Validity

- Cohort-label overlap counts are unavailable from the aggregate release and require restricted identifier-bearing linkage.
- The exact tie-break among multiple qualifying anchors is not exported as an aggregate receipt; one frozen anchor per stay is assumed by the released episode contract.
- Current-window exposure may be concurrent with physiology recorded in the same 4h bin; it is not a causal treatment assignment.
- Administrative/structured phenotypes may not match clinician-adjudicated disease states.
- The card fails point-dynamics comparability and remains a hard transition stress test.

## Reviewer Response

Complete the adjacent `reviewer_response.csv`. Allowed decision values: `approve`, `approve_with_change`, `disagree`.

Claim boundary: Independent ICU/cardiac construct review packet only; not clinical approval. No cohort or task contract change, treatment recommendation, causal effect, counterfactual benefit, policy improvement, clinical utility, deployment, or autonomous-decision claim.
