# Independent Review Packet: ami_fluid_vasopressor_k25

> This packet supports independent ICU/cardiac construct review. It is not clinical approval.

## Cohort And Episode

- Cohort: `acute_myocardial_infarction_hemodynamic_support`
- Frozen role: `axis_limited_transition_plus_policy_diagnostic_not_full_core`
- Anchor and direction: AMI diagnosis with first hemodynamic-support exposure when present; 72h window `[-24h,+48h]` around the anchor, 18 x 4h bins, window-overlaps-stay.
- Inclusion: Adult non-sepsis E060-eligible ICU stay with AMI diagnosis evidence.
- Exclusion: Exclude E060 sepsis-reference stays and non-overlapping anchor windows.
- Aggregate flow: eligible episodes=5055; subjects=4440; windows=90990. Stage-specific excluded counts are unavailable.
- Multiple-anchor rule: one frozen clinical anchor per ICU stay; exact competing-anchor/tie-break counts are not exported and require reviewer resolution.
- Cross-label overlap: unavailable from aggregate artifacts; no zero-overlap claim is made.

## Observations And Missingness

- Vitals and labs use source-specific canonical 4h aggregation, generally median for chart/lab values; treatments retain their documented sum/max/overlap aggregation.
- State channels are value, observation mask, and log-recency. Preprocessing uses training-only statistics, train median then within-episode past-only LOCF, and no future/backward fill.
- Missing measurement is distinct from physiological zero. For treatment exposures, zero means no qualifying recorded exposure only where the action mapping explicitly says so.

## Actions

- Released primary action families: `fluid;vasopressor`; K=25.
- Timing: current 4h exposure predicts next 4h state. Within-bin ordering is unresolved; current-window association is not a treatment effect.
- `fluid`: source=ICU inputevents E060 fluid item/rule set; unit=mL per 4h; aggregation=sum bolus starts within bin; 5-bin edges=50;137.960494995;578.397171021; counts=bin0=50556;bin1=3235;bin2=3464;bin3=3349;bin4=3350; start=0.08585950563732389; stop=0.07935299084452244; change=0.29651826956507343; disposition=promote_to_kdd087_repeated_action.
- `vasopressor`: source=ICU inputevents E060 vasopressor item set; unit=raw rate-or-amount proxy; aggregation=maximum overlapping value per bin; 5-bin edges=0.159880787134;0.699343681335;2.00330543518; counts=bin0=55528;bin1=2107;bin2=2105;bin3=2106;bin4=2108; start=0.02791344514163673; stop=0.023194980215559345; change=0.121719839075512; disposition=promote_to_kdd087_repeated_action.
- `inotrope`: source=prescriptions+inputevents inotrope regex; unit=overlap fraction per 4h bin; aggregation=summed interval overlap; 5-bin edges=1;1;1; counts=bin0=53572;bin1=0;bin2=0;bin3=0;bin4=10382; start=0.018443403254912998; stop=0.009486597904008211; change=0.02793000115892121; disposition=keep_as_context_or_few_step_action.
- families=fluid;vasopressor; theoretical_K=25; empirical_K=25; occupied=1.0; top1=0.7569190355568065; minimum_nonzero_train_count=244; median_nonzero_train_count=451.0

## Reward And Terminal Handling

- `terminal_only` (available): final transition receives +1 survival or -1 90d mortality
- `intermediate_physiology` (available): MAP/lactate/creatinine/urine-output change
- `terminal_plus_intermediate` (available): terminal + 0.25 * clipped intermediate physiology
- Terminal handling: when available, the final transition receives +1 for 90d survival or -1 for 90d mortality; prior transitions receive no terminal reward.
- Composite reward, when available, is terminal + 0.25 x clipped intermediate physiology. Components must be reported separately.

## Subgroup Warning

- KDD073 label: `benchmark_row_with_subgroup_warning`; material gap count=9; warning axes=behavior_calibration;ope_stability.
- Subgroup evidence is aggregate and confounded; it does not establish equitable treatment effects.

## Unresolved Construct Validity

- Cohort-label overlap counts are unavailable from the aggregate release and require restricted identifier-bearing linkage.
- The exact tie-break among multiple qualifying anchors is not exported as an aggregate receipt; one frozen anchor per stay is assumed by the released episode contract.
- Current-window exposure may be concurrent with physiology recorded in the same 4h bin; it is not a causal treatment assignment.
- Administrative/structured phenotypes may not match clinician-adjudicated disease states.
- Action support is behavior-denominator sensitive and the AMI/hemodynamic phenotype remains an aggregate structured proxy.

## Reviewer Response

Complete the adjacent `reviewer_response.csv`. Allowed decision values: `approve`, `approve_with_change`, `disagree`.

Claim boundary: Independent ICU/cardiac construct review packet only; not clinical approval. No cohort or task contract change, treatment recommendation, causal effect, counterfactual benefit, policy improvement, clinical utility, deployment, or autonomous-decision claim.
