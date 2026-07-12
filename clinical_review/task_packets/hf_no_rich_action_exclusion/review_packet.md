# Independent Review Packet: hf_no_rich_action_exclusion

> This packet supports independent ICU/cardiac construct review. It is not clinical approval.

## Cohort And Episode

- Cohort: `hf_diuretic_decongestion`
- Frozen role: `no_rich_action_exclusion`
- Anchor and direction: First decongestion, diuretic, or volume-overload anchor; 72h window `[-24h,+48h]` around the anchor, 18 x 4h bins, window-overlaps-stay.
- Inclusion: Adult non-sepsis E060-eligible ICU stay with heart-failure or volume-overload evidence.
- Exclusion: Exclude E060 sepsis-reference stays and non-overlapping anchor windows.
- Aggregate flow: eligible episodes=32552; subjects=27611; windows=585936. Stage-specific excluded counts are unavailable.
- Multiple-anchor rule: one frozen clinical anchor per ICU stay; exact competing-anchor/tie-break counts are not exported and require reviewer resolution.
- Cross-label overlap: unavailable from aggregate artifacts; no zero-overlap claim is made.

## Observations And Missingness

- Vitals and labs use source-specific canonical 4h aggregation, generally median for chart/lab values; treatments retain their documented sum/max/overlap aggregation.
- State channels are value, observation mask, and log-recency. Preprocessing uses training-only statistics, train median then within-episode past-only LOCF, and no future/backward fill.
- Missing measurement is distinct from physiological zero. For treatment exposures, zero means no qualifying recorded exposure only where the action mapping explicitly says so.

## Actions

- Released primary action families: `none`; K=0.
- Timing: current 4h exposure predicts next 4h state. Within-bin ordering is unresolved; current-window association is not a treatment effect.
- `diuretic`: source=prescriptions+inputevents diuretic regex; unit=overlap fraction per 4h bin; aggregation=summed interval overlap; 5-bin edges=1;1;1; counts=bin0=283876;bin1=0;bin2=0;bin3=0;bin4=127010; start=0.04370984824472567; stop=0.022455348284668053; change=0.06616519652939373; disposition=keep_as_context_or_few_step_action.
- `vasodilator`: source=prescriptions+inputevents vasodilator regex; unit=overlap fraction per 4h bin; aggregation=summed interval overlap; 5-bin edges=1;1;1; counts=bin0=237479;bin1=0;bin2=0;bin3=0;bin4=173407; start=0.03570848762688148; stop=0.015647105208228644; change=0.051355592835110125; disposition=keep_as_context_or_few_step_action.
- `inotrope_pressor`: source=inotrope interval OR E060 vasopressor exposure; unit=overlap fraction per 4h bin; aggregation=summed interval overlap; 5-bin edges=1;1;2; counts=bin0=349132;bin1=0;bin2=0;bin3=39587;bin4=22167; start=0.022823848950803874; stop=0.01743549305646822; change=0.048642087929928184; disposition=keep_as_context_or_few_step_action.
- families=diuretic;vasodilator;inotrope_pressor; theoretical_K=27; empirical_K=8; occupied=0.2962962962962963; top1=0.3601972323223473; minimum_nonzero_train_count=11534; median_nonzero_train_count=32287.5

## Reward And Terminal Handling

- `terminal_only` (unavailable): final transition receives +1 survival or -1 90d mortality
- `intermediate_physiology` (unavailable): not frozen
- Terminal handling: when available, the final transition receives +1 for 90d survival or -1 for 90d mortality; prior transitions receive no terminal reward.
- Composite reward, when available, is terminal + 0.25 x clipped intermediate physiology. Components must be reported separately.

## Subgroup Warning

- KDD073 label: `benchmark_row_with_subgroup_warning`; material gap count=7; warning axes=transition_error.
- Subgroup evidence is aggregate and confounded; it does not establish equitable treatment effects.

## Unresolved Construct Validity

- Cohort-label overlap counts are unavailable from the aggregate release and require restricted identifier-bearing linkage.
- The exact tie-break among multiple qualifying anchors is not exported as an aggregate receipt; one frozen anchor per stay is assumed by the released episode contract.
- Current-window exposure may be concurrent with physiology recorded in the same 4h bin; it is not a causal treatment assignment.
- Administrative/structured phenotypes may not match clinician-adjudicated disease states.
- No rich repeated action contract passed; diuretic change frequency narrowly missed the frozen support threshold.
- At least one reward component is unavailable; no missing reward is inferred or scalarized.

## Reviewer Response

Complete the adjacent `reviewer_response.csv`. Allowed decision values: `approve`, `approve_with_change`, `disagree`.

Claim boundary: Independent ICU/cardiac construct review packet only; not clinical approval. No cohort or task contract change, treatment recommendation, causal effect, counterfactual benefit, policy improvement, clinical utility, deployment, or autonomous-decision claim.
