# Independent Review Packet: af_rate_rhythm_k25

> This packet supports independent ICU/cardiac construct review. It is not clinical approval.

## Cohort And Episode

- Cohort: `af_flutter_rate_control`
- Frozen role: `axis_limited_transition_candidate_not_full_core`
- Anchor and direction: AF/flutter diagnosis with first rhythm/rate-control exposure when present; 72h window `[-24h,+48h]` around the anchor, 18 x 4h bins, window-overlaps-stay.
- Inclusion: Adult non-sepsis E060-eligible ICU stay with AF/flutter diagnosis evidence.
- Exclusion: Exclude E060 sepsis-reference stays and non-overlapping anchor windows.
- Aggregate flow: eligible episodes=14580; subjects=11820; windows=262440. Stage-specific excluded counts are unavailable.
- Multiple-anchor rule: one frozen clinical anchor per ICU stay; exact competing-anchor/tie-break counts are not exported and require reviewer resolution.
- Cross-label overlap: unavailable from aggregate artifacts; no zero-overlap claim is made.

## Observations And Missingness

- Vitals and labs use source-specific canonical 4h aggregation, generally median for chart/lab values; treatments retain their documented sum/max/overlap aggregation.
- State channels are value, observation mask, and log-recency. Preprocessing uses training-only statistics, train median then within-episode past-only LOCF, and no future/backward fill.
- Missing measurement is distinct from physiological zero. For treatment exposures, zero means no qualifying recorded exposure only where the action mapping explicitly says so.

## Actions

- Released primary action families: `rate_control;rhythm_control`; K=25.
- Timing: current 4h exposure predicts next 4h state. Within-bin ordering is unresolved; current-window association is not a treatment effect.
- `rate_control`: source=prescriptions+inputevents rate-control-specific regex; unit=overlap fraction per 4h bin; aggregation=summed interval overlap; 5-bin edges=1;1;1.75; counts=bin0=118796;bin1=12917;bin2=0;bin3=35260;bin4=16735; start=0.055394174130557576; stop=0.045653652407465044; change=0.24068310451752717; disposition=promote_to_kdd087_repeated_action.
- `rhythm_control`: source=prescriptions+inputevents rhythm-control-specific regex; unit=overlap fraction per 4h bin; aggregation=summed interval overlap; 5-bin edges=1;1;2; counts=bin0=142404;bin1=6132;bin2=0;bin3=23917;bin4=11255; start=0.032858410854053555; stop=0.024264849972910976; change=0.12527233115468409; disposition=promote_to_kdd087_repeated_action.
- `anticoagulation`: source=prescriptions anticoagulant regex; unit=overlap fraction per 4h bin; aggregation=summed interval overlap; 5-bin edges=1;1;1; counts=bin0=87460;bin1=0;bin2=0;bin3=0;bin4=96248; start=0.035452040898663995; stop=0.01815541031227306; change=0.05360745121093705; disposition=keep_as_context_or_few_step_action.
- families=rate_control;rhythm_control; theoretical_K=25; empirical_K=16; occupied=0.64; top1=0.5425729962767; minimum_nonzero_train_count=525; median_nonzero_train_count=3287.0

## Reward And Terminal Handling

- `terminal_only` (unavailable): final transition receives +1 survival or -1 90d mortality
- `intermediate_physiology` (unavailable): not frozen
- Terminal handling: when available, the final transition receives +1 for 90d survival or -1 for 90d mortality; prior transitions receive no terminal reward.
- Composite reward, when available, is terminal + 0.25 x clipped intermediate physiology. Components must be reported separately.

## Subgroup Warning

- KDD073 label: `benchmark_row_with_subgroup_warning`; material gap count=2; warning axes=transition_error.
- Subgroup evidence is aggregate and confounded; it does not establish equitable treatment effects.

## Unresolved Construct Validity

- Cohort-label overlap counts are unavailable from the aggregate release and require restricted identifier-bearing linkage.
- The exact tie-break among multiple qualifying anchors is not exported as an aggregate receipt; one frozen anchor per stay is assumed by the released episode contract.
- Current-window exposure may be concurrent with physiology recorded in the same 4h bin; it is not a causal treatment assignment.
- Administrative/structured phenotypes may not match clinician-adjudicated disease states.
- No frozen reward or OPE surface is available; the card is transition-only and rhythm/rate medication indication is not adjudicated.
- At least one reward component is unavailable; no missing reward is inferred or scalarized.

## Reviewer Response

Complete the adjacent `reviewer_response.csv`. Allowed decision values: `approve`, `approve_with_change`, `disagree`.

Claim boundary: Independent ICU/cardiac construct review packet only; not clinical approval. No cohort or task contract change, treatment recommendation, causal effect, counterfactual benefit, policy improvement, clinical utility, deployment, or autonomous-decision claim.
