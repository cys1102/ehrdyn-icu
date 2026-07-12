# Independent Review Packet: aki_no_rich_action_exclusion

> This packet supports independent ICU/cardiac construct review. It is not clinical approval.

## Cohort And Episode

- Cohort: `aki_diuretic_fluid_balance`
- Frozen role: `no_rich_action_exclusion`
- Anchor and direction: First AKI, RRT, or structured renal-deterioration anchor; 72h window `[-24h,+48h]` around the anchor, 18 x 4h bins, window-overlaps-stay.
- Inclusion: Adult non-sepsis E060-eligible ICU stay with AKI or renal-dysfunction evidence.
- Exclusion: Exclude E060 sepsis-reference stays and non-overlapping anchor windows.
- Aggregate flow: eligible episodes=16453; subjects=12979; windows=296154. Stage-specific excluded counts are unavailable.
- Multiple-anchor rule: one frozen clinical anchor per ICU stay; exact competing-anchor/tie-break counts are not exported and require reviewer resolution.
- Cross-label overlap: unavailable from aggregate artifacts; no zero-overlap claim is made.

## Observations And Missingness

- Vitals and labs use source-specific canonical 4h aggregation, generally median for chart/lab values; treatments retain their documented sum/max/overlap aggregation.
- State channels are value, observation mask, and log-recency. Preprocessing uses training-only statistics, train median then within-episode past-only LOCF, and no future/backward fill.
- Missing measurement is distinct from physiological zero. For treatment exposures, zero means no qualifying recorded exposure only where the action mapping explicitly says so.

## Actions

- Released primary action families: `none`; K=0.
- Timing: current 4h exposure predicts next 4h state. Within-bin ordering is unresolved; current-window association is not a treatment effect.
- `diuretic`: source=prescriptions+inputevents diuretic regex; unit=overlap fraction per 4h bin; aggregation=summed interval overlap; 5-bin edges=1;1;1; counts=bin0=149293;bin1=0;bin2=0;bin3=0;bin4=57113; start=0.03306162440558329; stop=0.02206331211301997; change=0.05512493651860326; disposition=keep_as_context_or_few_step_action.
- `fluid`: source=ICU inputevents E060 fluid item/rule set; unit=mL per 4h; aggregation=sum bolus starts within bin; 5-bin edges=77.0833358765;200.55557251;750; counts=bin0=158151;bin1=12063;bin2=12064;bin3=12008;bin4=12120; start=0.10075459502716234; stop=0.09181846628945466; change=0.3319499946136997; disposition=promote_to_kdd087_repeated_action.
- `rrt_crrt`: source=procedureevents dialysis/CRRT regex; unit=overlap fraction per 4h bin; aggregation=summed interval overlap; 5-bin edges=1;1;1; counts=bin0=195575;bin1=0;bin2=0;bin3=0;bin4=10831; start=0.006453300776140229; stop=0.002749578073140829; change=0.009202878849281056; disposition=keep_as_context_or_few_step_action.
- families=diuretic;fluid;rrt_crrt; theoretical_K=27; empirical_K=12; occupied=0.4444444444444444; top1=0.5248103252812418; minimum_nonzero_train_count=217; median_nonzero_train_count=4538.0

## Reward And Terminal Handling

- `terminal_only` (unavailable): final transition receives +1 survival or -1 90d mortality
- `intermediate_physiology` (unavailable): not frozen
- Terminal handling: when available, the final transition receives +1 for 90d survival or -1 for 90d mortality; prior transitions receive no terminal reward.
- Composite reward, when available, is terminal + 0.25 x clipped intermediate physiology. Components must be reported separately.

## Subgroup Warning

- KDD073 label: `benchmark_row_no_material_subgroup_gap_detected`; material gap count=0; warning axes=none_detected.
- Subgroup evidence is aggregate and confounded; it does not establish equitable treatment effects.

## Unresolved Construct Validity

- Cohort-label overlap counts are unavailable from the aggregate release and require restricted identifier-bearing linkage.
- The exact tie-break among multiple qualifying anchors is not exported as an aggregate receipt; one frozen anchor per stay is assumed by the released episode contract.
- Current-window exposure may be concurrent with physiology recorded in the same 4h bin; it is not a causal treatment assignment.
- Administrative/structured phenotypes may not match clinician-adjudicated disease states.
- Fewer than two repeated action families passed; RRT is few-step and the released task is a no-rich-action exclusion.
- At least one reward component is unavailable; no missing reward is inferred or scalarized.

## Reviewer Response

Complete the adjacent `reviewer_response.csv`. Allowed decision values: `approve`, `approve_with_change`, `disagree`.

Claim boundary: Independent ICU/cardiac construct review packet only; not clinical approval. No cohort or task contract change, treatment recommendation, causal effect, counterfactual benefit, policy improvement, clinical utility, deployment, or autonomous-decision claim.
