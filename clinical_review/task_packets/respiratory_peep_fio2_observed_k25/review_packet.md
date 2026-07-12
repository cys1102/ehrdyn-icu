# Independent Review Packet: respiratory_peep_fio2_observed_k25

> This packet supports independent ICU/cardiac construct review. It is not clinical approval.

## Cohort And Episode

- Cohort: `respiratory_ventilator_settings`
- Frozen role: `repaired_action_transition_observation_stress_test`
- Anchor and direction: First structured ventilation or respiratory-support anchor; 72h window `[-24h,+48h]` around the anchor, 18 x 4h bins, window-overlaps-stay.
- Inclusion: Adult non-sepsis E060-eligible ICU stay with respiratory failure or ventilation support evidence.
- Exclusion: Exclude E060 sepsis-reference stays and windows without the stay-overlap contract.
- Aggregate flow: eligible episodes=16389; subjects=15312; windows=295002. Stage-specific excluded counts are unavailable.
- Multiple-anchor rule: one frozen clinical anchor per ICU stay; exact competing-anchor/tie-break counts are not exported and require reviewer resolution.
- Cross-label overlap: unavailable from aggregate artifacts; no zero-overlap claim is made.

## Observations And Missingness

- Vitals and labs use source-specific canonical 4h aggregation, generally median for chart/lab values; treatments retain their documented sum/max/overlap aggregation.
- State channels are value, observation mask, and log-recency. Preprocessing uses training-only statistics, train median then within-episode past-only LOCF, and no future/backward fill.
- Missing measurement is distinct from physiological zero. For treatment exposures, zero means no qualifying recorded exposure only where the action mapping explicitly says so.

## Actions

- Released primary action families: `peep;fio2`; K=25.
- Timing: current 4h exposure predicts next 4h state. Within-bin ordering is unresolved; current-window association is not a treatment effect.
- `peep`: source=chartevents itemids 220339/224700; unit=cmH2O; aggregation=4h median charted setting; 5-bin edges=5;5;6; counts=bin0=762;bin1=915;bin2=0;bin3=39425;bin4=13934; start=0.007116290602529135; stop=0.016315397966774114; change=0.3204810314902058; disposition=promote_to_kdd087_repeated_action.
- `fio2`: source=chartevents itemids 223835/226754/227010/229280; unit=percent after fraction-to-percent conversion; aggregation=4h median charted setting; 5-bin edges=40;50;55; counts=bin0=0;bin1=7188;bin2=23682;bin3=19477;bin4=16925; start=0.0; stop=0.0; change=0.3961726225346612; disposition=promote_to_kdd087_repeated_action.
- families=peep;fio2; theoretical_K=25; empirical_K=16; occupied=0.64; top1=0.29815459868079747; minimum_nonzero_train_count=49; median_nonzero_train_count=508.0

## Reward And Terminal Handling

- `terminal_only` (available): final transition receives +1 survival or -1 90d mortality
- `intermediate_physiology` (available): SpO2/PaO2-FiO2 change
- `terminal_plus_intermediate` (available): terminal + 0.25 * clipped intermediate physiology
- Terminal handling: when available, the final transition receives +1 for 90d survival or -1 for 90d mortality; prior transitions receive no terminal reward.
- Composite reward, when available, is terminal + 0.25 x clipped intermediate physiology. Components must be reported separately.

## Subgroup Warning

- KDD073 label: `benchmark_row_with_subgroup_warning`; material gap count=14; warning axes=ope_stability;transition_error.
- Subgroup evidence is aggregate and confounded; it does not establish equitable treatment effects.

## Unresolved Construct Validity

- Cohort-label overlap counts are unavailable from the aggregate release and require restricted identifier-bearing linkage.
- The exact tie-break among multiple qualifying anchors is not exported as an aggregate receipt; one frozen anchor per stay is assumed by the released episode contract.
- Current-window exposure may be concurrent with physiology recorded in the same 4h bin; it is not a causal treatment assignment.
- Administrative/structured phenotypes may not match clinician-adjudicated disease states.
- PEEP/FiO2 evaluation is restricted to jointly observed pre-action settings; missing settings are not the lowest action class.

## Reviewer Response

Complete the adjacent `reviewer_response.csv`. Allowed decision values: `approve`, `approve_with_change`, `disagree`.

Claim boundary: Independent ICU/cardiac construct review packet only; not clinical approval. No cohort or task contract change, treatment recommendation, causal effect, counterfactual benefit, policy improvement, clinical utility, deployment, or autonomous-decision claim.
