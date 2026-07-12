# Independent Review Packet: sepsis_original25_reference

> This packet supports independent ICU/cardiac construct review. It is not clinical approval.

## Cohort And Episode

- Cohort: `sepsis_reference`
- Frozen role: `frozen_reference`
- Anchor and direction: E060 suspected-infection anchor with SOFA >=2; 72h window `[-24h,+48h]` around the anchor, 18 x 4h bins, window-overlaps-stay.
- Inclusion: Adult ICU stay >=24h with computable 90d mortality and an anchor window overlapping the stay.
- Exclusion: Exclude stays failing the E060 suspected-infection plus SOFA >=2 reference phenotype.
- Aggregate flow: eligible episodes=27236; subjects=22437; windows=490248. Stage-specific excluded counts are unavailable.
- Multiple-anchor rule: one frozen clinical anchor per ICU stay; exact competing-anchor/tie-break counts are not exported and require reviewer resolution.
- Cross-label overlap: unavailable from aggregate artifacts; no zero-overlap claim is made.

## Observations And Missingness

- Vitals and labs use source-specific canonical 4h aggregation, generally median for chart/lab values; treatments retain their documented sum/max/overlap aggregation.
- State channels are value, observation mask, and log-recency. Preprocessing uses training-only statistics, train median then within-episode past-only LOCF, and no future/backward fill.
- Missing measurement is distinct from physiological zero. For treatment exposures, zero means no qualifying recorded exposure only where the action mapping explicitly says so.

## Actions

- Released primary action families: `fluid;vasopressor`; K=25.
- Timing: current 4h exposure predicts next 4h state. Within-bin ordering is unresolved; current-window association is not a treatment effect.
- `fluid`: source=exact E060 inputevents mapping; unit=mL; aggregation=sum starts; 5-bin edges=50;176.008407593;482.671783447; counts=bin0=181971;bin1=40399;bin2=41667;bin3=41033;bin4=41034; start=0.057755846253625225; stop=0.02790354752260796; change=0.4580513711621532; disposition=not_in_frozen_marginal_manifest.
- `vasopressor`: source=exact E060 inputevents mapping; unit=raw rate-or-amount proxy; aggregation=maximum overlap; 5-bin edges=0.0999483056366;0.444362640381;2.39999985695; counts=bin0=286797;bin1=14827;bin2=14826;bin3=13426;bin4=16228; start=0.025263402635861916; stop=0.015684846853240984; change=0.13710397826698809; disposition=not_in_frozen_marginal_manifest.
- families=fluid;vasopressor; theoretical_K=25; empirical_K=25; occupied=1.0; top1=0.5195201442341031; minimum_nonzero_train_count=242; median_nonzero_train_count=3658.0

## Reward And Terminal Handling

- `terminal_only` (available): final transition receives +1 survival or -1 90d mortality
- `intermediate_physiology` (available): SOFA/lactate/MAP/urine-output change
- `terminal_plus_intermediate` (available): terminal + 0.25 * clipped intermediate physiology
- Terminal handling: when available, the final transition receives +1 for 90d survival or -1 for 90d mortality; prior transitions receive no terminal reward.
- Composite reward, when available, is terminal + 0.25 x clipped intermediate physiology. Components must be reported separately.

## Subgroup Warning

- KDD073 label: `benchmark_row_with_subgroup_warning`; material gap count=8; warning axes=ope_stability;transition_error.
- Subgroup evidence is aggregate and confounded; it does not establish equitable treatment effects.

## Unresolved Construct Validity

- Cohort-label overlap counts are unavailable from the aggregate release and require restricted identifier-bearing linkage.
- The exact tie-break among multiple qualifying anchors is not exported as an aggregate receipt; one frozen anchor per stay is assumed by the released episode contract.
- Current-window exposure may be concurrent with physiology recorded in the same 4h bin; it is not a causal treatment assignment.
- Administrative/structured phenotypes may not match clinician-adjudicated disease states.
- The exact E060 action grid is observational and diagnostic-only; suspected infection and SOFA proxy are not prospective adjudication.

## Reviewer Response

Complete the adjacent `reviewer_response.csv`. Allowed decision values: `approve`, `approve_with_change`, `disagree`.

Claim boundary: Independent ICU/cardiac construct review packet only; not clinical approval. No cohort or task contract change, treatment recommendation, causal effect, counterfactual benefit, policy improvement, clinical utility, deployment, or autonomous-decision claim.
