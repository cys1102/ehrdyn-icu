---
title: Cohort Feature Action and Reward Contracts
created: 2026-07-14
updated: 2026-07-15
type: writing
tags: [benchmark-contract, ehr, action, reward]
status: active
confidence: high
---

# Cohort, feature, action, and reward contracts

## Common EHR P/R/T component temporal contract

- Decision interval: four hours.
- State cutoff: strictly before action start.
- Action interval: `[t,t+4h)` recorded exposure or setting summary.
- Next-state target: begins at `t+4h`; one-step target is `[t+4h,t+8h)`.
- Eligibility: state, action, and target bins must remain inside one ICU stay.
- Forward fill: past observations only.
- Fit roles: imputation, normalization, feature selection, action edges, support masks, nuisance models, behavior denominators, hyperparameters, and checkpoints use development training/validation only.
- Split unit: subject-disjoint development roles; not a confirmatory holdout.

## Common state contract

- Timed dynamic physiology and laboratory values.
- Observation masks and feature-specific recency/time gaps.
- Static and pre-anchor context when auditable.
- Time/step index and previous action.
- Current action, post-action measurements, future masks, outcome fields, and whole-stay diagnosis proxies are prohibited.
- Context-only variables are not silently treated as next-state prediction targets.

## Action-label semantics

- The retrospective label is a recorded-exposure abstraction over the four-hour interval, not a claim that the complete label was assignable at interval start.
- It is used for logged-action-conditioned transition evaluation and policy support/collapse diagnostics.
- Assignable actions and true counterfactual values exist only inside the constructed known-value environments.
- No retrospective EHR policy value is inferred from the interval label.

## Sepsis: AI-Clinician-aligned K25 scaffold

- Population/anchor: adult ICU sepsis with time-stamped suspected infection followed by qualifying organ-dysfunction increase.
- Representation window: 24 hours before anchor to 48 hours after anchor, truncated by stay/outcome boundaries.
- Reference features: paper-listed 40 variables plus separate masks/time gaps and prior action.
- Action: training-derived fluid quintile × vasopressor quintile, K=25; zero plus four positive-dose quantiles per dimension.
- Reward tracks: locally parameterized SOFA/lactate shaping; terminal-only sensitivity; lactate-only diagnostic `-tanh(lactate[t+1]-lactate[t])`; treatment-isolated physiology composite; historical sensitivities.
- Critical gaps: numerical coefficients are local parameterization; full SOFA can mechanically encode vasopressor exposure; anchor-relative 90-day terminal outcome is unavailable; the contract is not an exact paper reproduction.
- Frozen scale target: 22,437 subjects and 27,236 episodes. The earlier 3,440-episode strict construction is superseded; no performance metric is transferred between constructions. Full P/R/T, uncertainty, action-information, policy-support, and applicable OPE diagnostics must be rerun before sepsis re-enters primary synthesis.

## Respiratory support

- State module: gas exchange, respiratory support/settings, ABG timing, hemodynamics, neurologic/sedation and renal context, and observation process where available.
- Action: observed PEEP-5 × observed FiO2-5 (`K=25`); bins missing either setting are excluded rather than mapped to no action; 12 classes pass the train-frozen support rule.
- Primary reward: `I(94<=SpO2[t+1]<=98)-0.5 I(outside) + I(70<=MBP[t+1]<=80)-0.5 I(outside)`.
- Sensitivity: SpO2 component alone. PF-ratio shaping is prohibited because future FiO2 mechanically overlaps the action.
- Role: task-specific policy diagnostic in known-value environments; no retrospective EHR policy value.

## Shock

- State module: perfusion/hemodynamics, MAP, lactate, urine output, shock index, fluid balance, organ dysfunction, and observation density.
- Action: fluid-5 × vasopressor-5, K=25.
- Reward components: `clip((MBP[t+1]-65)/25,-1,1)` and `-tanh(lactate[t+1]-lactate[t])`, retained separately; same-bin MAP/vasopressor circularity is excluded.
- Role: task-specific policy diagnostic in known-value environments; no retrospective EHR policy value.

## AF/flutter

- State module: rate/rhythm, hemodynamics, medications/procedures, anticoagulation context, and timing.
- Action: rate-control-5 × rhythm-control-5 transition view.
- Reward: a four-hour rate-control proxy was materialized, but strict action--reward ordering remains unresolved.
- Role: world-model only.
- Frozen scale target: 11,820 subjects and 14,580 episodes; large-lineage P/T and uncertainty rerun required.

## Acute kidney injury

- State module: creatinine, urine output, fluid balance, hemodynamics, nephrotoxic exposure, RRT timing, and observation process.
- World-model action: compact K=4 transition context.
- Separate policy-task action: daily wait versus initiate RRT (`K=2`), maximum three decisions, absorbing after initiation.
- Policy-task objective: in-hospital survival/discharge benchmark sensitivity; renal physiology and fluid-burden surfaces remain separate.
- Role: P/T world-model surface plus task-matched known-value policy comparison; no retrospective EHR policy value.

## Heart failure

- State module: congestion/decongestion, urine output, weight when available, diuretic exposure, blood pressure, renal deterioration, and respiratory support.
- World-model action: compact K=2 transition context.
- Separate policy-task action: six-hour loop-diuretic quartile × vaso/inotrope indicator (`K=8`), maximum 12 decisions.
- Policy-task objective: observed post-action urine-output proxy; creatinine, MAP, and terminal surfaces remain separate.
- Role: P/T world-model surface plus task-matched known-value policy comparison; no retrospective EHR policy value.
- Frozen scale target: 27,611 subjects and 32,552 episodes; all affected component and constructed-profile diagnostics must be rerun.
