# AKI / renal dysfunction: factorized_2_action_three_bin

- Benchmark version: `KDD2027-E060-4H-v1.0.0+KDD089`
- Task ID: `kdd2027_aki_diuretic_rrt_factorized_3bin`
- Release tier: `primary`
- Dynamics evaluation: `runnable_public_evaluator`
- Policy evaluation: `quarantined_not_public_leaderboard`

## Cohort Contract

- Clinical anchor: First AKI, RRT, or structured renal-deterioration anchor
- Inclusion: Adult non-sepsis E060-eligible ICU stay with AKI or renal-dysfunction evidence.
- Exclusion: Exclude E060 sepsis-reference stays and non-overlapping anchor windows.
- Episode unit: ICU stay; 72h [-24h,+48h] anchor window; 4h bins; 18 steps; window must overlap the ICU stay.

## Scale Profiles

| profile | train_episodes | validation_episodes | test_episodes | train_windows | validation_windows | test_windows |
| --- | --- | --- | --- | --- | --- | --- |
| full_eligible | 11467 | 2445 | 2541 | 206406 | 44010 | 45738 |
| development_cap5000 | 3460 | 731 | 809 | 62280 | 13158 | 14562 |

The `full_eligible` profile is the official benchmark. `development_cap5000` is a separate cost-controlled profile and cannot replace it.

## State And Missingness

- Feature view: common ICU features plus cohort-specific interpretation fields from `aki_renal_dysfunction`.
- Full aggregate observation fraction: `0.3922242224216461`.
- Full aggregate missing fraction: `0.6077757775783539`.
- Missingness contract: observation mask retained; train-only imputation/scaling; no future-value fill
- Cohort-specific feature subset: mbp, co2_bicarbonate, bun, creatinine, urine_output
- Per-feature full-cohort observation fractions were not exported by KDD067 and remain a KDD069 reporting requirement.

| feature_name | feature_group | feature_role | full_overall_observation_fraction | per_feature_observation_fraction | timing_role | leakage_risk |
| --- | --- | --- | --- | --- | --- | --- |
| age | demographics | common_icu | 0.3922242224216461 | not_exported_by_KDD067 | pre-action/history | low_if_lagged_to_pre_action_history |
| gender_male | demographics | common_icu | 0.3922242224216461 | not_exported_by_KDD067 | pre-action/history | low_if_lagged_to_pre_action_history |
| weight | demographics | common_icu | 0.3922242224216461 | not_exported_by_KDD067 | pre-action/history | low_if_lagged_to_pre_action_history |
| readmission | demographics | common_icu | 0.3922242224216461 | not_exported_by_KDD067 | pre-action/history when used as state | low only when lagged before the current action |
| elixhauser_score_proxy | demographics | common_icu | 0.3922242224216461 | not_exported_by_KDD067 | pre-action/history when used as state | low only when lagged before the current action |
| heart_rate | vital_signs | common_icu | 0.3922242224216461 | not_exported_by_KDD067 | pre-action/history | low_if_lagged_to_pre_action_history |
| sbp | vital_signs | common_icu | 0.3922242224216461 | not_exported_by_KDD067 | pre-action/history | low_if_lagged_to_pre_action_history |
| mbp | vital_signs | common_icu_and_cohort_specific | 0.3922242224216461 | not_exported_by_KDD067 | pre-action/history | low_if_lagged_to_pre_action_history |
| dbp | vital_signs | common_icu | 0.3922242224216461 | not_exported_by_KDD067 | pre-action/history | low_if_lagged_to_pre_action_history |
| respiratory_rate | vital_signs | common_icu | 0.3922242224216461 | not_exported_by_KDD067 | pre-action/history | low_if_lagged_to_pre_action_history |
| temperature_c | vital_signs | common_icu | 0.3922242224216461 | not_exported_by_KDD067 | pre-action/history | low_if_lagged_to_pre_action_history |
| spo2 | vital_signs | common_icu | 0.3922242224216461 | not_exported_by_KDD067 | pre-action/history | low_if_lagged_to_pre_action_history |
| shock_index | vital_signs | common_icu | 0.3922242224216461 | not_exported_by_KDD067 | pre-action/history when used as state | low only when lagged before the current action |
| sofa_proxy | vital_signs | common_icu | 0.3922242224216461 | not_exported_by_KDD067 | pre-action/history when used as state | low only when lagged before the current action |
| gcs_proxy | vital_signs | common_icu | 0.3922242224216461 | not_exported_by_KDD067 | pre-action/history | low_if_lagged_to_pre_action_history |
| fio2 | vital_signs | common_icu | 0.3922242224216461 | not_exported_by_KDD067 | pre-action/history when used as state | low only when lagged before the current action |
| sirs_proxy | vital_signs | common_icu | 0.3922242224216461 | not_exported_by_KDD067 | pre-action/history when used as state | low only when lagged before the current action |
| lactate | lab_values | common_icu | 0.3922242224216461 | not_exported_by_KDD067 | pre-action/history | low_if_lagged_to_pre_action_history |
| pao2 | lab_values | common_icu | 0.3922242224216461 | not_exported_by_KDD067 | pre-action/history when used as state | low only when lagged before the current action |
| paco2 | lab_values | common_icu | 0.3922242224216461 | not_exported_by_KDD067 | pre-action/history | low_if_lagged_to_pre_action_history |
| ph | lab_values | common_icu | 0.3922242224216461 | not_exported_by_KDD067 | pre-action/history | low_if_lagged_to_pre_action_history |
| base_excess | lab_values | common_icu | 0.3922242224216461 | not_exported_by_KDD067 | pre-action/history when used as state | low only when lagged before the current action |
| co2_bicarbonate | lab_values | common_icu_and_cohort_specific | 0.3922242224216461 | not_exported_by_KDD067 | pre-action/history | low_if_lagged_to_pre_action_history |
| pao2_fio2 | lab_values | common_icu | 0.3922242224216461 | not_exported_by_KDD067 | pre-action/history when used as state | low only when lagged before the current action |
| wbc | lab_values | common_icu | 0.3922242224216461 | not_exported_by_KDD067 | pre-action/history when used as state | low only when lagged before the current action |
| platelet | lab_values | common_icu | 0.3922242224216461 | not_exported_by_KDD067 | pre-action/history when used as state | low only when lagged before the current action |
| bun | lab_values | common_icu_and_cohort_specific | 0.3922242224216461 | not_exported_by_KDD067 | pre-action/history | low_if_lagged_to_pre_action_history |
| creatinine | lab_values | common_icu_and_cohort_specific | 0.3922242224216461 | not_exported_by_KDD067 | pre-action/history | low_if_lagged_to_pre_action_history |
| ptt | lab_values | common_icu | 0.3922242224216461 | not_exported_by_KDD067 | pre-action/history when used as state | low only when lagged before the current action |
| pt | lab_values | common_icu | 0.3922242224216461 | not_exported_by_KDD067 | pre-action/history when used as state | low only when lagged before the current action |
| inr | lab_values | common_icu | 0.3922242224216461 | not_exported_by_KDD067 | pre-action/history when used as state | low only when lagged before the current action |
| ast | lab_values | common_icu | 0.3922242224216461 | not_exported_by_KDD067 | pre-action/history when used as state | low only when lagged before the current action |
| alt | lab_values | common_icu | 0.3922242224216461 | not_exported_by_KDD067 | pre-action/history when used as state | low only when lagged before the current action |
| total_bilirubin | lab_values | common_icu | 0.3922242224216461 | not_exported_by_KDD067 | pre-action/history when used as state | low only when lagged before the current action |
| magnesium | lab_values | common_icu | 0.3922242224216461 | not_exported_by_KDD067 | pre-action/history | low_if_lagged_to_pre_action_history |
| ionized_calcium | lab_values | common_icu | 0.3922242224216461 | not_exported_by_KDD067 | pre-action/history when used as state | low only when lagged before the current action |
| calcium | lab_values | common_icu | 0.3922242224216461 | not_exported_by_KDD067 | pre-action/history when used as state | low only when lagged before the current action |
| urine_output | lab_values | common_icu_and_cohort_specific | 0.3922242224216461 | not_exported_by_KDD067 | pre-action/history | low_if_lagged_to_pre_action_history |
| mechanical_ventilation | lab_values | common_icu | 0.3922242224216461 | not_exported_by_KDD067 | pre-action/history when used as state | low only when lagged before the current action |
| step_id | other | common_icu | 0.3922242224216461 | not_exported_by_KDD067 | pre-action/history when used as state | low only when lagged before the current action |

## Action Contract

- Primary action view: `factorized_2_action_three_bin`.
- Action families: `diuretic;rrt_crrt`; K=`9`.
- Definition: Diuretic exposure burden;RRT/CRRT exposure burden using summed overlap fraction of structured medication intervals;structured procedure overlap burden.
- Train-only bin edges: `diuretic:[1];rrt_crrt:[1]`.
- Binning note: `empirical_action_count_matches_released_abstraction`.
- Required binary control: `factorized_2_action_binary` (Diuretic exposure burden;RRT/CRRT exposure burden).
- Timing: `current_4h_window_exposure_predicts_next_4h_state`; next-window action is excluded as leakage.
- Decision cadence: Repeated 4h diuretic exposure plus RRT/CRRT support status.

## Support Summary

| selected_action_count | mean_behavior_support | low_support_rate | temporal_density | episode_positive_rate | no_action_mass | top1_action_frequency | entropy_logk | supported_action_count_005 | ess_proxy | measurement_profile |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 9 | 0.29173929023471695 | 0.11233912600887079 | 0.25238129862575437 | 0.6823238566131026 | 0.7476187013742456 | 0.7476187013742456 | 0.7399100663224154 | 3 | 9299.045384556617 | development_cap5000_KDD067 |

Support and uncertainty evidence profile: `development_cap5000_KDD067`. These are historical aggregate diagnostics, not OPE. WIS, WPDIS, and FQE remain quarantined under `OPE_CONTRACT.md` because exact KDD078 estimator inputs and denominator provenance are unavailable; CWPDIS is not an accepted public metric.

## Outcome Components

| component | computable_definition | nonzero_density | sparsity | release_status |
| --- | --- | --- | --- | --- |
| terminal_mortality | 90d mortality indicator used as terminal outcome metric. | 0.014755836490474517 | 0.9852441635095255 | frozen_vector_outcome_component |
| intermediate_physiology | Creatinine trajectory, urine-output, potassium, and acid-base proxy. | 0.3704339878426658 | 0.6295660121573342 | frozen_vector_outcome_component |
| safety_burden | RRT, electrolyte, acid-base, and fluid-balance burden. | 0.5096235294117647 | 0.490376 | frozen_vector_outcome_component |
| combined_scalar_reward | terminal plus renal recovery proxy minus electrolyte/RRT burden. | 0.67826960677751 | 0.32173039322249 | diagnostic_components_only_no_scalar_policy_reward |

No scalar reward is released. Terminal, intermediate physiology, and safety/burden components must be reported separately.

## Readiness And Limits

- Tier reason: Full scale, clinically coherent renal support axes, measured same-family factorized gain, and runnable controls.
- Uncertainty warning: KDD067 intervals overcovered and Width90 must be interpreted with sparse-lab error.
- Known limitations: RRT initiation is often few-step; the RRT axis is support status, not a treatment recommendation target.
- Runnable baselines: `persistence_last_visible;fixed_ridge_residual_alpha_1`.
- Allowed tracks: `point_transition;uncertainty`.
- Cohort additions are frozen after KDD068 unless this task is invalidated.

## Claim Boundary

Benchmark construction, recorded-trajectory forecasting, uncertainty, and support diagnostics only. Treatment, causal, counterfactual, clinical-utility, optimization, and autonomous-decision claims remain blocked.
