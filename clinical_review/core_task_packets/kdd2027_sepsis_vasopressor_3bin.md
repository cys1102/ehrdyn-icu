# Sepsis reference: vasopressor__three_bin

- Benchmark version: `KDD2027-E060-4H-v1.0.0+KDD089`
- Task ID: `kdd2027_sepsis_vasopressor_3bin`
- Release tier: `primary`
- Dynamics evaluation: `runnable_public_evaluator`
- Policy evaluation: `quarantined_not_public_leaderboard`

## Cohort Contract

- Clinical anchor: E060 suspected-infection anchor with SOFA >=2
- Inclusion: Adult ICU stay >=24h with computable 90d mortality and an anchor window overlapping the stay.
- Exclusion: Exclude stays failing the E060 suspected-infection plus SOFA >=2 reference phenotype.
- Episode unit: ICU stay; 72h [-24h,+48h] anchor window; 4h bins; 18 steps; window must overlap the ICU stay.

## Scale Profiles

| profile | train_episodes | validation_episodes | test_episodes | train_windows | validation_windows | test_windows |
| --- | --- | --- | --- | --- | --- | --- |
| full_eligible | 19228 | 3920 | 4088 | 346104 | 70560 | 73584 |
| development_cap5000 | 3579 | 698 | 723 | 64422 | 12564 | 13014 |

The `full_eligible` profile is the official benchmark. `development_cap5000` is a separate cost-controlled profile and cannot replace it.

## State And Missingness

- Feature view: common ICU features plus cohort-specific interpretation fields from `sepsis_reference`.
- Full aggregate observation fraction: `0.390602256825`.
- Full aggregate missing fraction: `0.609397743175`.
- Missingness contract: observation mask retained; train-only imputation/scaling; no future-value fill
- Cohort-specific feature subset: mbp, sofa_proxy, lactate, creatinine, urine_output
- Per-feature full-cohort observation fractions were not exported by KDD067 and remain a KDD069 reporting requirement.

| feature_name | feature_group | feature_role | full_overall_observation_fraction | per_feature_observation_fraction | timing_role | leakage_risk |
| --- | --- | --- | --- | --- | --- | --- |
| age | demographics | common_icu | 0.390602256825 | not_exported_by_KDD067 | pre-action/history | low_if_lagged_to_pre_action_history |
| gender_male | demographics | common_icu | 0.390602256825 | not_exported_by_KDD067 | pre-action/history | low_if_lagged_to_pre_action_history |
| weight | demographics | common_icu | 0.390602256825 | not_exported_by_KDD067 | pre-action/history | low_if_lagged_to_pre_action_history |
| readmission | demographics | common_icu | 0.390602256825 | not_exported_by_KDD067 | pre-action/history when used as state | low only when lagged before the current action |
| elixhauser_score_proxy | demographics | common_icu | 0.390602256825 | not_exported_by_KDD067 | pre-action/history when used as state | low only when lagged before the current action |
| heart_rate | vital_signs | common_icu | 0.390602256825 | not_exported_by_KDD067 | pre-action/history | low_if_lagged_to_pre_action_history |
| sbp | vital_signs | common_icu | 0.390602256825 | not_exported_by_KDD067 | pre-action/history | low_if_lagged_to_pre_action_history |
| mbp | vital_signs | common_icu_and_cohort_specific | 0.390602256825 | not_exported_by_KDD067 | pre-action/history | low_if_lagged_to_pre_action_history |
| dbp | vital_signs | common_icu | 0.390602256825 | not_exported_by_KDD067 | pre-action/history | low_if_lagged_to_pre_action_history |
| respiratory_rate | vital_signs | common_icu | 0.390602256825 | not_exported_by_KDD067 | pre-action/history | low_if_lagged_to_pre_action_history |
| temperature_c | vital_signs | common_icu | 0.390602256825 | not_exported_by_KDD067 | pre-action/history | low_if_lagged_to_pre_action_history |
| spo2 | vital_signs | common_icu | 0.390602256825 | not_exported_by_KDD067 | pre-action/history | low_if_lagged_to_pre_action_history |
| shock_index | vital_signs | common_icu | 0.390602256825 | not_exported_by_KDD067 | pre-action/history when used as state | low only when lagged before the current action |
| sofa_proxy | vital_signs | common_icu_and_cohort_specific | 0.390602256825 | not_exported_by_KDD067 | pre-action/history | low_if_lagged_to_pre_action_history |
| gcs_proxy | vital_signs | common_icu | 0.390602256825 | not_exported_by_KDD067 | pre-action/history | low_if_lagged_to_pre_action_history |
| fio2 | vital_signs | common_icu | 0.390602256825 | not_exported_by_KDD067 | pre-action/history when used as state | low only when lagged before the current action |
| sirs_proxy | vital_signs | common_icu | 0.390602256825 | not_exported_by_KDD067 | pre-action/history when used as state | low only when lagged before the current action |
| lactate | lab_values | common_icu_and_cohort_specific | 0.390602256825 | not_exported_by_KDD067 | pre-action/history | low_if_lagged_to_pre_action_history |
| pao2 | lab_values | common_icu | 0.390602256825 | not_exported_by_KDD067 | pre-action/history when used as state | low only when lagged before the current action |
| paco2 | lab_values | common_icu | 0.390602256825 | not_exported_by_KDD067 | pre-action/history | low_if_lagged_to_pre_action_history |
| ph | lab_values | common_icu | 0.390602256825 | not_exported_by_KDD067 | pre-action/history | low_if_lagged_to_pre_action_history |
| base_excess | lab_values | common_icu | 0.390602256825 | not_exported_by_KDD067 | pre-action/history when used as state | low only when lagged before the current action |
| co2_bicarbonate | lab_values | common_icu | 0.390602256825 | not_exported_by_KDD067 | pre-action/history | low_if_lagged_to_pre_action_history |
| pao2_fio2 | lab_values | common_icu | 0.390602256825 | not_exported_by_KDD067 | pre-action/history when used as state | low only when lagged before the current action |
| wbc | lab_values | common_icu | 0.390602256825 | not_exported_by_KDD067 | pre-action/history when used as state | low only when lagged before the current action |
| platelet | lab_values | common_icu | 0.390602256825 | not_exported_by_KDD067 | pre-action/history when used as state | low only when lagged before the current action |
| bun | lab_values | common_icu | 0.390602256825 | not_exported_by_KDD067 | pre-action/history | low_if_lagged_to_pre_action_history |
| creatinine | lab_values | common_icu_and_cohort_specific | 0.390602256825 | not_exported_by_KDD067 | pre-action/history | low_if_lagged_to_pre_action_history |
| ptt | lab_values | common_icu | 0.390602256825 | not_exported_by_KDD067 | pre-action/history when used as state | low only when lagged before the current action |
| pt | lab_values | common_icu | 0.390602256825 | not_exported_by_KDD067 | pre-action/history when used as state | low only when lagged before the current action |
| inr | lab_values | common_icu | 0.390602256825 | not_exported_by_KDD067 | pre-action/history when used as state | low only when lagged before the current action |
| ast | lab_values | common_icu | 0.390602256825 | not_exported_by_KDD067 | pre-action/history when used as state | low only when lagged before the current action |
| alt | lab_values | common_icu | 0.390602256825 | not_exported_by_KDD067 | pre-action/history when used as state | low only when lagged before the current action |
| total_bilirubin | lab_values | common_icu | 0.390602256825 | not_exported_by_KDD067 | pre-action/history when used as state | low only when lagged before the current action |
| magnesium | lab_values | common_icu | 0.390602256825 | not_exported_by_KDD067 | pre-action/history | low_if_lagged_to_pre_action_history |
| ionized_calcium | lab_values | common_icu | 0.390602256825 | not_exported_by_KDD067 | pre-action/history when used as state | low only when lagged before the current action |
| calcium | lab_values | common_icu | 0.390602256825 | not_exported_by_KDD067 | pre-action/history when used as state | low only when lagged before the current action |
| urine_output | lab_values | common_icu_and_cohort_specific | 0.390602256825 | not_exported_by_KDD067 | pre-action/history | low_if_lagged_to_pre_action_history |
| mechanical_ventilation | lab_values | common_icu | 0.390602256825 | not_exported_by_KDD067 | pre-action/history when used as state | low only when lagged before the current action |
| step_id | other | common_icu | 0.390602256825 | not_exported_by_KDD067 | pre-action/history when used as state | low only when lagged before the current action |

## Action Contract

- Primary action view: `vasopressor__three_bin`.
- Action families: `vasopressor`; K=`3`.
- Definition: Vasopressor intensity using maximum recorded rate or amount proxy per 4h bin.
- Train-only bin edges: `0.495270386338`.
- Binning note: `empirical_action_count_matches_released_abstraction`.
- Required binary control: `vasopressor__binary` (Vasopressor intensity).
- Timing: `current_4h_window_exposure_predicts_next_4h_state`; next-window action is excluded as leakage.
- Decision cadence: Repeated 4h vasopressor-intensity exposure.

## Support Summary

| selected_action_count | mean_behavior_support | low_support_rate | temporal_density | episode_positive_rate | no_action_mass | top1_action_frequency | entropy_logk | supported_action_count_005 | ess_proxy | measurement_profile |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | 0.6850940253272824 | 0.032462777642177205 | 0.17581970547555123 | 0.4107883817427386 | 0.8241802945244487 | 0.8241802945244487 | 0.549092543387285 | 3 | 10570.345818358368 | development_cap5000_KDD067 |

Support and uncertainty evidence profile: `development_cap5000_KDD067`. These are historical aggregate diagnostics, not OPE. WIS, WPDIS, and FQE remain quarantined under `OPE_CONTRACT.md` because exact KDD078 estimator inputs and denominator provenance are unavailable; CWPDIS is not an accepted public metric.

## Outcome Components

| component | computable_definition | nonzero_density | sparsity | release_status |
| --- | --- | --- | --- | --- |
| terminal_mortality | 90d mortality indicator used as terminal outcome metric. | 0.018213230854587925 | 0.9817867691454121 | frozen_vector_outcome_component |
| intermediate_physiology | SOFA/lactate/MAP stability proxy with lagged vasoactive burden. | 0.3689021314458334 | 0.6310978685541666 | frozen_vector_outcome_component |
| safety_burden | Vasopressor, ventilation, RRT, and support-burden proxy. | 0.7000117647058823 | 0.299988 | frozen_vector_outcome_component |
| combined_scalar_reward | terminal plus SOFA/lactate/MAP physiology minus lagged support burden. | 0.7900301301486186 | 0.20996986985138144 | diagnostic_components_only_no_scalar_policy_reward |

No scalar reward is released. Terminal, intermediate physiology, and safety/burden components must be reported separately.

## Readiness And Limits

- Tier reason: Full scale, coherent repeated support action, measured KDD067 promotion, and runnable persistence/ridge controls.
- Uncertainty warning: KDD067 intervals overcovered; calibration and sharpness must be reported together.
- Known limitations: High severity-action confounding; action-control results are forecasting diagnostics only.
- Runnable baselines: `persistence_last_visible;fixed_ridge_residual_alpha_1`.
- Allowed tracks: `point_transition;uncertainty`.
- Cohort additions are frozen after KDD068 unless this task is invalidated.

## Claim Boundary

Benchmark construction, recorded-trajectory forecasting, uncertainty, and support diagnostics only. Treatment, causal, counterfactual, clinical-utility, optimization, and autonomous-decision claims remain blocked.
