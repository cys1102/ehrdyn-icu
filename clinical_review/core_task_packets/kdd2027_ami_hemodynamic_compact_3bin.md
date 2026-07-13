# AMI hemodynamic support: compact_joint_2_action

- Benchmark version: `KDD2027-E060-4H-v1.0.0+KDD089`
- Task ID: `kdd2027_ami_hemodynamic_compact_3bin`
- Release tier: `extended`
- Dynamics evaluation: `runnable_public_evaluator`
- Policy evaluation: `quarantined_not_public_leaderboard`

## Cohort Contract

- Clinical anchor: AMI diagnosis with first hemodynamic-support exposure when present
- Inclusion: Adult non-sepsis E060-eligible ICU stay with AMI diagnosis evidence.
- Exclusion: Exclude E060 sepsis-reference stays and non-overlapping anchor windows.
- Episode unit: ICU stay; 72h [-24h,+48h] anchor window; 4h bins; 18 steps; window must overlap the ICU stay.

## Scale Profiles

| profile | train_episodes | validation_episodes | test_episodes | train_windows | validation_windows | test_windows |
| --- | --- | --- | --- | --- | --- | --- |
| full_eligible | 3553 | 792 | 710 | 63954 | 14256 | 12780 |
| development_cap5000 | 3513 | 783 | 704 | 63234 | 14094 | 12672 |

The `full_eligible` profile is the official benchmark. `development_cap5000` is a separate cost-controlled profile and cannot replace it.

## State And Missingness

- Feature view: common ICU features plus cohort-specific interpretation fields from `cardiogenic_shock`.
- Full aggregate observation fraction: `0.3861553370952606`.
- Full aggregate missing fraction: `0.6138446629047394`.
- Missingness contract: observation mask retained; train-only imputation/scaling; no future-value fill
- Cohort-specific feature subset: heart_rate, mbp, lactate, creatinine, urine_output
- Per-feature full-cohort observation fractions were not exported by KDD067 and remain a KDD069 reporting requirement.

| feature_name | feature_group | feature_role | full_overall_observation_fraction | per_feature_observation_fraction | timing_role | leakage_risk |
| --- | --- | --- | --- | --- | --- | --- |
| age | demographics | common_icu | 0.3861553370952606 | not_exported_by_KDD067 | pre-action/history | low_if_lagged_to_pre_action_history |
| gender_male | demographics | common_icu | 0.3861553370952606 | not_exported_by_KDD067 | pre-action/history | low_if_lagged_to_pre_action_history |
| weight | demographics | common_icu | 0.3861553370952606 | not_exported_by_KDD067 | pre-action/history | low_if_lagged_to_pre_action_history |
| readmission | demographics | common_icu | 0.3861553370952606 | not_exported_by_KDD067 | pre-action/history when used as state | low only when lagged before the current action |
| elixhauser_score_proxy | demographics | common_icu | 0.3861553370952606 | not_exported_by_KDD067 | pre-action/history when used as state | low only when lagged before the current action |
| heart_rate | vital_signs | common_icu_and_cohort_specific | 0.3861553370952606 | not_exported_by_KDD067 | pre-action/history | low_if_lagged_to_pre_action_history |
| sbp | vital_signs | common_icu | 0.3861553370952606 | not_exported_by_KDD067 | pre-action/history | low_if_lagged_to_pre_action_history |
| mbp | vital_signs | common_icu_and_cohort_specific | 0.3861553370952606 | not_exported_by_KDD067 | pre-action/history | low_if_lagged_to_pre_action_history |
| dbp | vital_signs | common_icu | 0.3861553370952606 | not_exported_by_KDD067 | pre-action/history | low_if_lagged_to_pre_action_history |
| respiratory_rate | vital_signs | common_icu | 0.3861553370952606 | not_exported_by_KDD067 | pre-action/history | low_if_lagged_to_pre_action_history |
| temperature_c | vital_signs | common_icu | 0.3861553370952606 | not_exported_by_KDD067 | pre-action/history | low_if_lagged_to_pre_action_history |
| spo2 | vital_signs | common_icu | 0.3861553370952606 | not_exported_by_KDD067 | pre-action/history | low_if_lagged_to_pre_action_history |
| shock_index | vital_signs | common_icu | 0.3861553370952606 | not_exported_by_KDD067 | pre-action/history when used as state | low only when lagged before the current action |
| sofa_proxy | vital_signs | common_icu | 0.3861553370952606 | not_exported_by_KDD067 | pre-action/history when used as state | low only when lagged before the current action |
| gcs_proxy | vital_signs | common_icu | 0.3861553370952606 | not_exported_by_KDD067 | pre-action/history | low_if_lagged_to_pre_action_history |
| fio2 | vital_signs | common_icu | 0.3861553370952606 | not_exported_by_KDD067 | pre-action/history when used as state | low only when lagged before the current action |
| sirs_proxy | vital_signs | common_icu | 0.3861553370952606 | not_exported_by_KDD067 | pre-action/history when used as state | low only when lagged before the current action |
| lactate | lab_values | common_icu_and_cohort_specific | 0.3861553370952606 | not_exported_by_KDD067 | pre-action/history | low_if_lagged_to_pre_action_history |
| pao2 | lab_values | common_icu | 0.3861553370952606 | not_exported_by_KDD067 | pre-action/history when used as state | low only when lagged before the current action |
| paco2 | lab_values | common_icu | 0.3861553370952606 | not_exported_by_KDD067 | pre-action/history | low_if_lagged_to_pre_action_history |
| ph | lab_values | common_icu | 0.3861553370952606 | not_exported_by_KDD067 | pre-action/history | low_if_lagged_to_pre_action_history |
| base_excess | lab_values | common_icu | 0.3861553370952606 | not_exported_by_KDD067 | pre-action/history when used as state | low only when lagged before the current action |
| co2_bicarbonate | lab_values | common_icu | 0.3861553370952606 | not_exported_by_KDD067 | pre-action/history | low_if_lagged_to_pre_action_history |
| pao2_fio2 | lab_values | common_icu | 0.3861553370952606 | not_exported_by_KDD067 | pre-action/history when used as state | low only when lagged before the current action |
| wbc | lab_values | common_icu | 0.3861553370952606 | not_exported_by_KDD067 | pre-action/history when used as state | low only when lagged before the current action |
| platelet | lab_values | common_icu | 0.3861553370952606 | not_exported_by_KDD067 | pre-action/history when used as state | low only when lagged before the current action |
| bun | lab_values | common_icu | 0.3861553370952606 | not_exported_by_KDD067 | pre-action/history | low_if_lagged_to_pre_action_history |
| creatinine | lab_values | common_icu_and_cohort_specific | 0.3861553370952606 | not_exported_by_KDD067 | pre-action/history | low_if_lagged_to_pre_action_history |
| ptt | lab_values | common_icu | 0.3861553370952606 | not_exported_by_KDD067 | pre-action/history when used as state | low only when lagged before the current action |
| pt | lab_values | common_icu | 0.3861553370952606 | not_exported_by_KDD067 | pre-action/history when used as state | low only when lagged before the current action |
| inr | lab_values | common_icu | 0.3861553370952606 | not_exported_by_KDD067 | pre-action/history when used as state | low only when lagged before the current action |
| ast | lab_values | common_icu | 0.3861553370952606 | not_exported_by_KDD067 | pre-action/history when used as state | low only when lagged before the current action |
| alt | lab_values | common_icu | 0.3861553370952606 | not_exported_by_KDD067 | pre-action/history when used as state | low only when lagged before the current action |
| total_bilirubin | lab_values | common_icu | 0.3861553370952606 | not_exported_by_KDD067 | pre-action/history when used as state | low only when lagged before the current action |
| magnesium | lab_values | common_icu | 0.3861553370952606 | not_exported_by_KDD067 | pre-action/history | low_if_lagged_to_pre_action_history |
| ionized_calcium | lab_values | common_icu | 0.3861553370952606 | not_exported_by_KDD067 | pre-action/history when used as state | low only when lagged before the current action |
| calcium | lab_values | common_icu | 0.3861553370952606 | not_exported_by_KDD067 | pre-action/history when used as state | low only when lagged before the current action |
| urine_output | lab_values | common_icu_and_cohort_specific | 0.3861553370952606 | not_exported_by_KDD067 | pre-action/history | low_if_lagged_to_pre_action_history |
| mechanical_ventilation | lab_values | common_icu | 0.3861553370952606 | not_exported_by_KDD067 | pre-action/history when used as state | low only when lagged before the current action |
| step_id | other | common_icu | 0.3861553370952606 | not_exported_by_KDD067 | pre-action/history when used as state | low only when lagged before the current action |

## Action Contract

- Primary action view: `compact_joint_2_action`.
- Action families: `vasopressor;inotrope`; K=`9`.
- Definition: Vasopressor intensity;Inotrope exposure burden using maximum recorded rate or amount proxy per 4h bin;summed overlap fraction of structured medication intervals.
- Train-only bin edges: `vasopressor:[0.699833810329];inotrope:[1.67291665077]`.
- Binning note: `empirical_action_count_matches_released_abstraction`.
- Required binary control: `factorized_2_action_binary` (Vasopressor intensity;Inotrope exposure burden).
- Timing: `current_4h_window_exposure_predicts_next_4h_state`; next-window action is excluded as leakage.
- Decision cadence: Repeated vasopressor/inotrope exposure around AMI support windows.

## Support Summary

| selected_action_count | mean_behavior_support | low_support_rate | temporal_density | episode_positive_rate | no_action_mass | top1_action_frequency | entropy_logk | supported_action_count_005 | ess_proxy | measurement_profile |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 9 | 0.46540569238406765 | 0.09792780748663102 | 0.19852941176470587 | 0.5241477272727273 | 0.8014705882352942 | 0.8014705882352942 | 0.568924169300211 | 2 | 8164.239028241235 | development_cap5000_KDD067 |

Support and uncertainty evidence profile: `development_cap5000_KDD067`. These are historical aggregate diagnostics, not OPE. WIS, WPDIS, and FQE remain quarantined under `OPE_CONTRACT.md` because exact KDD078 estimator inputs and denominator provenance are unavailable; CWPDIS is not an accepted public metric.

## Outcome Components

| component | computable_definition | nonzero_density | sparsity | release_status |
| --- | --- | --- | --- | --- |
| terminal_mortality | 90d mortality indicator used as terminal outcome metric. | 0.00844245538427535 | 0.9915575446157247 | frozen_vector_outcome_component |
| intermediate_physiology | MAP/lactate/urine-output perfusion proxy with inotrope support context. | 0.3617457399765651 | 0.6382542600234349 | frozen_vector_outcome_component |
| safety_burden | Inotrope, vasodilator, MCS, anticoagulation, and procedure burden proxies. | 0.6894823529411764 | 0.310518 | frozen_vector_outcome_component |
| combined_scalar_reward | terminal plus perfusion proxy minus cardiac support burden. | 0.7792423358889716 | 0.22075766411102837 | diagnostic_components_only_no_scalar_policy_reward |

No scalar reward is released. Terminal, intermediate physiology, and safety/burden components must be reported separately.

## Readiness And Limits

- Tier reason: Measured same-family compact gain, but full scale is near the minimum and uncertainty is very wide.
- Uncertainty warning: KDD067 Width90 is extreme; this task cannot enter the primary uncertainty leaderboard.
- Known limitations: AMI support mixes heterogeneous hemodynamic states and procedure timing.
- Runnable baselines: `persistence_last_visible;fixed_ridge_residual_alpha_1`.
- Allowed tracks: `point_transition;uncertainty`.
- Cohort additions are frozen after KDD068 unless this task is invalidated.

## Claim Boundary

Benchmark construction, recorded-trajectory forecasting, uncertainty, and support diagnostics only. Treatment, causal, counterfactual, clinical-utility, optimization, and autonomous-decision claims remain blocked.
