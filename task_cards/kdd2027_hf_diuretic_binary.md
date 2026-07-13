# Heart failure / volume overload: diuretic__binary

- Benchmark version: `KDD2027-E060-4H-v1.0.0+KDD089`
- Task ID: `kdd2027_hf_diuretic_binary`
- Release tier: `primary`
- Dynamics evaluation: `runnable_public_evaluator`
- Policy evaluation: `quarantined_not_public_leaderboard`

## Cohort Contract

- Clinical anchor: First decongestion, diuretic, or volume-overload anchor
- Inclusion: Adult non-sepsis E060-eligible ICU stay with heart-failure or volume-overload evidence.
- Exclusion: Exclude E060 sepsis-reference stays and non-overlapping anchor windows.
- Episode unit: ICU stay; 72h [-24h,+48h] anchor window; 4h bins; 18 steps; window must overlap the ICU stay.

## Scale Profiles

| profile | train_episodes | validation_episodes | test_episodes | train_windows | validation_windows | test_windows |
| --- | --- | --- | --- | --- | --- | --- |
| full_eligible | 22827 | 4879 | 4846 | 410886 | 87822 | 87228 |
| development_cap5000 | 3562 | 730 | 708 | 64116 | 13140 | 12744 |

The `full_eligible` profile is the official benchmark. `development_cap5000` is a separate cost-controlled profile and cannot replace it.

## State And Missingness

- Feature view: common ICU features plus cohort-specific interpretation fields from `acute_decompensated_hf_volume_overload`.
- Full aggregate observation fraction: `0.3779990375041961`.
- Full aggregate missing fraction: `0.6220009624958038`.
- Missingness contract: observation mask retained; train-only imputation/scaling; no future-value fill
- Cohort-specific feature subset: mbp, spo2, bun, creatinine, urine_output
- Per-feature full-cohort observation fractions were not exported by KDD067 and remain a KDD069 reporting requirement.

| feature_name | feature_group | feature_role | full_overall_observation_fraction | per_feature_observation_fraction | timing_role | leakage_risk |
| --- | --- | --- | --- | --- | --- | --- |
| age | demographics | common_icu | 0.3779990375041961 | not_exported_by_KDD067 | pre-action/history | low_if_lagged_to_pre_action_history |
| gender_male | demographics | common_icu | 0.3779990375041961 | not_exported_by_KDD067 | pre-action/history | low_if_lagged_to_pre_action_history |
| weight | demographics | common_icu | 0.3779990375041961 | not_exported_by_KDD067 | pre-action/history | low_if_lagged_to_pre_action_history |
| readmission | demographics | common_icu | 0.3779990375041961 | not_exported_by_KDD067 | pre-action/history when used as state | low only when lagged before the current action |
| elixhauser_score_proxy | demographics | common_icu | 0.3779990375041961 | not_exported_by_KDD067 | pre-action/history when used as state | low only when lagged before the current action |
| heart_rate | vital_signs | common_icu | 0.3779990375041961 | not_exported_by_KDD067 | pre-action/history | low_if_lagged_to_pre_action_history |
| sbp | vital_signs | common_icu | 0.3779990375041961 | not_exported_by_KDD067 | pre-action/history | low_if_lagged_to_pre_action_history |
| mbp | vital_signs | common_icu_and_cohort_specific | 0.3779990375041961 | not_exported_by_KDD067 | pre-action/history | low_if_lagged_to_pre_action_history |
| dbp | vital_signs | common_icu | 0.3779990375041961 | not_exported_by_KDD067 | pre-action/history | low_if_lagged_to_pre_action_history |
| respiratory_rate | vital_signs | common_icu | 0.3779990375041961 | not_exported_by_KDD067 | pre-action/history | low_if_lagged_to_pre_action_history |
| temperature_c | vital_signs | common_icu | 0.3779990375041961 | not_exported_by_KDD067 | pre-action/history | low_if_lagged_to_pre_action_history |
| spo2 | vital_signs | common_icu_and_cohort_specific | 0.3779990375041961 | not_exported_by_KDD067 | pre-action/history | low_if_lagged_to_pre_action_history |
| shock_index | vital_signs | common_icu | 0.3779990375041961 | not_exported_by_KDD067 | pre-action/history when used as state | low only when lagged before the current action |
| sofa_proxy | vital_signs | common_icu | 0.3779990375041961 | not_exported_by_KDD067 | pre-action/history when used as state | low only when lagged before the current action |
| gcs_proxy | vital_signs | common_icu | 0.3779990375041961 | not_exported_by_KDD067 | pre-action/history | low_if_lagged_to_pre_action_history |
| fio2 | vital_signs | common_icu | 0.3779990375041961 | not_exported_by_KDD067 | pre-action/history when used as state | low only when lagged before the current action |
| sirs_proxy | vital_signs | common_icu | 0.3779990375041961 | not_exported_by_KDD067 | pre-action/history when used as state | low only when lagged before the current action |
| lactate | lab_values | common_icu | 0.3779990375041961 | not_exported_by_KDD067 | pre-action/history | low_if_lagged_to_pre_action_history |
| pao2 | lab_values | common_icu | 0.3779990375041961 | not_exported_by_KDD067 | pre-action/history when used as state | low only when lagged before the current action |
| paco2 | lab_values | common_icu | 0.3779990375041961 | not_exported_by_KDD067 | pre-action/history | low_if_lagged_to_pre_action_history |
| ph | lab_values | common_icu | 0.3779990375041961 | not_exported_by_KDD067 | pre-action/history | low_if_lagged_to_pre_action_history |
| base_excess | lab_values | common_icu | 0.3779990375041961 | not_exported_by_KDD067 | pre-action/history when used as state | low only when lagged before the current action |
| co2_bicarbonate | lab_values | common_icu | 0.3779990375041961 | not_exported_by_KDD067 | pre-action/history | low_if_lagged_to_pre_action_history |
| pao2_fio2 | lab_values | common_icu | 0.3779990375041961 | not_exported_by_KDD067 | pre-action/history when used as state | low only when lagged before the current action |
| wbc | lab_values | common_icu | 0.3779990375041961 | not_exported_by_KDD067 | pre-action/history when used as state | low only when lagged before the current action |
| platelet | lab_values | common_icu | 0.3779990375041961 | not_exported_by_KDD067 | pre-action/history when used as state | low only when lagged before the current action |
| bun | lab_values | common_icu_and_cohort_specific | 0.3779990375041961 | not_exported_by_KDD067 | pre-action/history | low_if_lagged_to_pre_action_history |
| creatinine | lab_values | common_icu_and_cohort_specific | 0.3779990375041961 | not_exported_by_KDD067 | pre-action/history | low_if_lagged_to_pre_action_history |
| ptt | lab_values | common_icu | 0.3779990375041961 | not_exported_by_KDD067 | pre-action/history when used as state | low only when lagged before the current action |
| pt | lab_values | common_icu | 0.3779990375041961 | not_exported_by_KDD067 | pre-action/history when used as state | low only when lagged before the current action |
| inr | lab_values | common_icu | 0.3779990375041961 | not_exported_by_KDD067 | pre-action/history when used as state | low only when lagged before the current action |
| ast | lab_values | common_icu | 0.3779990375041961 | not_exported_by_KDD067 | pre-action/history when used as state | low only when lagged before the current action |
| alt | lab_values | common_icu | 0.3779990375041961 | not_exported_by_KDD067 | pre-action/history when used as state | low only when lagged before the current action |
| total_bilirubin | lab_values | common_icu | 0.3779990375041961 | not_exported_by_KDD067 | pre-action/history when used as state | low only when lagged before the current action |
| magnesium | lab_values | common_icu | 0.3779990375041961 | not_exported_by_KDD067 | pre-action/history | low_if_lagged_to_pre_action_history |
| ionized_calcium | lab_values | common_icu | 0.3779990375041961 | not_exported_by_KDD067 | pre-action/history when used as state | low only when lagged before the current action |
| calcium | lab_values | common_icu | 0.3779990375041961 | not_exported_by_KDD067 | pre-action/history when used as state | low only when lagged before the current action |
| urine_output | lab_values | common_icu_and_cohort_specific | 0.3779990375041961 | not_exported_by_KDD067 | pre-action/history | low_if_lagged_to_pre_action_history |
| mechanical_ventilation | lab_values | common_icu | 0.3779990375041961 | not_exported_by_KDD067 | pre-action/history when used as state | low only when lagged before the current action |
| step_id | other | common_icu | 0.3779990375041961 | not_exported_by_KDD067 | pre-action/history when used as state | low only when lagged before the current action |

## Action Contract

- Primary action view: `diuretic__binary`.
- Action families: `diuretic`; K=`2`.
- Definition: Diuretic exposure burden using summed overlap fraction of structured medication intervals.
- Train-only bin edges: `none_binary_exposure`.
- Binning note: `empirical_action_count_matches_released_abstraction`.
- Required binary control: `diuretic__binary` (Diuretic exposure burden).
- Timing: `current_4h_window_exposure_predicts_next_4h_state`; next-window action is excluded as leakage.
- Decision cadence: Repeated 4h diuretic exposure.

## Support Summary

| selected_action_count | mean_behavior_support | low_support_rate | temporal_density | episode_positive_rate | no_action_mass | top1_action_frequency | entropy_logk | supported_action_count_005 | ess_proxy | measurement_profile |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 0.6161566259141559 | 0.0019940179461615153 | 0.2209205716184779 | 0.6440677966101694 | 0.7790794283815221 | 0.7790794283815221 | 0.8302014231232138 | 2 | 10865.996641432477 | development_cap5000_KDD067 |

Support and uncertainty evidence profile: `development_cap5000_KDD067`. These are historical aggregate diagnostics, not OPE. WIS, WPDIS, and FQE remain quarantined under `OPE_CONTRACT.md` because exact KDD078 estimator inputs and denominator provenance are unavailable; CWPDIS is not an accepted public metric.

## Outcome Components

| component | computable_definition | nonzero_density | sparsity | release_status |
| --- | --- | --- | --- | --- |
| terminal_mortality | 90d mortality indicator used as terminal outcome metric. | 0.008637462111902972 | 0.991362537888097 | frozen_vector_outcome_component |
| intermediate_physiology | Urine output/net fluid balance/oxygenation decongestion proxy. | 0.3569990909761853 | 0.6430009090238147 | frozen_vector_outcome_component |
| safety_burden | Creatinine/BUN/electrolyte safety plus diuretic burden. | 0.5602470588235294 | 0.439753 | frozen_vector_outcome_component |
| combined_scalar_reward | terminal plus decongestion proxy minus renal/electrolyte and treatment-burden proxy. | 0.6998403670513726 | 0.30015963294862735 | diagnostic_components_only_no_scalar_policy_reward |

No scalar reward is released. Terminal, intermediate physiology, and safety/burden components must be reported separately.

## Readiness And Limits

- Tier reason: Large full scale and strong-support binary control; richer KDD067 views did not pass the frozen gate.
- Uncertainty warning: KDD067 intervals overcovered; renal and electrolyte sharpness warnings remain required.
- Known limitations: Net fluid balance is incompletely observed and diuretic exposure is confounded by congestion severity.
- Runnable baselines: `persistence_last_visible;fixed_ridge_residual_alpha_1`.
- Allowed tracks: `point_transition;uncertainty`.
- Cohort additions are frozen after KDD068 unless this task is invalidated.

## Claim Boundary

Benchmark construction, recorded-trajectory forecasting, uncertainty, and support diagnostics only. Treatment, causal, counterfactual, clinical-utility, optimization, and autonomous-decision claims remain blocked.
