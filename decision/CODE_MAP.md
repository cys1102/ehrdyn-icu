# Paper-to-code map

| Paper component | Public code | Status |
| --- | --- | --- |
| AI-Clinician-aligned K25 sepsis materialization | `decision/reference_code/run_kdd_s02_canonical_sepsis_materialization.py` | portable path-sanitized executed construction snapshot; legacy filename retained |
| P/R/T-component training and evaluation | `decision/reference_code/run_kdd098r_world_models.py` and `kdd098r_world_model.json` | portable path-sanitized runner plus byte-identical configuration snapshot |
| Model-free diagnostic training | `decision/reference_code/run_kdd101_model_free_diagnostics.py` and `kdd101_model_free_diagnostics_v5.json` | portable path-sanitized runner plus byte-identical configuration snapshot |
| GRU-D transition component | `decision/reference_code/kdd069_sequence_models.py` | byte-identical architecture snapshot |
| Causal Transformer transition component | `decision/reference_code/kdd069_sequence_models.py` | byte-identical architecture snapshot |
| Gaussian transition head | `decision/reference_code/kdd069_model_types.py` | byte-identical architecture snapshot |
| Categorical RSSM | `decision/reference_code/kdd069_rssm_models.py` | byte-identical architecture snapshot |
| Exact finite known-value evaluator and categorical CEM | `decision/reference_code/kdd_e01_evaluator.py` | byte-identical evaluator snapshot |
| Respiratory/shock known-value policy benchmark | `decision/reference_code/run_kdd_x02_cross_cohort_policy_benchmark.py` | byte-identical executed runner snapshot |
| AKI/HF task-matched evaluator | `decision/reference_code/run_kdd_x08_task_matched_evaluator.py` | byte-identical executed runner snapshot |
| AKI/HF policy benchmark | `decision/reference_code/run_kdd_x09_promoted_cohort_policy_benchmark.py` | byte-identical executed runner snapshot |
| Full OPE grid and gates | `decision/reference_code/run_kdd_e02_known_value_full.py` | byte-identical executed evaluator snapshot |
| Adaptive exact-finite policy benchmark | `decision/reference_code/run_kdd_adapt01_adaptive_known_value.py` and `kdd_adapt01_adaptive_known_value_v1.json` | byte-identical executed runner and configuration snapshots |
| Heterogeneous exact-finite policy benchmark | `decision/reference_code/run_kdd107_heterogeneous_known_value.py` and `kdd107_heterogeneous_known_value_v1.json` | portable path-sanitized executed runner and byte-identical configuration snapshot |
| Repeated-dataset OPE calibration | `decision/reference_code/run_kdd_ope_rd01_repeated_dataset.py` and `kdd_ope_rd01_repeated_dataset_v1.json` | byte-identical executed runner and configuration snapshots |
| EHR-to-known-value diagnostic bridge | `decision/reference_code/run_kdd_bridge01_ehr_known_value.py` and `kdd_bridge01_ehr_known_value_v1.json` | byte-identical executed runner and configuration snapshots |
| Shared known-value utilities | `decision/reference_code/run_kdd100_complete_known_value.py` and `run_kdd100r_task_matched_known_value.py` | byte-identical dependency snapshots |
| Frozen executed configurations | `decision/reference_code/*.json` | byte-identical configuration snapshots |
| Portable planner/CRN smoke | `src/kdd2027_benchmark/decision/known_value.py` | executable clean-room implementation |
| IS/WIS/PDIS/WPDIS formulas | `src/kdd2027_benchmark/decision/ope.py` | executable clean-room implementation |
| Evidence and contract validation | `src/kdd2027_benchmark/decision/contract.py` | executable release validator |
| Cross-layer baseline inventory and point-leader atlas | `decision/evidence/baseline_method_inventory.csv` and `baseline_surface_atlas.csv` | complete aggregate method identities and layer-specific outcomes |
| Cohort-level transition and uncertainty leaders | `decision/evidence/all_baseline_transition_leaders.csv` and `cohort_uncertainty_leaders.csv` | source-derived six-cohort summaries |
| All transition methods on all cohorts | `decision/evidence/complete_model_performance_by_cohort.csv` | complete 36-row reusable result matrix, including non-leading methods |
| All adaptive policy labels | `decision/evidence/adaptive_policy_performance_all_methods.csv`, `adaptive_policy_true_returns_all_rows.csv`, and `adaptive_policy_regret_all_rows.csv` | all task-by-method summaries and all 700 seed rows |
| Adaptive model exploitation | `decision/evidence/adaptive_exploitation_gap_all_rows.csv` | all 480 planner-seed rows |
| All heterogeneous policy labels | `decision/evidence/heterogeneous_policy_true_returns_all_rows.csv` and `heterogeneous_all_method_seed_means.csv` | all 4,080 policy-seed rows and complete 34-label summaries |
| Heterogeneous world-model planning | `decision/evidence/heterogeneous_world_model_planner_all_rows.csv` and `heterogeneous_model_exploitation_all_rows.csv` | all 2,880 planner-seed and exploitation rows |
| Repeated-dataset OPE | `decision/evidence/repeated_dataset_ope_coverage.csv` and `repeated_dataset_ope_authorization.csv` | all 10,368 coverage rows and 1,728 frozen contract dispositions |
| EHR-to-known-value bridge | `decision/evidence/cross_surface_model_family_rows.csv` and `ehr_known_value_bridge_coefficients.csv` | complete model-family rows and descriptive bridge coefficients |
| Representative prior-study landscape | `decision/evidence/related_work_landscape.csv` | source-verified motivation table; not a systematic review |
| Complete paper result rows | `decision/evidence/` | hash-verified aggregate evidence |

The original-source and packaged hashes, plus any path-only sanitization label,
are recorded in `decision/reference_code/source_manifest.csv`. Full MIMIC-IV construction and
training require credentialed local inputs and are not run by the unrestricted
smoke command. The smoke path validates executable invariants; it is not a
substitute for numerical paper reproduction.
