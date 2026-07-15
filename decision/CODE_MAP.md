# Paper-to-code map

| Paper component | Public code | Status |
| --- | --- | --- |
| GRU-D transition model | `decision/reference_code/kdd069_sequence_models.py` | byte-identical architecture snapshot |
| Causal Transformer transition model | `decision/reference_code/kdd069_sequence_models.py` | byte-identical architecture snapshot |
| Gaussian transition head | `decision/reference_code/kdd069_model_types.py` | byte-identical architecture snapshot |
| Categorical RSSM | `decision/reference_code/kdd069_rssm_models.py` | byte-identical architecture snapshot |
| Exact finite known-value evaluator and categorical CEM | `decision/reference_code/kdd_e01_evaluator.py` | byte-identical evaluator snapshot |
| Respiratory/shock known-value policy benchmark | `decision/reference_code/run_kdd_x02_cross_cohort_policy_benchmark.py` | byte-identical executed runner snapshot |
| AKI/HF task-matched evaluator | `decision/reference_code/run_kdd_x08_task_matched_evaluator.py` | byte-identical executed runner snapshot |
| AKI/HF policy benchmark | `decision/reference_code/run_kdd_x09_promoted_cohort_policy_benchmark.py` | byte-identical executed runner snapshot |
| Full OPE grid and gates | `decision/reference_code/run_kdd_e02_known_value_full.py` | byte-identical executed evaluator snapshot |
| Shared known-value utilities | `decision/reference_code/run_kdd100_complete_known_value.py` and `run_kdd100r_task_matched_known_value.py` | byte-identical dependency snapshots |
| Frozen executed configurations | `decision/reference_code/*.json` | byte-identical configuration snapshots |
| Portable planner/CRN smoke | `src/kdd2027_benchmark/decision/known_value.py` | executable clean-room implementation |
| IS/WIS/PDIS/WPDIS formulas | `src/kdd2027_benchmark/decision/ope.py` | executable clean-room implementation |
| Evidence and contract validation | `src/kdd2027_benchmark/decision/contract.py` | executable release validator |
| Cross-layer baseline inventory and point-leader atlas | `decision/evidence/baseline_method_inventory.csv` and `baseline_surface_atlas.csv` | complete aggregate method identities and layer-specific outcomes |
| Cohort-level transition and uncertainty leaders | `decision/evidence/all_baseline_transition_leaders.csv` and `cohort_uncertainty_leaders.csv` | source-derived six-cohort summaries |
| Complete paper result rows | `decision/evidence/` | hash-verified aggregate evidence |

The byte-identical source hashes are recorded in
`decision/reference_code/source_manifest.csv`. Full MIMIC-IV construction and
training require credentialed local inputs and are not run by the unrestricted
smoke command. The smoke path validates executable invariants; it is not a
substitute for numerical paper reproduction.
