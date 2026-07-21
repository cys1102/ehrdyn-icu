# Frozen KDD220AR5 current-lineage source-port contract

KDD220AR5 is a code-only, synthetic source-differential stage based on public
commit `d8c99bd48c74049b16b3d4ddc5b7ca15fe73b521`. It does not access MIMIC-IV,
KDD152 row-level interfaces, model outputs, checkpoints, or expected clinical
aggregate counts.

The retained tasks are sepsis, respiratory support, shock, AKI, and heart
failure/decongestion. Decisions are four hours. The task interface spans 24
hours before and 48 hours after the anchor, for 72 hours and 18 bins. The
96-hour value is an upstream compact-lineage raw extraction buffer only; it is
not an episode, forecasting, reward, policy, or OPE horizon. Roles remain
subject-disjoint train, validation, and historical_other, with no sealed-role
opening. Action dimensions remain K25, K25, K25, K4, and K2. The model-facing
forecasting interface remains the frozen 33 SAFE features.

The source port is constrained to the named authoritative commits and symbols
in `authoritative_source_commit_and_symbol_map.csv`. It changes no model,
policy, estimator, support threshold, reward definition, or termination rule.
Clinical aggregate counts are not a test oracle in this stage.

The completion gate requires source mapping, synthetic stagewise and boundary
fixtures, chunk/compression invariance, focused and full public tests,
`py_compile`, Draft 2020-12 schema checks, checksum verification, privacy and
hidden-path scans, and an isolated-install synthetic smoke. Completion
authorizes KDD220BR4 from the exact pushed commit but establishes neither
credentialed EHR parity nor independent reconstruction.
