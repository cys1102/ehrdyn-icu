# KDD235A result audit

## Outcome

The public package now has an entrant-owned, typed, versioned recursive probabilistic
world-model path over the repaired constructed benchmark. The bounded smoke exercised sepsis
(K=25) and AKI (K=4), seed 171901, with point and independent-Gaussian entrants. It produced
one-step and recursive errors, uncertainty metrics or structural NA, frozen H4 planning,
common-random-number direct return, learned/direct return gap, and OPE-ready probabilities.

## Identity and preflight

- Isolated base: `f030c5793e4519a2b961eda1085d0bbeef29dbd0`.
- Branch: `kdd235a-recursive-world-model-entrant`.
- KDD212, KDD215R, and KDD216R are verified ancestors.
- Repaired KDD198/KDD199 result-tree and source identities pass.
- KDD224/KDD232 metric, uncertainty, H4 planner, seed, support, and tie contracts pass.
- No older generator, reward, termination, planner, or validation definition was substituted.

## Verification

| Gate | Command or evidence | Result |
|---|---|---|
| Focused tests | `uv run python -m unittest tests.test_kdd235a -q` | 11/11 pass |
| Full public tests | `uv run --frozen --extra credentialed python -m unittest discover -s tests -q` | 130/130 pass |
| Schemas | `uv run ehrdyn-icu validate-schemas --schema-dir schemas` | 11/11 Draft 2020-12 valid |
| Static import | `uv run python -m compileall -q src world_model_entrant_example tests/test_kdd235a.py` | pass |
| Diff hygiene | `git diff --check` | pass |
| Portability | canonical fixture on Python 3.11/3.12/3.13 | identical SHA-256 `c5ef848c...6356` |
| Clean source | isolated candidate copy, locked 3.11 install, focused tests, documented smoke | pass |
| Smoke | `evaluate-world-model-smoke`, two profiles, two entrants | four aggregate rows, pass |
| Privacy | release and prohibited-field scans | pass |

Positive paths cover deterministic point, single Gaussian, and Gaussian ensemble outputs.
Negative paths cover missing/nonbinary masks, wrong actions, unsupported actions, inconsistent
horizons, nonfinite values, invalid policy sums, fabricated point uncertainty, ensemble variance
identity, component-source mismatch, and a recursive shape failure after a valid one-step result.

## Compatibility and claim boundary

Existing KDD215 entrant declarations and `kdd215.runtime.v1` remain the default runtime behavior.
KDD235A opts into its own declaration schema and `kdd235a.runtime.v1`; no existing public API was
removed. The interface validates executable structure and deterministic transport, not arbitrary
model correctness. Results concern constructed environments only. They do not establish EHR
counterfactual truth, clinical utility, treatment recommendations, causal effects, deployment
readiness, or independent credentialed reconstruction.
