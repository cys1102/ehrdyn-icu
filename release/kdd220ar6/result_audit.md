# KDD220AR6 result audit

Decision: `complete_exact_dual_surface_and_order_source_port_ready_for_kdd220br5`.

Base public commit: `9233375b9ef123b73ac262f1743749849be24cbf`. Branch: `kdd220ar6-dual-surface-order-port`.

Implemented boundaries:

- independent legacy action and repaired SAFE-state FiO2 parsers and masks;
- same-bin observed PEEP x legacy FiO2 K25 action contract;
- cutpoint and encoding before original membership, missing-action, SAFE repair, and KDD201 filtering;
- pre-repair membership and final repaired feature masks retained under separate names;
- KDD201 identical row/class subsetting with no refit or re-encoding;
- source-closed sepsis, shock, AKI, and HF dispositions for the non-respiratory BR4 deltas; and
- aggregate-only train/validation/historical_other/all-role stage counts and KDD152-compatible component digests.

Verification commands:

1. `.venv/bin/python -m unittest tests.test_kdd220ar6`
2. `.venv/bin/python -m unittest discover -s tests -p 'test_*.py'`
3. `.venv/bin/python -m compileall -q src tests/test_kdd220ar6.py`
4. Draft 2020-12 `check_schema` for every released schema plus positive/negative fixtures.
5. `UV_PROJECT_ENVIRONMENT=/tmp/kdd220ar6-py$PY uv sync --frozen --extra credentialed --python $PY` and the focused suite for Python 3.11, 3.12, and 3.13.
6. Cross-runtime synthetic reconstruction and `scientific_surface_sha256` comparison.
7. `ehrdyn-icu scan-release --root .`, hidden-path scan, prohibited-field scan, `git diff --check`, checksum generation, and checksum verification.
8. Fresh isolated Python 3.11 install followed by the focused synthetic reconstruction suite.

The focused suite passed 7/7 on each supported Python runtime. The complete public suite passed after checksum regeneration. The runtime-bearing JSON receipt records the actual interpreter and chunk instrumentation; its separately named scientific aggregate surface hash excludes those operational fields and is byte-identical across Python 3.11/3.12/3.13 and reader chunk sizes: `f412a1c03d2325339543628384c4aad14dd0ffbf92160a4d51d11c4c42b750a3`.

No MIMIC-IV data, private KDD220BR4 runtime output, KDD152 row-level interface, expected clinical aggregate count, checkpoint, policy output, or OPE output was accessed. KDD220AR6 authorizes KDD220BR5 from the exact pushed commit only; it proves neither credentialed aggregate parity nor independent reconstruction and authorizes no scientific rerun.
