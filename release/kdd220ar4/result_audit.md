# KDD220AR4 release-finalization audit

## Frozen identity

- Base and pre-commit HEAD:
  `78a823fc428a0d7b241242f874e4926e3ad65417`
- Branch: `kdd220ar4-respiratory-streaming-repair`
- Candidate source/test/schema patch identity:
  `18e908593fe1a29f9c5ab0abdef377aa69c313d3875acf0e4028f2eca55f7a2e`
- Preserved release inventory before this audit: 16 files
- Decision token:
  `complete_respiratory_missing_action_filter_repair_ready_for_kdd220br3`
- Failure ledger: no unresolved failure

The identity was computed exactly with:

```bash
( git diff --binary -- schemas src tests; \
  for f in schemas/credentialed_controlled_stop_receipt.schema.json \
           tests/test_kdd220ar4.py; do \
    printf '%s\\0' "$f"; sha256sum "$f"; \
  done ) | sha256sum
```

No source, configuration, schema, test, scientific contract, or existing
receipt was edited during finalization. No MIMIC-IV file, KDD152 result,
private path, patient row, identifier, timestamp, membership, or credential
was opened.

## Verification commands and outcomes

```bash
/tmp/ehrdyn-icu-kdd220ar4-zdw2Kg/venv/bin/python \
  -m unittest -v tests.test_kdd220ar4
/tmp/ehrdyn-icu-kdd220ar4-zdw2Kg/venv/bin/python \
  -m unittest discover -s tests -v
/tmp/ehrdyn-icu-kdd220ar4-zdw2Kg/venv/bin/python -m compileall -q src tests
/tmp/ehrdyn-icu-kdd220ar4-zdw2Kg/venv/bin/ehrdyn-icu \
  validate-schemas --schema-dir schemas
/tmp/ehrdyn-icu-kdd220ar4-zdw2Kg/venv/bin/ehrdyn-icu scan-release --root .
/tmp/ehrdyn-icu-kdd220ar4-zdw2Kg/venv/bin/ehrdyn-icu verify-checksums --root .
git diff --check
```

- Focused KDD220AR4 suite: 7/7 passed.
- Full public suite: 97/97 passed after the required checksum-manifest rebuild.
- Python compilation: passed.
- Released Draft 2020-12 schemas: all passed.
- Positive and negative schema cases: passed through the public and focused
  test suites.
- Checksum manifest: regenerated from the complete public file inventory and
  verified without mismatch.
- Privacy and prohibited-field scan: passed with zero findings.
- Isolated Python 3.11 install used the documented `.[credentialed]` extra;
  schema validation and eight focused synthetic/nonregression tests passed.
- Git whitespace/error check: passed.

The initial full-suite invocation before manifest regeneration passed 96 tests
and failed only the expected stale-manifest checksum assertion. It was not
accepted as final evidence; the manifest was rebuilt and the entire 97-test
suite was rerun successfully.

## Scientific and privacy boundary

Existing aggregate receipts and tests verify that missing respiratory PEEP or
FiO2 excludes only the corresponding transition, never maps to class 0, and
never invokes forward/future/state imputation. Retained respiratory classes
remain in the frozen K=25 range; state/action/target order remains aligned;
non-respiratory synthetic interfaces remain unchanged; and chunk-size and
CSV/CSV.GZ outputs remain scientifically invariant. Streaming instrumentation
is aggregate-only.

KDD220AR4 authorizes KDD220BR3 to start from the exact pushed commit. It does
not prove credentialed reconstruction parity, independent reconstruction,
external validation, clinical utility, or any need for a scientific rerun.

## Required output inventory

All prompt-required files are present under `release/kdd220ar4/`:

- `frozen_kdd220ar4_respiratory_missing_action_contract.md`
- `preserved_kdd220ar3_source_identity.csv`
- `preserved_stopped_kdd220br2_and_kdd220c2_identity.csv`
- `authoritative_respiratory_action_filter_source_map.csv`
- `respiratory_missing_action_before_after.csv`
- `respiratory_action_filter_regression_results.csv`
- `transition_order_and_index_alignment_receipt.csv`
- `streaming_instrumentation_contract.csv`
- `five_task_synthetic_nonregression.csv`
- `chunk_size_and_compression_invariance.csv`
- `clean_clone_smoke_receipt.csv`
- `schema_checksum_and_privacy_scan.csv`
- `kdd220br3_resource_contract.md`
- `failure_ledger.csv`
- `result_audit.md`
- `decision.md`
- `kdd220br3_authorization_or_nonexecution.md`

Finalization decision:
`complete_kdd220ar4_release_finalized_ready_for_kdd220br3`.
