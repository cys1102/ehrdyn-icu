# KDD212 public entrant workflow

KDD212 is a public, aggregate-safe release candidate. It contains a repaired
constructed POMDP smoke surface, bounded repeated-dataset OPE checks, transition
submission validation, and the aggregate bundle used by the manuscript. It
does not distribute MIMIC data, restricted derivatives, trained checkpoints,
row-level probabilities, or patient trajectories.

## Clean install and deterministic smoke

Python 3.11 is the supported runtime. Dependencies are pinned by
`pyproject.toml` and `uv.lock`.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --no-deps -e .

ehrdyn-icu validate-config --config-dir configs/tasks
ehrdyn-icu validate-transition-submission \
  --submission fixtures/transition_submission_small.json \
  --config-dir configs/tasks
ehrdyn-icu pomdp-smoke \
  --config configs/public_pomdp/kdd198_repaired_v2.json \
  --profile aki --environment-seed 21201 --episodes 64 --seed 3408 \
  --output build/kdd212/pomdp_smoke.json
ehrdyn-icu ope-smoke \
  --config configs/public_pomdp/kdd198_repaired_v2.json \
  --profile aki --environment-seed 21201 --datasets 4 --episodes 64 \
  --bootstrap 8 --seed 3411 --output build/kdd212/ope_smoke.json
ehrdyn-icu rebuild-public-bundle \
  --bundle public_bundle --output build/kdd212/manuscript
python -m unittest discover -s tests
ehrdyn-icu scan-release --root .
ehrdyn-icu verify-checksums --root .
```

The expected smoke hashes are recorded in
`fixtures/kdd212_public_smoke_hashes.json`. The bounded OPE run is intentionally
smaller than KDD202B and cannot replace its manuscript evidence.

## Credentialed reconstruction boundary

An uncredentialed user cannot reconstruct MIMIC-derived aggregates. Authorized
users may follow `PUBLIC_RECONSTRUCTION.md` and `MIMIC_ACCESS.md` locally, but
must not publish restricted intermediate arrays, role membership, row keys, or
patient-level outputs. Only separately reviewed aggregate receipts may enter a
future public bundle.

KDD205 independent reconstruction requires a genuine non-author executor with
their own valid credential, a clean clone, and no private author inputs. This
release contains no such attestation.
