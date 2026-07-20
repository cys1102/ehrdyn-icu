# Public credentialed reconstruction instructions

KDD220A prepares a source-closed candidate; it does not run MIMIC-IV. KDD220B
must bind an exact KDD220A commit before credentialed execution.

Use Rocky Linux-compatible tooling, Python 3.11--3.13, and an authorized local
copy of the official MIMIC-IV v3.1 flat-file release. Both CSV and CSV.GZ are
accepted. The local root must be named `3.1` and contain exactly one supported
encoding for every relative table declared in `current_five_task/contracts.py`.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[credentialed]'
python -m kdd2027_benchmark.current_five_task.reconstruct \
  --mimiciv-root /LOCAL/PRIVATE/MIMIC-IV/3.1 \
  --runtime-config src/kdd2027_benchmark/current_five_task/runtime_config.json \
  --schema schemas/credentialed_aggregate_receipt.schema.json \
  --output /LOCAL/PRIVATE/kdd220b-output
```

The command fails closed on the wrong release layout, missing columns,
duplicate primary identifiers, malformed timestamps, unsupported source
encoding, missing task transitions, invalid action observations, runtime config
drift, or aggregate schema failure. It does not read a result tree, expected
cohort count, patient manifest, split manifest, checkpoint, or model output.

The output directory must not already exist. For restart, remove or archive an
incomplete local output only inside the credentialed workspace, rerun the same
immutable commit and command, and record the intervention locally. Share only
the schema-valid aggregate receipt after the prescribed privacy scan. Never
share the MIMIC path, source rows, identifiers, timestamps, trajectories,
membership, arrays, tensors, checkpoints, or logs containing protected paths.
