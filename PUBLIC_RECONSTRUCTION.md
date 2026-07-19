# KDD187 author-side reconstruction release candidate

Evidence role: `author_side_documentation_readiness`. This release has not
been executed by an independent credentialed user. KDD182 remains externally
blocked and is not superseded by these checks.

## Rocky Linux setup

On Rocky Linux 8 or 9 with Miniforge/conda installed:

```bash
conda env create -f environment-rocky-linux.yml
conda activate ehrdyn-icu-kdd187
uv sync --frozen --extra credentialed
```

`uv.lock` is the dependency lock. `uv sync --frozen` must fail rather than
rewrite it. The core public-only commands do not need MIMIC-IV or credentials.

## Public-only bounded workflow

```bash
uv run ehrdyn-icu validate-config --config-dir configs/tasks
uv run ehrdyn-icu generate-fixture --output /tmp/ehrdyn_fixture.csv --episodes 3 --seed 7
uv run ehrdyn-icu evaluate --fixture /tmp/ehrdyn_fixture.csv \
  --task-config configs/tasks/kdd2027_sepsis_vasopressor_3bin.json \
  --output /tmp/ehrdyn_metrics.json
uv run ehrdyn-icu validate-manifest \
  --task-manifest contracts/paper_task_manifest.csv \
  --contract-manifest contracts/paper_contract_manifest.csv \
  --evidence evidence/core/contract_transition_leaderboard.csv
uv run python -m unittest discover -s tests
uv run ehrdyn-icu scan-release --root .
uv run ehrdyn-icu verify-checksums --root .
```

The synthetic fixture is an interface check, not a clinical reconstruction.

## Credentialed reconstruction

Use MIMIC-IV v3.1 only inside an approved credentialed environment. Run
`credentialed/sql/00_base_eligible_stays.sql` through
`credentialed/sql/45_static_context.sql` in numeric order. For the additive
corrected sepsis lineage, then run
`credentialed/sql/current_lineage/46_kdd184_corrected_sepsis_dbp.sql` and
export `observation_events_kdd184_corrected` instead of the historical
`observation_events` view. Export the three documented internal surfaces to
secure local storage, then run:

```bash
uv run python credentialed/build_local_contract.py \
  --observations /secure/ehrdyn/observation_events.csv \
  --actions /secure/ehrdyn/action_exposures.csv \
  --static-context /secure/ehrdyn/static_context.csv \
  --expected credentialed/expected/frozen_task_aggregate_checks.csv \
  --output-dir /secure/ehrdyn/contract-v1
```

The builder output is restricted. Do not commit or transmit arrays,
preprocessing objects, identifiers, split membership, timestamps, predictions,
probabilities, or trajectories. Only a separately privacy-reviewed aggregate
receipt may leave the credentialed environment. Parity failure is reported as
a failure; it must not be repaired by filtering rows or changing tolerances.
The KDD184 DBP rule accepts only item IDs 220051, 220180, and 225310 with
documented `mmHg` units and values in the prespecified inclusive 10--200 mmHg
support. It is source/unit filtering, not post-hoc clipping.

## Current-lineage additive contract

`contracts/kdd187_five_task_contract_manifest.csv` records the latest five-task
aggregate contract. The corrected KDD184 sepsis DBP lineage is additive and
does not delete the historical E060 manifests. KDD186 uncertainty-penalty
sensitivity is included as optional aggregate evidence, not as a core
credentialed reconstruction dependency.
