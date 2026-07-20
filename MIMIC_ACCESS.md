# MIMIC-IV Access and Local Setup

MIMIC-IV v3.1 is credentialed data. This release does not redistribute it.

1. Create a PhysioNet account and complete credentialing.
2. Complete approved human-subject and data-privacy training. PhysioNet
   currently documents the CITI `Data or Specimens Only Research` route.
3. Sign the PhysioNet Credentialed Health Data Use Agreement for MIMIC-IV v3.1.
4. Obtain MIMIC-IV v3.1 through PhysioNet or an approved linked service.
5. Build the MIMIC-IV concepts required by the versioned task cards and local
   extraction implementation, including sepsis, SOFA, ventilation, vitals,
   vasopressor, and KDIGO concepts.
6. Run extraction only in the credentialed environment. Replace schema names
   through database configuration rather than copying source tables.

## Current Five-Task Flat-File Candidate

KDD220A source-closes the current retained five-task candidate against the
official MIMIC-IV v3.1 CSV/CSV.GZ layout. PostgreSQL and derived concept tables
are not required for this candidate. KDD220A is synthetic source-differential
evidence only; real-EHR parity remains pending KDD220B.

From a credentialed local environment, after KDD220B authorization, the
candidate command is:

```bash
python -m pip install -e '.[credentialed]'
python -m kdd2027_benchmark.current_five_task.reconstruct \
  --mimiciv-root /LOCAL/PRIVATE/MIMIC-IV/3.1 \
  --schema schemas/credentialed_aggregate_receipt.schema.json \
  --output /LOCAL/PRIVATE/kdd220b-receipt
```

The MIMIC root is supplied only at runtime and must not be committed or printed
in a shareable receipt. The source-closed temporal contract is 24 hours before
and 48 hours after the final anchor (72 hours, 18 four-hour bins). A separate
96-hour post-base-anchor raw extraction buffer accommodates an allowed sepsis
anchor shift of up to 48 hours; 96 hours is not an episode, forecasting,
reward, policy, or OPE horizon.

## Historical Seven-Task SQL Path (Superseded for Current Reconstruction)

The following path is preserved for historical provenance only and must not be
used as the current five-task reconstruction. It previously ran
`credentialed/sql/00_base_eligible_stays.sql` through
`credentialed/sql/45_static_context.sql` in numeric order. Export the following
internal views only to secure local storage:

- `ehrdyn_icu_internal.observation_events`
- `ehrdyn_icu_internal.action_exposures`
- `ehrdyn_icu_internal.static_context`

Then install the credentialed extra and build private arrays:

```bash
python -m pip install -e '.[credentialed]'
python credentialed/build_local_contract.py \
  --observations /secure/ehrdyn/observation_events.csv \
  --actions /secure/ehrdyn/action_exposures.csv \
  --static-context /secure/ehrdyn/static_context.csv \
  --output-dir /secure/ehrdyn/contract-v1
```

That historical command fails if its frozen aggregate parity is not reproduced. Its arrays,
preprocessing statistics, keys, and split membership are restricted local
artifacts and must never be committed. Only a separately reviewed aggregate
receipt may be considered for release.

The public extraction path prorates medication amounts by overlap with each
four-hour window and encodes RRT/CRRT as procedure-overlap fraction. Review the
local `amountuom` and `rateuom` distributions before execution: the compact
axes are recorded-exposure proxies and do not claim cross-drug dose
equivalence. The aggregate receipt must reproduce the frozen occupied action
cardinality for every task.

Official references:

- MIMIC-IV v3.1: https://physionet.org/content/mimiciv/3.1/
- Credentialed DUA: https://physionet.org/content/mimiciv/view-dua/3.1/
- Training instructions: https://physionet.org/about/citi-course/
- Credentialing FAQ: https://physionet.org/about/faqs/

Do not upload MIMIC-IV rows or restricted derivatives to public repositories,
third-party APIs, shared prompts, or unapproved online services. Keep local
database keys and split assignments inside the approved environment.
