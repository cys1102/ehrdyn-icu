# KDD220M1 canonical v2 migration contract

KDD220M1 changes the release criterion, not the retained historical evidence. Exact
equality to the KDD152-v2A/KDD184/KDD201 aggregate lineage is no longer required.
The historical lineage remains immutable provenance. The public constructor at
`0331aa36e5b09824d5a1a04dc5b189976458ddaa`, its version-controlled configuration,
and the frozen BR8R2 aggregate candidate define the candidate-v2 inputs.

The canonical identifier is `ehrdyn-icu-five-task-constructor-v2-kdd220m1`. It is
bound to the seven-file candidate tree digest
`8b6ecaab4ae4a42c387f3f34612f9cb0135b10e29bcd3a7e95b83056dca25d29`, the
constructor/config/schema/lock hashes in the identity receipt, and the aggregate
manifest in this directory. No patient membership, row-level match, identifier,
timestamp, trajectory, private path, or private array is part of the contract.

## Frozen semantics

- Inputs are one official MIMIC-IV v3.1 CSV or CSV.GZ encoding per required table.
- Invalid ICU time-order rows are excluded before anchors; duplicate primary keys
  and malformed nonempty timestamps fail closed.
- Compact-lineage tasks retain adult valid-time stays. Large-lineage tasks retain
  adult stays of at least 24 hours with the source-required gender and discharge
  fields. No new hospice or diagnosis gate is added.
- Large-lineage roles use the frozen SHA-256 subject salt. Compact-lineage roles
  use the frozen linear-congruential subject bucket. Roles are subject-disjoint.
- All tasks use four-hour bins, a 24-hour pre-anchor history, a 48-hour post-anchor
  limit, and an 18-bin/72-hour episode surface. The 96-hour RV01R quantity is only
  a raw extraction buffer. The longest recursive target sequence is 44 hours.
- The task-facing state is the frozen 33-feature SAFE surface. Observation masks,
  past-only carry-forward, capped recency, train-only mean/scale normalization,
  state/action/target order, and canonical array digests follow the public source.
- Sepsis uses the public E060-style blood-culture/antibiotic suspected-infection
  construction and the frozen final-anchor logic; its action is fluid-5 by
  vasopressor-5 and its terminal discharge-origin 90-day mortality proxy is emitted
  exactly once.
- Respiratory support anchors on the first structured ventilation/support event,
  including positive labeled input events. Its action is directly observed legacy
  PEEP-5 by legacy FiO2-5. Missing action is excluded, never imputed to class zero.
  Its dense reward uses strictly post-action SpO2 and MBP.
- Shock anchors on the first sustained-hypotension or positive vasopressor event,
  uses fluid-5 by vasopressor-5, and scores the strictly post-action next-MBP dense
  component.
- AKI anchors on the first two-window KDIGO creatinine event or time-stamped RRT
  start, uses diuretic by RRT context (K=4), and emits the terminal mortality proxy
  exactly once.
- Heart failure/decongestion requires no prior HF diagnosis; it anchors on the first
  current-stay diuretic or vasodilator prescription/input, uses binary diuretic
  exposure (K=2), and emits the terminal mortality proxy exactly once.
- Train-only positive cutpoints are frozen before target-membership, action, and
  KDD201 filtering. KDD201 removes only named first-decision dispositions and never
  refits actions.

## Migration interpretation

All five tasks differ from the retained manuscript lineage on at least one cohort,
role, transition, action, reward-availability, horizon, preprocessing, or digest
surface. These differences are accepted as versioned candidate-v2 outputs; they are
not evidence that historical results still apply. KDD220M2 must recompute every
real-EHR consumer listed in the refresh manifest. Constructed-environment results
are outside this refresh because their mechanisms and inputs are unchanged.

The frozen candidate receipt does not expose per-feature observation-rate or
quantile summaries. This is recorded as an aggregate comparison limitation, not
silently filled from row data. KDD220M2 must emit those summaries from the frozen v2
inputs before any manuscript synchronization.
