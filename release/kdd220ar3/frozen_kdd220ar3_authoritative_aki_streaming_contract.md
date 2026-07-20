# KDD220AR3 authoritative AKI and streaming contract

KDD220AR3 is a code-only repair from public commit
`c752060608e70b991bac84343d6867a9dfaecc48`. It does not access MIMIC-IV,
KDD152 interfaces, restricted arrays, models, checkpoints, policies, OPE, or
hidden expected results.

The stopped KDD220BR and KDD220C receipts passed both separately named digest
definitions and every anchor hash before editing. Their byte contents were not
copied, normalized, or changed.

The scientific contract remains unchanged: five retained tasks, frozen lineage
router and roles, subject-disjoint splits, 33-feature SAFE interface, masks,
actions K25/K25/K25/K4/K2, rewards, termination, 4-hour bins, 24-hour history,
48-hour follow-up, 72-hour episode, 18 bins, and the distinct 96-hour RV01R raw
extraction buffer.

The authorized repair has two parts only:

1. AKI creatinine rows are filtered for required identifiers, time, item and
   positive numeric value before integer conversion. Events use deterministic
   admission-local ordering and distinct monotonic 48-hour and seven-day
   baseline windows. The absolute gate is at least 0.3 within 48 hours; the
   ratio gate is at least 1.5 within seven days.
2. High-volume event sources use required-column chunked readers, early item,
   key and candidate-window filtering, stable source-order tie breaking, and
   no unbounded `_read` fallback. The versioned default is 250,000 source rows
   per chunk.

Missing required keys/times are excluded at their earliest relevant event
eligibility step and counted in an aggregate sidecar. Nonempty malformed keys
or times fail with table, field and aggregate reason only. Missing identifiers
are never mapped to sentinel entities.

Synthetic memory evidence is implementation evidence only. KDD220BR2 must
verify peak RSS no greater than 64 GiB on credentialed MIMIC-IV v3.1; the design
target is no greater than 32 GiB. KDD220AR3 authorizes neither scientific reruns
nor credentialed or independent reconstruction claims.
