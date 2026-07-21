# KDD220AR7 result audit

## Identity and scope

- Base public commit: `94e3362fcf20888f8744e435174d9f3292314b0b`.
- Branch: `kdd220ar7-bounded-memory-repair`.
- AR6 decision hash: `60a2cca17d8600c715e16d368485e2d33b25f7ea800b5292f7661d7c662d292a`.
- AR6 release MANIFEST hash: `168af6b3c36b2ee5a2c6491fcc80941fa12834c9068e1577fd690f078f95b053` (483 entries).
- Runtime config: `d731ea37e4cb47e20e76dc7d6de8160fe361da88ebf6d2385d2a50b096f84c89`.
- Aggregate schema: `1f2291181f996d668830169c7ce9fd9bc536506a0628232b70093df01ddc9694`.
- Dependency lock: `20fdf5d204803e8b94e3afaf70d9b43e15d2d3aed34f0c6ea09be48f2d64e223`.
- Aggregate-only BR5 resource receipt: `8b528ed5cc0abe3a40ef5ac7a92f068b1207fd54abf0070415fbe6fbcd13fa14`; it records a stopped 84,777,864 KiB peak. No private BR5 runtime output was opened.

No MIMIC-IV source, credential, private path, patient-level reconstruction,
KDD152 row-level reference, trained artifact, checkpoint, policy output, OPE
output, or expected clinical aggregate was accessed.

## Verification commands

1. `.venv/bin/python -m unittest tests.test_kdd220ar7 -v` — 5/5 passed.
2. `.venv/bin/python -m unittest tests.test_kdd220ar6 -v` — 7/7 passed.
3. `.venv/bin/python -m unittest discover -s tests -p 'test_*.py'` — 119/119 passed.
4. `.venv/bin/python -m compileall -q src scripts tests` — passed.
5. `.venv/bin/ehrdyn-icu validate-schemas --schema-dir schemas` — seven Draft 2020-12 schemas passed.
6. `.venv/bin/ehrdyn-icu verify-checksums --root .` — passed.
7. `.venv/bin/ehrdyn-icu scan-release --root .` and focused prohibited-field/path scans — zero findings.
8. `git diff --check` — passed.
9. Isolated `uv run --frozen --extra credentialed` fixture checks on Python 3.11, 3.12, and 3.13 — scientific hash exact.
10. AR6/AR7 generated stress probes at 100k, 500k, and one million rows — canonical row digests identical. At one million rows peak RSS changed from 243,184 KiB to 200,088 KiB.
11. Fresh staged-tree worktree: frozen installation, complete tests, schemas, checksums, privacy scan, and documented synthetic reconstruction module — passed.

The Python 3.11 synthetic aggregate receipt is byte-identical to AR6
(`8f125bbb524132b79db68fa4a28b4bf04aa5438ffc742ebb979bb15dec405f8a`).
Every tested chunk/compression/Python variant has scientific hash
`f412a1c03d2325339543628384c4aad14dd0ffbf92160a4d51d11c4c42b750a3`.

## Required-output inventory

All prompt-required KDD220AR7 outputs are present: frozen contract, AR6-to-AR7
map, stage schema, diagnosis, design, five-task parity, chunk/compression/order/
Python parity, generated stress results, temporary-storage/privacy receipt,
schema/test/checksum/clean-clone receipt, KDD220BR6 authorization, failure
ledger, this audit, and decision.

## Claim boundary

KDD220AR7 establishes an exact synthetic scientific-surface port and a lower-
RSS bounded-processing implementation on the frozen generated stress workload.
It cannot establish the real MIMIC peak or credentialed parity. Only the exact
pushed KDD220AR7 commit may authorize KDD220BR6. This stage authorizes no
handoff tag, independent reconstruction claim, scientific rerun, or manuscript
result.
