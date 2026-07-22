# KDD220AR8 result audit

## Identity and preservation

- Base public commit: `f030c5793e4519a2b961eda1085d0bbeef29dbd0`.
- Branch: `kdd220ar8-localized-respiratory-parity`.
- Runtime config SHA-256: `d731ea37e4cb47e20e76dc7d6de8160fe361da88ebf6d2385d2a50b096f84c89`.
- Aggregate schema SHA-256: `1f229de694f995b9a500270f3e411dab8bddfee46b15ac038bdb1a21b561f117`.
- Dependency lock SHA-256: `20fdf5d204803e8b94e3afaf70d9b43e15d2d3aed34f0c6ea09be48f2d64e223`.
- KDD220AR7 has 14 files, path-plus-bytes digest
  `0cb4fcbf691952494873c192705ea7d7976d9dace047e0e3056df85866fd0545`,
  and zero diff from the base commit.
- The immutable stopped KDD220BR7 aggregate receipt has 16 files,
  path-plus-bytes digest
  `cc737c7b0b1bf9becaf841277554cc6d960e675f89e610397c37b6f315c4f931`,
  and decision SHA-256
  `be0b03818d9c3d9fe01ccf7ad1c2a5bcdac3b799e2b93d463e8b7e5455fde39a`.

No KDD220AR7 or KDD220BR7 file was modified. No MIMIC-IV source, credential,
patient-level value, identifier, membership list, row differential, restricted
reference, checkpoint, model result, or private runtime output was opened.

## Source finding and repair

The authoritative KDD097 source reads the complete frozen respiratory-support
label pattern from `procedureevents` and positive `inputevents`. The public
AR7 constructor retained only a shortened procedure-only path. A new fixture
using a positive `High Flow Nasal Cannula` input event failed before the edit
and passed afterward. The patch changes only respiratory anchor event
collection; all downstream transition, action, SAFE feature, reward,
termination, continuation, and role code is unchanged.

The repaired fixture produces one respiratory episode, 11 stay-bounded
candidate transitions, one frozen KDD201 removal, and 10 final decisions. Its
scientific-surface SHA-256 is
`b30a3f7dd0f30f5453dc9f6b513044adaae65169d8805943d0a4527f1fe6c45a`.
The unchanged five-task fixture retains AR6/AR7 scientific SHA-256
`f412a1c03d2325339543628384c4aad14dd0ffbf92160a4d51d11c4c42b750a3`.

## Verification

1. Focused KDD220AR8 tests: 2/2 passed.
2. Selected KDD220AR2--AR8 lineage regressions: passed.
3. Complete public test suite: 121/121 passed after the checksum manifest was refreshed.
4. `compileall` for `src`, `scripts`, and `tests`: passed.
5. All released Draft 2020-12 schemas and negative fixtures: passed.
6. Frozen Python 3.11, 3.12, and 3.13 scientific-surface portability: exact.
7. Checksum verification, privacy scan, prohibited-field scan, and `git diff --check`: passed.
8. Isolated clean installation and synthetic reconstruction smoke: passed.

## Required outputs and claim boundary

All prompt-required KDD220AR8 outputs are present. The result is a localized,
source-justified public-constructor repair ready for an author-side KDD220BR8
parity run. It is not credentialed reconstruction evidence, independent
reproduction, external validation, a scientific rerun, or a clinical claim.
