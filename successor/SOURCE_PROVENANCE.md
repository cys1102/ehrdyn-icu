# Successor source provenance

The successor contract is derived from KDD-RV00R, KDD-RV01R, and KDD-RV02R.
The exact construction and model/evaluation backend is pinned to world-ehr
commit `64d76a34b7e14db364f6025889e82289d8bdd8c2`. Every required source and
source-contract test is listed with a SHA-256 digest in
`../src/kdd2027_benchmark/rv/contracts/source_manifest.csv`.
The byte-identical RV00R task table is in
`contracts/task_contract_candidates.csv`; its whole-file and canonical-row
hashes are checked by `rv-validate-config`.

The expanded verifier checks the origin repository, exact Git HEAD, tracked
membership at that commit, and byte hashes for 25 RV modules, direct
project-local construction dependencies, contract tests, and selected
execution receipts. It does not import or execute the credentialed backend.

The installable `kdd2027_benchmark.rv` namespace implements the portable
contract surfaces needed to validate five task identities, reproduce the
subject-role hash, validate train-only normalization, run aggregate-only local
evaluation, perform paired subject-cluster bootstrap inference, and exercise
the conditional-recursive state update on synthetic inputs.

The credentialed adapter verifies and invokes the pinned backend rather than
silently translating the historical KDD089 SQL builder. This is intentional:
the historical builder has a different cohort contract and must not be
relabeled as the successor construction.

This source verification is still not a self-contained public construction
package. Python/system dependencies, credentialed MIMIC-IV access, and the
ResearchForge control records remain external. Passing it must not be called
independent reconstruction.

Remaining external release gates are independent credentialed reproduction,
final-manuscript parity, five completed clinical task reviews, and the public
release decision. None is simulated by this release candidate.
