# KDD220A source-closed five-task reconstruction contract

KDD220A is an additive public-release repair of the stopped KDD217AR3AR
candidate. The KDD217AR3A and KDD217AR3AR evidence trees and stop decisions are
preserved byte-for-byte. This stage uses synthetic flat files only and makes no
credentialed, cohort-count, real-EHR parity, clinical, causal, or deployment
claim.

The sole historical result input was the aggregate backend configuration
receipt with SHA-256
`663c1fb4ead074be1057eeea1a97ed65d6e056b6a27c4d7df5fd6ef2025b06e2`.
Only allowlisted configuration fields were read. No cohort counts, model
outputs, expected returns, membership, row arrays, paths, or patient data were
used.

The public episode contract is 24 hours before and 48 hours after the final
anchor: 72 hours represented by 18 four-hour bins. The 96-hour quantity is a
post-base-anchor raw extraction buffer. It exists only to cover an allowed
sepsis base-to-final-anchor shift of up to 48 hours plus the frozen 48-hour
post-final-anchor episode. It is forbidden as an episode, policy, reward, OPE,
or forecasting horizon. State/action/target alignment yields at most eleven
recursive target offsets, ending at 44 hours after the final anchor.

All runtime-defining public values are closed by
`src/kdd2027_benchmark/current_five_task/runtime_config.json` and the hashed
source modules in the same package. The MIMIC-IV v3.1 root is accepted only as
a CLI argument. Runtime reads from result trees, artifact trees, private
manifests, author paths, or prior output trees are prohibited.

Completion authorizes KDD220B author-side credentialed parity from the exact
KDD220A commit. It does not authorize independent reconstruction or a final
release tag.
