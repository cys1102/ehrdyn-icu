# Validated successor aggregate evidence

This directory contains byte-identical copies of selected aggregate-only
KDD-RV01R and KDD-RV02R outputs. `evidence_manifest.csv` records the source run,
source filename, SHA-256 digest, byte count, privacy receipt, and packaged path.

Included RV01R construction evidence covers cohort/role counts, action-contract
selection, operability gates, zero role overlap, the construction report, and
the privacy receipt. RV01R did not open sealed-test performance.

Included RV02R evidence covers primary normalized observed-cell metrics,
horizon and feature-group views, paired subject-cluster deltas, practical
leader sets, Gaussian diagnostics, negative controls, the final report, and the
privacy receipt. Bootstrap replicate rows, preprocessing values, local
predictions, checkpoints, logs, and test-opening material are not included.

The byte-identical RV02R practical-leader table retains its historical
`unique_winner_claim_allowed` diagnostic for provenance. It is not an
authorized successor claim: the portable evaluator always emits `false` and
labels fixed-test-reference sets descriptive. The raw copied field must not be
used in a paper, leaderboard, or submission decision.

These files reproduce validated internal aggregate evidence; they do not by
themselves constitute an independent credentialed reproduction. Negative and
null results remain in the copied tables. Clinical review is still pending.

The evidence supports recorded-trajectory forecasting and benchmark-validity
auditing only. It does not support treatment, causal, counterfactual,
policy-selection, clinical-utility, deployment, or autonomous-decision claims.
