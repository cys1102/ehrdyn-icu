# Successor submission integrity

The prior one-row aggregate JSON is retained only as a negative compatibility
example. It is deliberately rejected because self-reported metrics do not bind
predictions, normalization, split lineage, or complete target-cell coverage.

Accepted submissions are JSON files written by `rv-evaluate-local` or
`rv-evaluate-fixture`. Their evaluator receipt binds:

- benchmark and evaluator version;
- frozen target-cell evaluation-contract hash;
- prediction and normalization hashes;
- task/version/mode coverage and cell-set digests;
- aggregate payload hash;
- complete-cell attestation and synthetic status.

Credentialed results below 100 subject clusters or 1,000 observed cells per
primary metric row are rejected from public aggregate submission. These floors
are disclosure controls, not evidence of statistical precision or clinical
safety.
