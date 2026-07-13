# Aggregate Evidence

These CSVs are publication-safe aggregate outputs. They contain no MIMIC-IV
rows, identifiers, timestamps, split membership, trajectories, row-level
predictions, or credentials.

- `core/`: controlled dynamics leaderboard and horizon/seed rank diagnostics.
- `rich_action/`: exact-sepsis-reference task roles and matched fixed-family
  transition summaries; this audit never replaces the compact core ranks.
- `uncertainty/`: same-family one-step and conditional-recursive NLL, Cov90,
  and Width90. Unavailable proper-score and decomposition fields remain explicit.
- `temporal/`: later-period patient-disjoint rank confirmation; no retuning or
  model selection is permitted.
- `external/`: bounded eICU evaluator-portability summaries. The mappings are
  nonidentical and are not phenotype-matched external validation.
- `quarantine/policy/`: policy registry, behavior calibration, overlap, WIS/WPDIS,
  clipping, ESS, and FQE diagnostics. All rows remain diagnostic-only.
- `audits/`: KDD091 headline reconciliation, KDD093 subject-cluster feasibility,
  and KDD094 OPE code-to-document provenance. Audit rows override broader prose
  whenever a convention is marked blocked.

The evidence files preserve their immutable internal task and experiment IDs for
provenance. User-facing interpretation follows [CANONICAL_SURFACES.md](../CANONICAL_SURFACES.md).
