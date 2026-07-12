# Aggregate Evidence

These CSVs are publication-safe aggregate outputs. They contain no MIMIC-IV
rows, identifiers, timestamps, split membership, trajectories, row-level
predictions, or credentials.

- `core/`: controlled dynamics leaderboard and horizon/seed rank diagnostics.
- `temporal/`: later-period patient-disjoint rank confirmation; no retuning or
  model selection is permitted.
- `policy/`: policy registry, behavior calibration, overlap, WIS/WPDIS,
  clipping, ESS, and FQE diagnostics. All rows remain diagnostic-only.

The evidence files preserve their immutable internal task and experiment IDs for
provenance. User-facing interpretation follows [CANONICAL_SURFACES.md](../CANONICAL_SURFACES.md).

