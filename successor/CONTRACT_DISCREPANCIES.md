# Frozen task-table and backend discrepancies

These findings were discovered during independent Round 5 artifact review.
They are retained as blockers and are not silently repaired in the frozen
scientific rows.

1. The task table describes validation selection among action alternatives.
   The inspected backend hard-codes the encodings and checks operability from
   training action counts; validation does not choose among alternatives.
2. The respiratory label says train-quantile five class. Duplicate edges
   collapse the realized training support to classes 0--2, and evaluator
   materialization excludes transitions without observed PEEP. Construction
   and evaluator-eligible denominators therefore differ.
3. The AKI task row states that chronic or pre-existing RRT is excluded. The
   inspected construction path unions qualifying creatinine events with the
   first time-stamped RRT event and has no separately verifiable exclusion
   receipt for that statement.

The local RC validates and exposes these discrepancies. Correcting any item
requires a new versioned task contract and, where rows change, a separately
registered credentialed rerun. Current aggregate results must not be relabeled
as if that repair had occurred.
