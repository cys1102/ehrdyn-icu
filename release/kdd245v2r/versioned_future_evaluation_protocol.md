# Versioned future evaluation protocol

Every submitted result must bind the benchmark version, evaluator version,
task, action count, maximum recursive horizon, feature count, schema hashes,
and canonical serialization contract before scoring.

The v2.0.0 evaluator definitions, 1e-12 declared identity tolerance,
50/80/90/95 percent interval levels, risk-retention grid, termination
threshold, support formulas, and 100-subject/100-episode suppression rule may
not change after entrant results are inspected. A change requires a new
evaluator version and migration note; mixed-version pooling is prohibited.

Future protected or independent evaluation must run once, freeze aggregate
outputs before opening expected manifests, preserve failures and Structural
N/A cells, and expose no row-level differential. It may test reproducibility
but cannot convert real-EHR retrospective diagnostics into planning, direct
return, known policy value, causal effects, treatment recommendations, or
clinical utility.
