# Public versus protected evaluation boundary

All 40 KDD215 environments and their deterministic generator payloads are
public development benchmark environments. There is no protected final-seed
service in this release. The workflow is therefore a transparent,
deterministic development benchmark, not a hidden test server or lockbox.

Role separation is still enforced within each public run: public constructed
train data may fit an entrant, validation may select its checkpoint, and final
direct-return/OPE streams are evaluator-only. Because every mechanism and seed
is public, results should be interpreted as reproducible constructed-benchmark
performance rather than evidence of generalization to unseen mechanisms.
