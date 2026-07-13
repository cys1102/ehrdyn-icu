# Independent Clinical Review Status

Status: **pending**.

The current task cards are literature-grounded and executable benchmark
contracts, but they have not yet received documented independent ICU and
cardiology adjudication. They must not be described as clinically validated.

The exact compact paper packets are under `clinical_review/core_task_packets/`.
The prior K25 audit packets remain under `clinical_review/task_packets/` and do
not substitute for review of the compact headline contracts. Review status is
machine-readable in `clinical_review/core_review_status.csv`. Reviewers should
return one completed copy of `clinical_review/reviewer_response_template.md`
per task. The benchmark maintainers then record the immutable response path,
decision, date, and adjudication commit in the status file.

Each reviewer should assess:

1. cohort anchor and index-event logic;
2. inclusion, exclusion, stay-overlap, and multiple-anchor handling;
3. observation source, timestamp, within-bin aggregation, and carry-forward;
4. action source, ordered-versus-delivered semantics, units, bin edges,
   combination rules, and missing-versus-no-action handling;
5. whether the four-hour action is a repeated decision or a sparse/few-step event;
6. reward component direction, scale, weighting, missingness, and terminal handling;
7. expected failure modes and prohibited interpretations.

The public review record should name requested changes, adjudicated decisions,
and unresolved disagreements. Clinical task-card approval does not establish
causal treatment effects, counterfactual validity, or clinical utility.

No row may be changed from `pending` by a software or language-model review.
Completion requires a named independent clinician with the relevant specialty.
