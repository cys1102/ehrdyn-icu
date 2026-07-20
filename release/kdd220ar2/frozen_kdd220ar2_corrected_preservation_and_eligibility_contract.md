# KDD220AR2 corrected preservation and eligibility contract

KDD220AR2 starts from KDD220A commit
`9db6dc1fc2c3b68645fe04934379e4e4b5f3f1cf`. It is a code-only repair and
does not access MIMIC, credentials, patient rows, historical cohort counts,
restricted arrays, checkpoints, model outputs, or row-aligned references.

The preserved stopped KDD220B receipt contains exactly 16 files. Its
repository-standard path-plus-bytes digest is
`db55da00eb1c32400a8003505abc26a6f1def5e7a5995ea451199f331bacb4bd`.
Its secondary path-plus-per-file-digest manifest identity is
`d1a2e18716e6a0ccd93e90dbce84698c5e21ee418344f3a2ef9af2920b229daa`.
The algorithms and names are intentionally distinct.

The only semantic change is the ordering of invalid ICU-stay eligibility:
after schema, uniqueness, and strict nonempty timestamp parsing, rows are
classified by the mutually exclusive precedence missing `intime`, missing
`outtime` when `intime` is present, equal times, reversed times, or valid
strictly increasing times. Invalid rows are removed before anchors, windows,
events, actions, rewards, or transitions. An invalid row cannot abort the full
release; an invalid time order remaining after selection remains a hard error.

All KDD220A task, lineage, temporal, SAFE-feature, action, reward,
termination, ordering, precision, schema, and privacy contracts remain frozen.
RV01R tasks receive no new minimum length. KDD121's existing 24-hour
task-specific eligibility remains separate and unchanged.
