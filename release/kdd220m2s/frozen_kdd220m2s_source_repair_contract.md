# Frozen KDD220M2S source-repair contract

KDD220M2S permits only a null-safe deterministic implementation repair in
`blood_culture_events`. Culture text uses pandas nullable-string semantics,
nullable numeric identifiers use explicit numeric coercion, and group-level
`max` remains the deterministic aggregate. Missing organism values remain
missing and never become positive string literals.

The blood-specimen filter, chart-time fallback, positive-culture rule,
excluded organism/item rule, suspected-infection pairing windows, anchor
selection, and stable sort order are unchanged. Cohorts, roles, tasks,
features, actions, rewards, termination, schemas, tolerances, and MIMIC-IV 3.1
are frozen.

Credentialed execution is limited to one constructor invocation against the
explicit operator-provided root. The output must be frozen before scientific
comparison. A controlled stop is not a candidate and cannot authorize M2R.
