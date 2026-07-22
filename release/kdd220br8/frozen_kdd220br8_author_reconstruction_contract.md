# KDD220BR8 author-side reconstruction contract

KDD220BR8 is an author-side credentialed reconstruction from immutable public
commit `0331aa36e5b09824d5a1a04dc5b189976458ddaa`. It is not independent
reconstruction or external validation. Source, configuration, schemas,
dependencies, Python 3.11, the 250,000-row chunk size, the MIMIC-IV v3.1
CSV.GZ input contract, comparison axes, and respiratory-first reference order
were frozen before execution.

The public constructor was invoked exactly once. It stopped before validating
the input layout because the private candidate output directory already
existed. The frozen constructor explicitly rejects an existing output path.
No MIMIC table or authoritative reference was opened, no candidate aggregate
was produced, and no retry or in-stage repair was attempted.

All parity axes are therefore `not_run`. This runtime/setup stop provides no
evidence for or against scientific-interface parity and authorizes neither a
scientific rerun nor KDD220C8.
