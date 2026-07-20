# KDD217AR3A Python flat-file reconstruction contract

Base: public tag `v1.2.0-bounded.1`, commit
`dda838b9867fbf6a75be391ca669f8a32954a580`, branch
`kdd217ar3a-python-reconstruction-candidate`.

The only authorized input is the official MIMIC-IV v3.1 flat-file release,
with exactly one CSV or CSV.GZ representation per required table. PostgreSQL is
not part of this contract. Python 3.11--3.13, stable merges/sorts, UTF-8 CSV,
UTC, IEEE-754 float64 contract calculations, and the locked public dependency
set are frozen. The runtime fails closed on a wrong `3.1` layout, ambiguous
compressed/uncompressed duplicates, missing columns, malformed dates, duplicate
primary identifiers, invalid ICU time order, unknown tasks, missing action
observations, schema failure, and an existing output directory.

The candidate regenerates intermediates locally and exports only a Draft
2020-12 validated task-level aggregate receipt. It does not read an E060
manifest, patient manifest, split manifest, checkpoint, prediction, or result
tree at runtime. This stage did not access MIMIC and does not claim aggregate
parity.

The frozen acceptance rule requires exact synthetic differential equality for
task inclusion, transition order, role, action, masks, reward masks,
termination, and counts, plus `rtol=1e-12, atol=1e-12` for floating values and
rewards. The observed mask and feature differences violate that rule.

