# KDD220B result audit

The run began from a clean clone detached at the exact complete KDD220A commit.
Configuration and schema hashes matched. The only runtime input was official
MIMIC-IV v3.1 flat files through a local root argument. No historical result
directory was supplied.

The initial local root exposed both CSV and CSV.GZ copies and correctly failed
the exactly-one-encoding gate. A read-only local view selected the official
compressed encoding without changing file contents. The public constructor
then stopped in `load_core`: it asserts that no ICU stay has missing or
nonpositive time order before filtering those rows. The frozen authoritative
lineage filters invalid ICU stays as an eligibility step. This is a public
runtime/source-closure failure occurring before any task cohort was built.

Per the preaccess contract, historical aggregate/interface reference evidence
was not opened. Therefore cohort, SAFE-feature, preprocessing, action, reward,
and termination parity are unknown, not failed or inferred. No model, policy,
OPE estimator, or constructed environment was executed.

The stopped outcome does not authorize a scientific rerun, a public repair in
this stage, KDD220C, a commit, a push, or a tag. A separately reviewed runtime
repair must align invalid-stay filtering with the frozen authoritative source,
add an official-shape regression fixture, and repeat KDD220B from a new exact
commit before parity can be assessed.
