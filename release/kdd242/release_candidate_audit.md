# KDD242 release-candidate audit

The candidate is based on public KDD235A commit `4739d59` and frozen KDD235B
implementation commit `5e6cbd7`.

Prepublication verification:

- 133/133 public tests pass, including negative entrant fixtures;
- 11/11 Draft 2020-12 schemas validate;
- point, independent-Gaussian, and Gaussian-ensemble quickstart runs are
  byte-identical across two executions;
- KDD235B has 40 checkpoint, 440 forecast-horizon, 40 direct-return, and 240
  OPE summary rows;
- the 64-dataset OPE contract uses 256 episodes per dataset and refits its
  denominator and nuisance dynamics inside every dataset;
- checksum, documentation, forbidden-import, and privacy checks pass;
- the demonstration entrant is not inserted into the scientific leaderboard.

Publication remains conditional on a clean anonymous-clone install and smoke,
absence of a remote `v1.3.0` collision, immutable annotated-tag creation, and a
post-publication archive replay.
