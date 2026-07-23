# KDD245V2R frozen release contract

KDD245V2R binds the public scorer to:

- canonical constructor source commit
  `79deff45c5d4e5349e4ce648e212e6fbce1a27bd`;
- KDD220M2R2A authority commit
  `f2c6d770e41c284895050c46ef521dbba17d42d4`;
- R2A decision
  `authorize_kdd220m2r2_scientific_outputs_after_reporting_only_privacy_adjudication`;
- immutable scientific surface
  `848be7b3103f6272c020e4ba1a7c23fe278c51bfd4aaf2d7ad53a171569a5505`;
- scientific configuration
  `46026d7d59959c80d7518647423174ae9c543504ac1020ed2da65b28a799bea6`.

The original KDD220M2R2 decision remains a terminal stop. R2A is a separate
reporting-only authorization and does not rename that run.

The scorer versions are `ehrdyn-icu-canonical-v2.0.0` and
`ehr-component-scorer-v2.0.0`. It supports the five canonical-v2 tasks,
33-feature one-step and recursive forecasts, point/Gaussian/ensemble
prediction forms, probabilistic and calibration metrics where defined,
termination scoring, and retrospective support/evaluability diagnostics.

Planning, direct return, known real-EHR policy value, treatment benefit, causal
effect, clinical utility, deployment, and independent reconstruction are not
supported. No MIMIC file or MIMIC-derived scientific result bundle is included
in the software release.

Publication is permitted only after complete tests, schemas, synthetic
expected scores, clean-clone installation, manifest, checksum, privacy,
private-path, credential, identifier, documentation-link, citation, license,
and remote collision gates pass.
