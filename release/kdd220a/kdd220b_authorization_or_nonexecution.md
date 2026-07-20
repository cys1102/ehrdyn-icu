# KDD220B authorization

KDD220B author-side credentialed parity is authorized only from the exact
immutable KDD220A commit recorded after all KDD220A gates pass. The authorized
input is official MIMIC-IV v3.1 flat files supplied locally by CLI argument.

KDD220A itself did not access MIMIC, credentials, patient rows, restricted
arrays, checkpoints, or model outputs and does not claim real-EHR parity.
KDD220B must preserve mismatches and may not tune semantics using historical
cohort counts.
