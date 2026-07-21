# KDD220AR6 frozen code-only contract

KDD220AR6 starts from public commit `9233375b9ef123b73ac262f1743749849be24cbf`. It is a source-port and synthetic differential stage only. It did not open MIMIC-IV, KDD152 row-level interfaces, KDD220BR4 private runtime output, checkpoints, policies, or OPE outputs.

The retained target is KDD097 -> KDD152-v2A -> KDD201: five tasks, four-hour decisions, -24/+48 hour episode window, subject-disjoint train/validation/historical_other roles, K25/K25/K25/K4/K2 actions, 33 SAFE features, and unchanged reward and termination contracts. The 96-hour compact-lineage quantity remains a raw extraction buffer, not an episode or policy horizon.

Respiratory action and state are separate frozen surfaces. Legacy action FiO2 accepts item IDs 223835, 226754, 227010, and 229280 without a unit gate, converts `[0,1]` to percent, and retains `[21,100]`. Repaired SAFE-state FiO2 excludes 223835, requires `%` for the other three IDs, and retains `[21,100]`. PEEP uses item IDs 220339 and 224700 in `[0,30]`. Same-bin PEEP and legacy FiO2 must both be directly observed.

Action cutpoints and K25 classes are frozen on the unfiltered KDD097 transition surface using positive observed train-role settings. Original target membership and missing-action filters are then applied; KDD151 repairs only the SAFE feature surface; KDD201 finally subsets rows and the already-frozen class array with the identical mask.

Expected clinical aggregate counts were not used by the constructor or fixtures. Credentialed reconstruction and parity remain for KDD220BR5.
