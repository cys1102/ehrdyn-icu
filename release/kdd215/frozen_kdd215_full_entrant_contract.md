# Frozen KDD215 full entrant contract

KDD215 starts at public KDD214 commit
`52127b903550012b281cc4a1dd4f8666e9552e41`, whose decision is
`complete_schema_and_cross_python_exact_serialization`. This is a new additive
full-protocol workflow. The earlier built-in `pomdp-smoke` and `ope-smoke`
remain `smoke_only` and are not interpreted as KDD198/KDD199/KDD202B parity.

The public generator is reconstructed from the 40 aggregate-safe serialized
KDD198-v2 environments. The five profiles and eight seeds are fixed before any
entrant runs. Training, validation, final exogenous, stochastic-policy, logged
dataset, and bootstrap namespaces are separate. Entrants see observable values,
masks, recency, past actions, and public task metadata only.

Policy evaluation starts at 4,096 episodes per environment and follows the
frozen doubling ladder through 65,536 until normalized return SE is at most
0.0025. Common exogenous streams are shared by the entrant and preregistered
controls. Environment is the inferential unit; profiles receive equal weight.

OPE contains 5 profiles x 8 environments x 8 independent logged datasets = 320
datasets, with 256 episodes per dataset. Each of four frozen contracts runs IS,
WIS, CWPDIS, DR, WDR, and FQE with 500 episode-bootstrap nuisance refits.
Failures, nonfinite values, low ESS, and unsupported mass are retained.

The source hashes, thresholds, schemas, resources, seed roles, and estimator
ordering are frozen before public entrant results. No EHR rows, patient
artifacts, restricted checkpoints, or latent simulator traces are released.
