# KDD235B result audit

## Outcome

The full public KDD235A interface is executable for an entrant-owned recurrent
Gaussian model across five profiles and eight environment seeds per profile.
The demonstration produced 40 checkpoint receipts, 440 horizon metrics, 40
full-episode direct-return rows, and 240 summaries covering six OPE estimators
over 64 independently generated 256-episode datasets per environment.

## Identity and isolation

- Public base: `4739d59392a660cf215c29ebca02fb2f52cd7804`.
- Frozen pre-final implementation commit: `5e6cbd7`.
- Entrant source imports no benchmark transition class or private artifact.
- Fit/load receives train/validation role labels with final access explicitly
  false.
- Direct-return and OPE streams are disjoint frozen namespaces.
- The example remains excluded from the scientific leaderboard.

## Verification

- KDD235A/KDD235B focused tests: 14/14 pass.
- Full inventory: 40/40 environments complete.
- Forecasting: 440/440 horizon rows finite.
- Direct return: 40/40 rows finite with normalized supported probabilities.
- Repeated OPE: 240/240 environment-estimator summaries finite.
- Deterministic overlap: all 19 sepsis/171901 aggregate rows identical.
- Forbidden-import and aggregate privacy scans pass.

Some importance-weighted estimates are extremely unstable. They remain in the
release as a truthful interface result and are not evidence that an estimator
is reliable.

## Claim boundary

This result demonstrates constructed-benchmark interface usability. It is not
independent EHR reconstruction, disease-faithful simulation, clinical utility,
treatment-effect evidence, policy deployment evidence, or scientific
leaderboard eligibility for the demonstration entrant.
