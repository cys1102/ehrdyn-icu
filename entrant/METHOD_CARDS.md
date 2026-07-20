# Public entrant method cards

These reference implementations exercise the public interfaces. They
are not the restricted-data models used for manuscript evidence.

The KDD212 implementations below remain `smoke_only`. KDD215 adds the isolated
`history_softmax_reference` policy and `locf_gaussian_reference` transition
component. Neither is relabeled as an authoritative KDD199 learned method.

## Behavior cloning

`behavior_cloning` estimates a categorical action distribution from synthetic
logged context/action counts with one pseudocount on each supported action. It
observes emitted values, masks, recency, and the preceding action only.

## Tabular component model plus H4

`tabular_component_plus_h4` estimates context/action rewards and next contexts
from synthetic logged data. Its H4 categorical CEM planner uses three updates,
64 action-sequence candidates per update, eight elites, smoothing 0.2, a
support mask at every step, and receding-horizon execution of the first action.
It is a public interface baseline, not a clinical policy or an official
implementation of a named external simulator method.

## Repeated-dataset OPE smoke

The bounded scorer exposes IS, WIS, CWPDIS, DR, WDR, and FQE. Behavior and
return nuisance tables are refit for every independent synthetic dataset and
inside every bootstrap resample. The smoke is an implementation check; the
immutable KDD202B aggregate evidence remains separate.

## KDD215 history softmax conformance policy

`history_softmax_reference` consumes only values, masks, recency, past action,
and step through the isolated JSONL protocol. It returns a complete supported
categorical distribution. It is deterministic and has no fitting stage, so it
is a conformance policy rather than a reconstructed KDD199 behavior-cloning
checkpoint.

## KDD215 LOCF Gaussian component

`locf_gaussian_reference` emits a point mean and positive predictive scale from
observable history. It does not emit reward or termination components. H1/H4/H8
planning is therefore structural NA; the evaluator does not attach arbitrary
P/R/T heads.
