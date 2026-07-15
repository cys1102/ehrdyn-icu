# KDD-RV successor task cards

All five task identities are `defined_pending_clinical_review`. Each requires a
fully ICU-contained pre-action/action/target triple and uses the same frozen
subject-role hash. Detailed machine-readable contracts and immutable row hashes
are in `configs/tasks/`.

## Sepsis

Identity is time-stamped suspected infection plus an organ-dysfunction increase
established before forecasting. Diagnosis-only, future-maximum, and
medication-only fallbacks are excluded. The recorded action is vasopressor
intensity; it is not a treatment recommendation or policy label. Required
reviewers: critical care and infectious diseases.

## Respiratory support

Identity is the first structured respiratory-support event with observed
ventilator settings. It is not relabeled as exact invasive-ventilation onset.
The action is observed PEEP; missing PEEP is unavailable, not class zero.
Required reviewers: pulmonary and critical care.

## AKI

Identity is an earlier-value-only KDIGO-compatible creatinine change or first
recorded renal-replacement start. Admission diagnosis, BUN-only, future
baseline, and urine-output identity fallbacks are excluded. Actions are
factorized recorded diuretic and renal-replacement exposures. Required
reviewers: nephrology and critical care.

## AF/flutter

Identity requires a completed prior-encounter AF/flutter diagnosis plus a
current time-stamped rate/rhythm-management anchor. Current-admission diagnosis
or medication exposure alone cannot establish disease identity. The action is
binary recorded rate/rhythm-control exposure. Required reviewers: cardiology
or electrophysiology and critical care.

## Heart failure

Identity requires a completed prior-encounter heart-failure diagnosis plus a
current decongestion anchor. Current-admission diagnosis, medication exposure
alone, or generic volume-overload codes are excluded. The action is binary
recorded diuretic exposure. Required reviewers: heart-failure cardiology and
critical care.
