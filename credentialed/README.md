# Credentialed MIMIC-IV Construction

This directory contains the public construction path for the seven compact
paper tasks. It contains code and aggregate parity targets, never MIMIC-IV rows.

## Inputs

The SQL expects MIMIC-IV v3.1 in PostgreSQL with the official `mimiciv_derived`
concepts installed. Run the SQL files in numeric order. The views contain
restricted join keys and must remain inside the authorized environment.

## Private Build

Export `observation_events`, `action_exposures`, and `static_context` to secure
local CSV files. Run `build_local_contract.py` as documented in
`../MIMIC_ACCESS.md`. The builder applies the frozen subject split, past-only
forward filling initialized by training medians, training-only scaling,
mask/recency channels, four derived physiology features, and compact action
encoders. It emits restricted arrays plus an aggregate parity receipt.

## Required Parity

`expected/frozen_task_aggregate_checks.csv` freezes episode, window, subject,
observation-fraction, mortality, and occupied action-count summaries. The
aggregate receipt also reports every action-class count and whether observed
and configured cardinalities match. A local build is not the paper contract
unless every exact-tolerance check passes. This gate checks
construction consistency; it does not establish clinical validity, causal
identification, policy evaluability, or deployment safety.

Medication amounts are prorated by event-window overlap; RRT/CRRT is represented
as the fraction of each four-hour window covered by a structured procedure
interval. Drug-family values remain recorded-exposure proxies rather than
cross-drug dose-equivalent treatment intensities. Their units and item mappings
require credentialed verification and independent clinical adjudication.

## Privacy

Do not commit local exports, arrays, preprocessing statistics, split membership,
or receipts that have not passed aggregate privacy review. The public release
contains only code, task definitions, and aggregate targets.
