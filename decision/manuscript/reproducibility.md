---
title: Reproducibility Commands
created: 2026-07-14
updated: 2026-07-15
type: writing
tags: [reproducibility, build, benchmark]
status: active
confidence: high
---

# Reproducibility commands

Run from the ResearchWiki repository after checking out the commits in `provenance-manifest.csv`. Use a Python 3.11 environment with the dependencies recorded in `environment-manifest.md`.

## Regenerate reader-facing aggregate tables and figures

```bash
python \
  scripts/build_kdd_syn_manuscript_package.py \
  --world-ehr ../world-ehr \
  --source kdd_benchmark_discovery/results/kdd_syn_final_benchmark_20260714_213941 \
  --output writing/kdd-benchmark-final
```

## Rebuild the current aggregate synthesis from frozen aggregate sources

```bash
cd ../world-ehr
python \
  -m kdd_benchmark_discovery.run_kdd_syn_final_benchmark \
  --config configs/kdd_syn_final_benchmark_v1.json \
  --output kdd_benchmark_discovery/results/kdd_syn_final_benchmark_reproduction
```

The reproduction directory is additive and must not overwrite the frozen source.

## Rebuild the primary heterogeneous exact-finite policy benchmark

```bash
cd ../world-ehr
python \
  -m kdd_benchmark_discovery.run_kdd107_heterogeneous_known_value \
  --config configs/kdd107_heterogeneous_known_value_v1.json \
  --output kdd_benchmark_discovery/results/reproduction_heterogeneous_known_value
```

This stage uses known constructed mechanisms and does not require patient-level EHR rows. It regenerates the full prior 4,080 policy--seed and 2,880 component-model--planner ledgers. The manuscript builder then applies the frozen scale status and exports 3,060/2,160 rows for the current primary view; heart-failure rows remain historical until its large-lineage rerun finishes.

## Rebuild repeated-dataset OPE and the diagnostic bridge

```bash
cd ../world-ehr
python \
  -m kdd_benchmark_discovery.run_kdd_ope_rd01_repeated_dataset \
  --config configs/kdd_ope_rd01_repeated_dataset_v1.json \
  --output kdd_benchmark_discovery/results/reproduction_repeated_dataset_ope

python \
  -m kdd_benchmark_discovery.run_kdd_bridge01_ehr_known_value \
  --config configs/kdd_bridge01_ehr_known_value_v1.json \
  --output kdd_benchmark_discovery/results/reproduction_ehr_known_value_bridge
```

The OPE stage regenerates independent logged datasets and refits behavior/nuisance surfaces inside each dataset. The bridge consumes aggregate EHR summaries and independently retrains known-value architectures; it does not reuse EHR weights.

## Compile the manuscript

```bash
cd ../ResearchWiki/writing/kdd-benchmark-final
tectonic manuscript.tex --keep-logs --keep-intermediates
cd ../..
python \
  scripts/build_kdd_syn_manuscript_package.py \
  --output writing/kdd-benchmark-final \
  --hash-only
```

## Validate

```bash
cd ../ResearchWiki
python3 scripts/wiki_lint.py --root .
git diff --check
rg -n '\\bKDD(?:-|[0-9])' writing/kdd-benchmark-final/manuscript.tex
```

The last command must return no match. Internal experiment IDs remain only in provenance artifacts. The builder regenerates task scale, full known-value matrices, repeated-dataset OPE surfaces, and bridge tables from the immutable sources listed in `provenance-manifest.csv`.

For an unrestricted artifact smoke test that does not require MIMIC-IV, follow the quick start at <https://anonymous.4open.science/r/ehrdyn-icu-65FB>.
