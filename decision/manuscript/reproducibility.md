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

The last command must return no match. Internal experiment IDs remain only in provenance artifacts. The builder regenerates task scale, full known-value matrices, and reference/task-matched OPE tuple surfaces from the immutable sources listed in `provenance-manifest.csv`.

For an unrestricted artifact smoke test that does not require MIMIC-IV, follow the quick start at <https://anonymous.4open.science/r/ehrdyn-icu-65FB>.
