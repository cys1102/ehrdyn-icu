# KDD-RV successor credentialed adapter

This adapter verifies the exact RV01R/RV02R source hashes pinned in
`src/kdd2027_benchmark/rv/contracts/source_manifest.csv` before it can invoke the backend.
It does not copy clinical rows, arrays, predictions, checkpoints, or split
membership into this repository.

Source verification only:

```bash
python credentialed/successor/run_frozen_backend.py \
  --backend-root /path/to/world-ehr \
  --verify-only
```

Authorized construction and model stages must continue to use their frozen
ResearchForge configs. The wrapper deliberately has no default execution
stage. Sealed evaluation additionally requires the frozen test-opening
receipt and preserves the backend's single-opening guard.

This adapter makes the local release candidate executable against the pinned
backend checkout. It is not a claim of independent reproduction, final-paper
parity, clinical adjudication, or a self-contained public source archive.
