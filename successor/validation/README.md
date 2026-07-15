# Successor software validation

`source_metric_parity.json` compares the portable evaluator with the exact
frozen `kdd_rv02r.metrics` implementation on the same deterministic synthetic
arrays. Twenty-three point, Gaussian, interval, association, and paired
subject-cluster bootstrap values agreed within absolute tolerance `1e-12`; the
maximum observed difference was floating-point roundoff.

This receipt validates software parity for the tested synthetic surface. It is
not a clinical result, an independent credentialed reconstruction, or evidence
that an external prediction producer respected the recursive contract.
