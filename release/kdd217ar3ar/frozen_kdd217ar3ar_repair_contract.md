# KDD217AR3AR frozen bounded repair contract

This additive stage preserves `release/kdd217ar3a/` byte-for-byte and uses only
de-identified synthetic MIMIC-shaped flat files. No MIMIC data, credentials,
patient rows, result directories, split manifests, checkpoints, tensors, or
author aggregate outputs are inputs.

The hard differential requires exact masks, roles, actions, reward masks,
termination, ordering, and counts, plus `rtol=1e-12` and `atol=1e-12` for
observed floating cells. A missing source-closed parameter is a failure, not an
invitation to infer it from a historical result receipt.

This is the final bounded source-port repair. Any incomplete five-task
end-to-end differential terminates the public credentialed-reconstruction
branch without commit, push, or KDD217AR3B authorization.
