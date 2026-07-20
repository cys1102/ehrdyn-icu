# KDD220B frozen author credentialed reconstruction contract

KDD220B is bound to public KDD220A commit
`9db6dc1fc2c3b68645fe04934379e4e4b5f3f1cf`. The frozen runtime config SHA-256
is `4f776cb6d10d4809cf94f3f6686d9ff35eae808d72012f9f17aa9cc54b2c5b95`
and the aggregate receipt schema SHA-256 is
`f6d8646985cf2cf23859e9227b6038f0dbe453a5ebe3c4998c69702250131cd8`.

Construction was required to finish before any historical aggregate/interface
reference could be opened. The runtime stopped during ICU-stay schema/time
validation, before cohort construction. Consequently no reference evidence was
opened, no parity comparison was attempted, no model was trained, and no
scientific result was rerun.

The first local source root contained both official CSV and CSV.GZ encodings.
The documented exactly-one-encoding contract was satisfied using a read-only
local view containing only official CSV.GZ files. The second run then stopped
because the public `load_core` implementation rejects the full input when any
ICU stay has missing or nonpositive time order before applying the frozen
authoritative eligibility filter. No code, config, source row, or output was
changed after observing the failure.
