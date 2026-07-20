# KDD214 result audit

Decision: `complete_schema_and_cross_python_exact_serialization`.

The transition false acceptance is closed: the previously accepted document lacked the schema-required submission identifier and used a malformed source commit. Every released schema validates as Draft 2020-12 and every accepting validator now validates the complete instance before semantic checks. All requested negative cases fail for their intended rule.

The canonical computed probe is byte-identical on Python 3.11, 3.12, and 3.13. Maximum raw semantic drift is below 1e-15 absolute; no runtime narrowing was required. Release artifact commit: `pending`; push status: `pending`.
