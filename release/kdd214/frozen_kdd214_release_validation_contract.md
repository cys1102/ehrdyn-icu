# Frozen KDD214 release validation contract

Immutable base: `f7d8123f6b2156afe0e754b76478361a7f1fc3af`. Implementation commit: `c42dd0d657efffda9023bc906c69ca6737055855`; tree: `6f65c8591dae0c987ddc64907bfe02be0a12bd41`.

All three released schemas are Draft 2020-12 documents and are the sole authority for required fields, types, enums, constants, patterns, bounds, and additional properties. Semantic validation follows schema validation and binds task/config identity, ordered features, action/timing views, allowed tracks, metric identities, and count consistency.

Computed JSON uses 12-decimal half-even quantization, compact sorted UTF-8 bytes, and LF. The frozen semantic tolerance is 5e-12 absolute, tighter than 1e-6 reporting precision. Exact byte equality is required across Python 3.11, 3.12, and 3.13.
