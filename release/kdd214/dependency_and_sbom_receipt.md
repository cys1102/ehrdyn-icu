# Dependency and SBOM receipt

The runtime dependency is pinned as `jsonschema==4.26.0`; transitive dependencies are locked in `uv.lock` for Python 3.11--3.13. Clean installation must resolve dependencies; the obsolete `--no-deps` instruction was removed.

- Lock SHA-256: `7cf2cc0730da7437507a205ee77aa8679b1bce5a867f8ba545f80e1f740cd026`
- Deterministic CycloneDX 1.5 SBOM SHA-256: `97cf18abb67264cf74f3bca4ff5a0816f045eaf45e517627509b73fbdb6e8318`
- SBOM component count: 6
- SBOM generator: uv 0.10.11 with timestamp and random serial removed; lock hash embedded
