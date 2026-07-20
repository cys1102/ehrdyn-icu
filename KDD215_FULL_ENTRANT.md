# KDD215 full entrant workflow

The KDD215 surface is additive to the bounded KDD212 smoke. It reconstructs the
aggregate-safe KDD198-v2 generator contract, evaluates policies on 40 public
environments with KDD199 common random numbers, and runs the KDD202B nesting of
320 independent logged datasets.

Full entrant execution currently requires Linux with `bubblewrap` (`bwrap`) on
`PATH`; the evaluator fails closed rather than importing entrant code when the
sandbox is unavailable. Python 3.11--3.13, `jsonschema==4.26.0`, and
`numpy==2.3.3` are pinned by the project and `uv.lock`.

```bash
MANIFEST=configs/full_benchmark/kdd198_v2_generator_contract.json

ehrdyn-icu generate-full-suite --manifest "$MANIFEST" --output /tmp/suite.csv
ehrdyn-icu validate-entrant --entrant policy_entrant_example/entrant.json --manifest "$MANIFEST"
ehrdyn-icu train-entrant --entrant policy_entrant_example/entrant.json --manifest "$MANIFEST" --output /tmp/train-receipt.json
ehrdyn-icu evaluate-transition --entrant component_entrant_example/entrant.json --manifest "$MANIFEST" --output /tmp/component.csv
ehrdyn-icu evaluate-policy-return --entrant policy_entrant_example/entrant.json --manifest "$MANIFEST" --output /tmp/direct.csv --contrasts /tmp/contrasts.csv
ehrdyn-icu evaluate-policy-ope --entrant policy_entrant_example/entrant.json --manifest "$MANIFEST" --direct-returns /tmp/direct.csv --workers 8 --output /tmp/ope.csv
ehrdyn-icu summarize-submission --input /tmp/ope.csv --output /tmp/ope-summary.csv
```

`train-entrant` freezes and validates the public train/validation/final role
boundary. An entrant's own documented command performs fitting; the evaluator
does not import or mutate entrant code. The bundled policy example is a
deterministic no-fit conformance entrant. The component example is
transition-only and therefore has structural NA for H1/H4/H8 P/R/T planning.

All 40 mechanisms and seeds are public development assets. There is no hidden
test service. No EHR row, patient-derived checkpoint, latent simulator state,
or privileged reference input is exposed.
