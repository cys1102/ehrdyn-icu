# Local Prediction Evaluation

`ehrdyn-icu evaluate-local` aggregates credentialed cell-level predictions. The
input remains local and may contain an opaque episode key; the output contains
only aggregate metrics and never returns that key.

Required CSV columns:

| Column | Meaning |
| --- | --- |
| `local_episode_key` | opaque local episode grouping key |
| `step_index` | zero-based transition/window index |
| `feature_name` | frozen feature name |
| `feature_group` | `static`, `dynamic`, or `sparse_lab` evaluation group |
| `current_value` | normalized state value at the forecast origin |
| `previous_value` | normalized value from the preceding window |
| `target_value` | normalized held-out observed target |
| `prediction_mean` | model predictive mean |
| `prediction_std` | positive predictive standard deviation |
| `action_index` | compact action index used by the forecast |
| `reward_component` | diagnostic reward component; zero is allowed |

Example:

```bash
ehrdyn-icu evaluate-local \
  --predictions /secure/ehrdyn/predictions.csv \
  --episode-key-column local_episode_key \
  --task-config configs/tasks/kdd2027_sepsis_vasopressor_3bin.json \
  --output /secure/ehrdyn/aggregate_metrics.json
```

The evaluator reports point error, NLL, CRPS, coverage, width, interval score,
and uncertainty-error association. It does not perform OPE and does not validate
causal or counterfactual predictions.
