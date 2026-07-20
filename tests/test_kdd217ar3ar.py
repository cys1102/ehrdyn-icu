from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from kdd2027_benchmark.current_five_task.authoritative_semantics import (
    clean_feature_values,
    compute_corrected_derived_features,
    gcs_total_rows,
    overlap_bin_amounts,
    overlap_bins,
)
from kdd2027_benchmark.current_five_task.contracts import (
    FEATURE_INDEX,
    FEATURE_NAMES,
    SAFE_FEATURE_INDICES,
    SAFE_FEATURE_NAMES,
)


class KDD217AR3ARTests(unittest.TestCase):
    def test_safe_interface_is_exact_frozen_subset(self) -> None:
        excluded = {"age", "gender_male", "weight", "readmission", "elixhauser_score_proxy", "sofa_proxy", "step_id"}
        self.assertEqual(tuple(FEATURE_NAMES[index] for index in SAFE_FEATURE_INDICES), SAFE_FEATURE_NAMES)
        self.assertFalse(excluded & set(SAFE_FEATURE_NAMES))
        self.assertIn("mechanical_ventilation", SAFE_FEATURE_NAMES)

    def test_gcs_direct_priority_and_complete_components(self) -> None:
        rows = pd.DataFrame([
            {"episode_idx": 0, "bin": 1, "itemid": 220739, "feature_value": 4.0},
            {"episode_idx": 0, "bin": 1, "itemid": 223900, "feature_value": 5.0},
            {"episode_idx": 0, "bin": 1, "itemid": 223901, "feature_value": 6.0},
            {"episode_idx": 0, "bin": 1, "itemid": 227013, "feature_value": 13.0},
            {"episode_idx": 0, "bin": 2, "itemid": 220739, "feature_value": 3.0},
            {"episode_idx": 0, "bin": 2, "itemid": 223900, "feature_value": 4.0},
            {"episode_idx": 0, "bin": 2, "itemid": 223901, "feature_value": 5.0},
        ])
        output = gcs_total_rows(rows).sort_values("bin")
        np.testing.assert_array_equal(output["feature_value"].to_numpy(), np.array([13.0, 12.0]))

    def test_interval_boundary_and_amount_allocation(self) -> None:
        origin = pd.Timestamp("2020-01-01 00:00:00")
        start, end = origin + pd.Timedelta(hours=3), origin + pd.Timedelta(hours=9)
        self.assertEqual(list(overlap_bins(start, end, origin, bin_hours=4, n_steps=4)), [0, 1, 2])
        pieces = overlap_bin_amounts(start, end, origin, bin_hours=4, n_steps=4, amount=600.0)
        self.assertEqual([index for index, _ in pieces], [0, 1, 2])
        np.testing.assert_allclose([amount for _, amount in pieces], [100.0, 400.0, 100.0], rtol=0, atol=1e-12)

    def test_corrected_derived_masks_and_values(self) -> None:
        values = np.full((1, 2, len(FEATURE_NAMES)), np.nan, np.float32)
        mask = np.zeros_like(values, bool)
        observed = {"heart_rate": [100, 80], "sbp": [100, 120], "temperature_c": [39, 37], "respiratory_rate": [22, 15], "wbc": [13, 8]}
        for name, sequence in observed.items():
            values[0, :, FEATURE_INDEX[name]] = sequence; mask[0, :, FEATURE_INDEX[name]] = True
        compute_corrected_derived_features(values, mask, np.zeros((1, 2), np.float32), feature_index=FEATURE_INDEX, bin_hours=4)
        np.testing.assert_allclose(values[0, :, FEATURE_INDEX["shock_index"]], [1.0, 2 / 3], rtol=1e-7)
        np.testing.assert_array_equal(mask[0, :, FEATURE_INDEX["sirs_proxy"]], [True, True])
        np.testing.assert_array_equal(values[0, :, FEATURE_INDEX["sirs_proxy"]], [4, 0])

    def test_clinical_ranges_fail_closed(self) -> None:
        values = clean_feature_values("lactate", np.array([-1, 0, 12, 31, np.nan]))
        np.testing.assert_array_equal(np.isfinite(values), [False, True, True, False, False])


if __name__ == "__main__":
    unittest.main()
