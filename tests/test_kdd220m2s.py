from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from kdd2027_benchmark.current_five_task.lineage_source_port import blood_culture_events


COLUMNS = (
    "micro_specimen_id", "subject_id", "hadm_id", "chartdate", "charttime",
    "spec_type_desc", "org_itemid", "org_name",
)


def _frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=COLUMNS)
    for column in ("chartdate", "charttime"):
        frame[column] = pd.to_datetime(frame[column])
    return frame


def _row(**updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "micro_specimen_id": 1,
        "subject_id": 11,
        "hadm_id": 21,
        "chartdate": "2020-01-01",
        "charttime": "2020-01-01 04:00:00",
        "spec_type_desc": "BLOOD CULTURE",
        "org_itemid": 123,
        "org_name": "organism-a",
    }
    row.update(updates)
    return row


class KDD220M2SNullSafeBloodCultureTests(unittest.TestCase):
    def test_valid_and_null_organism_preserves_valid_value(self) -> None:
        result = blood_culture_events(_frame([_row(org_name=None), _row(org_name="organism-a")]))
        self.assertEqual(result.loc[0, "org_name"], "organism-a")
        self.assertEqual(int(result.loc[0, "positive_culture"]), 1)

    def test_all_null_organism_remains_missing_and_negative(self) -> None:
        result = blood_culture_events(_frame([_row(org_name=None), _row(org_name=pd.NA)]))
        self.assertTrue(pd.isna(result.loc[0, "org_name"]))
        self.assertEqual(int(result.loc[0, "positive_culture"]), 0)

    def test_null_aware_max_is_deterministic_under_permutation(self) -> None:
        rows = [_row(org_name="organism-a"), _row(org_name="organism-z"), _row(org_name=None)]
        forward = blood_culture_events(_frame(rows))
        reverse = blood_culture_events(_frame(list(reversed(rows))))
        pd.testing.assert_frame_equal(forward, reverse)
        self.assertEqual(forward.loc[0, "org_name"], "organism-z")

    def test_chartdate_fallback_nonblood_and_excluded_item(self) -> None:
        rows = [
            _row(micro_specimen_id=1, charttime=None, chartdate="2020-01-03", org_name="organism-a"),
            _row(micro_specimen_id=2, spec_type_desc="URINE", org_name="organism-b"),
            _row(micro_specimen_id=3, org_itemid=90856, org_name="organism-c"),
        ]
        result = blood_culture_events(_frame(rows))
        self.assertEqual(result["micro_specimen_id"].astype(int).tolist(), [3, 1])
        by_specimen = result.set_index("micro_specimen_id")
        self.assertEqual(by_specimen.loc[1, "culture_time"], pd.Timestamp("2020-01-03"))
        self.assertEqual(by_specimen["positive_culture"].astype(int).to_dict(), {3: 0, 1: 1})

    def test_csv_and_gzip_typed_inputs_are_identical(self) -> None:
        rows = [_row(org_name=None), _row(org_name="organism-a", org_itemid="123")]
        frame = pd.DataFrame(rows, columns=COLUMNS)
        with tempfile.TemporaryDirectory() as directory:
            plain = Path(directory) / "events.csv"
            compressed = Path(directory) / "events.csv.gz"
            frame.to_csv(plain, index=False)
            frame.to_csv(compressed, index=False, compression="gzip")
            outputs = []
            for path in (plain, compressed):
                loaded = pd.read_csv(path, low_memory=False)
                for column in ("chartdate", "charttime"):
                    loaded[column] = pd.to_datetime(loaded[column])
                outputs.append(blood_culture_events(loaded))
        pd.testing.assert_frame_equal(outputs[0], outputs[1])

    def test_group_and_output_sort_are_deterministic(self) -> None:
        rows = [
            _row(micro_specimen_id=3, subject_id=12, charttime="2020-01-02 04:00:00"),
            _row(micro_specimen_id=1, subject_id=11, charttime="2020-01-02 04:00:00"),
            _row(micro_specimen_id=2, subject_id=11, charttime="2020-01-01 04:00:00"),
        ]
        expected = blood_culture_events(_frame(rows))
        for seed in range(5):
            shuffled = _frame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
            pd.testing.assert_frame_equal(expected, blood_culture_events(shuffled))
        self.assertEqual(expected["micro_specimen_id"].astype(int).tolist(), [2, 1, 3])


if __name__ == "__main__":
    unittest.main()
