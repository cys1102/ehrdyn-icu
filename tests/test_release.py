from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from kdd2027_benchmark.gates import absolute_policy_gate, sepsis_relative_gate


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
TASK = ROOT / "configs/tasks/sepsis_original25_reference.json"


class KDD089ReleaseTests(unittest.TestCase):
    def test_seven_task_configs_validate(self):
        result=_json(_run("validate-config","--config-dir",str(ROOT/"configs/tasks")).stdout)
        self.assertEqual(result["valid_configs"],7)

    def test_exact_sepsis_k25_and_respiratory_semantics(self):
        sepsis=json.loads(TASK.read_text(encoding="utf-8")); self.assertEqual(sepsis["action"]["action_count"],25)
        respiratory=json.loads((ROOT/"configs/tasks/respiratory_peep_fio2_observed_k25.json").read_text(encoding="utf-8"))
        self.assertFalse(respiratory["action"]["missing_is_action_zero"])
        self.assertEqual(respiratory["action"]["class_zero_semantics"],"lowest_observed_setting_pair")

    def test_aki_hf_are_nonrunnable_exclusions(self):
        for name in ("aki_no_rich_action_exclusion","hf_no_rich_action_exclusion"):
            config=json.loads((ROOT/"configs/tasks"/f"{name}.json").read_text(encoding="utf-8"))
            self.assertFalse(config["runnable"]); self.assertEqual(config["action"]["action_count"],0)

    def test_synthetic_evaluator_reports_uncertainty_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); fixture=root/"fixture.csv"; metrics=root/"metrics.json"
            _run("generate-fixture","--output",str(fixture),"--episodes","3","--seed","7")
            _run("evaluate","--fixture",str(fixture),"--task-config",str(TASK),"--output",str(metrics))
            values=json.loads(metrics.read_text(encoding="utf-8"))
        for key in ("gaussian_nll","gaussian_crps","cov80","cov90","width80","width90","interval_score90","uncertainty_absolute_error_spearman"):
            self.assertTrue(isinstance(values[key],int|float))

    def test_gate_layers_are_separate(self):
        relative=sepsis_relative_gate({"rmse":1.0,"cov90":.88,"behavior_nll":1.1,"low_support_rate":.05},{"rmse":1.0,"cov90":.9,"behavior_nll":1.0},{"rmse_ratio_max":1.1,"coverage_abs_difference_max":.05,"behavior_nll_ratio_max":1.2,"low_support_rate_max":.064405})
        absolute=absolute_policy_gate({"ess":10,"ess_fraction":.001,"wis":0,"wpdis":0,"cwpdis":0,"fqe_finite":True,"clipping_stable":True,"denominator_ranking_stable":True,"reward_robust":True,"naive_policy_sanity":True})
        self.assertTrue(relative["pass"]); self.assertFalse(absolute["pass"])

    def test_submission_template_validates(self):
        result=_json(_run("validate-submission","--submission",str(ROOT/"submission/leaderboard_submission_template.json"),"--config-dir",str(ROOT/"configs/tasks")).stdout)
        self.assertEqual(result["valid_rows"],1)

    def test_clinical_policy_submission_rejected(self):
        template=json.loads((ROOT/"submission/leaderboard_submission_template.json").read_text(encoding="utf-8")); template["rows"][0]["track"]="clinical_policy"
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"bad.json"; path.write_text(json.dumps(template),encoding="utf-8")
            result=_run("validate-submission","--submission",str(path),"--config-dir",str(ROOT/"configs/tasks"),expected=2)
        self.assertIn("Unsupported submission track",result.stdout)

    def test_split_is_deterministic(self):
        self.assertEqual(_run("split","--entity-key","1001").stdout,_run("split","--entity-key","1001").stdout)

    def test_privacy_scan_passes(self):
        result=_json(_run("scan-release","--root",str(ROOT)).stdout); self.assertTrue(result["pass"])

    def test_privacy_scan_rejects_identifier_header(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"unsafe.csv"; path.write_text("patient_id,metric\n1,0.1\n",encoding="utf-8")
            result=_run("scan-release","--root",directory,expected=2)
        self.assertIn("restricted_csv_header",result.stdout)

    def test_manifest_checksums(self):
        for line in (ROOT/"MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
            digest,relative=line.split("  ",1); self.assertEqual(hashlib.sha256((ROOT/relative).read_bytes()).hexdigest(),digest)
        self.assertTrue(_json(_run("verify-checksums","--root",str(ROOT)).stdout)["pass"])

    def test_fixture_has_no_restricted_headers(self):
        with (ROOT/"fixtures/synthetic_small.csv").open(newline="",encoding="utf-8") as handle: headers=set(next(csv.reader(handle)))
        self.assertFalse(headers&{"subject_id","stay_id","hadm_id","patient_id","timestamp"})


def _run(*arguments:str,expected:int=0):
    env=dict(os.environ); env["PYTHONPATH"]=str(SOURCE); env["PYTHONDONTWRITEBYTECODE"]="1"
    result=subprocess.run([sys.executable,"-m","kdd2027_benchmark.cli",*arguments],cwd=ROOT,env=env,capture_output=True,text=True,check=False)
    if result.returncode!=expected: raise AssertionError(f"exit {result.returncode}, expected {expected}: {result.stdout} {result.stderr}")
    return result


def _json(text:str): return json.loads(text)


if __name__=="__main__": unittest.main()
