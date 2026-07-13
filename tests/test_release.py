from __future__ import annotations

import csv
import hashlib
import importlib.util
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
TASK = ROOT / "configs/tasks/kdd2027_sepsis_vasopressor_3bin.json"


class KDD089ReleaseTests(unittest.TestCase):
    def test_seven_task_configs_validate(self):
        result=_json(_run("validate-config","--config-dir",str(ROOT/"configs/tasks")).stdout)
        self.assertEqual(result["valid_configs"],7)

    def test_compact_and_rich_action_surfaces_are_separate(self):
        sepsis=json.loads(TASK.read_text(encoding="utf-8")); self.assertEqual(sepsis["action"]["action_count"],3)
        rich=json.loads((ROOT/"configs/rich_action/sepsis_original25_reference.json").read_text(encoding="utf-8")); self.assertEqual(rich["action"]["action_count"],25)
        respiratory=json.loads((ROOT/"configs/rich_action/respiratory_peep_fio2_observed_k25.json").read_text(encoding="utf-8"))
        self.assertFalse(respiratory["action"]["missing_is_action_zero"])
        self.assertEqual(respiratory["action"]["class_zero_semantics"],"lowest_observed_setting_pair")

    def test_core_aki_hf_are_runnable_and_rich_exclusions_remain_separate(self):
        self.assertEqual(json.loads((ROOT/"configs/tasks/kdd2027_aki_diuretic_rrt_factorized_3bin.json").read_text())["action"]["action_count"],9)
        self.assertEqual(json.loads((ROOT/"configs/tasks/kdd2027_hf_diuretic_binary.json").read_text())["action"]["action_count"],2)
        for name in ("aki_no_rich_action_exclusion","hf_no_rich_action_exclusion"):
            config=json.loads((ROOT/"configs/rich_action"/f"{name}.json").read_text(encoding="utf-8"))
            self.assertFalse(config["runnable"]); self.assertEqual(config["action"]["action_count"],0)

    def test_synthetic_evaluator_reports_uncertainty_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); fixture=root/"fixture.csv"; metrics=root/"metrics.json"
            _run("generate-fixture","--output",str(fixture),"--episodes","3","--seed","7")
            _run("evaluate","--fixture",str(fixture),"--task-config",str(TASK),"--output",str(metrics))
            values=json.loads(metrics.read_text(encoding="utf-8"))
        for key in ("gaussian_nll","gaussian_crps","cov80","cov90","width80","width90","interval_score90","uncertainty_absolute_error_spearman"):
            self.assertTrue(isinstance(values[key],int|float))

    def test_local_evaluator_does_not_export_episode_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); fixture=root/"fixture.csv"; local=root/"local.csv"; metrics=root/"metrics.json"
            _run("generate-fixture","--output",str(fixture),"--episodes","3","--seed","7")
            with fixture.open(newline="",encoding="utf-8") as source, local.open("w",newline="",encoding="utf-8") as target:
                reader=csv.DictReader(source); fields=["local_episode_key" if item=="synthetic_episode_key" else item for item in reader.fieldnames or ()]
                writer=csv.DictWriter(target,fieldnames=fields); writer.writeheader()
                for row in reader:
                    row["local_episode_key"]=row.pop("synthetic_episode_key"); writer.writerow(row)
            _run("evaluate-local","--predictions",str(local),"--task-config",str(TASK),"--output",str(metrics))
            values=json.loads(metrics.read_text(encoding="utf-8"))
        self.assertFalse(values["synthetic_fixture"])
        self.assertFalse(any("key" in name for name in values))

    def test_gate_layers_are_separate(self):
        relative=sepsis_relative_gate({"rmse":1.0,"cov90":.88,"behavior_nll":1.1,"low_support_rate":.05},{"rmse":1.0,"cov90":.9,"behavior_nll":1.0},{"rmse_ratio_max":1.1,"coverage_abs_difference_max":.05,"behavior_nll_ratio_max":1.2,"low_support_rate_max":.064405})
        absolute=absolute_policy_gate({"ess":10,"ess_fraction":.001,"wis":0,"wpdis":0,"fqe_finite":True,"clipping_stable":True,"denominator_ranking_stable":True,"reward_robust":True,"naive_policy_sanity":True,"ope_provenance_complete":False})
        self.assertTrue(relative["pass"]); self.assertFalse(absolute["pass"])

    def test_manifest_maps_seven_tasks_41_contracts_and_533_rows(self):
        result=_json(_run("validate-manifest","--task-manifest",str(ROOT/"contracts/paper_task_manifest.csv"),"--contract-manifest",str(ROOT/"contracts/paper_contract_manifest.csv"),"--evidence",str(ROOT/"evidence/core/contract_transition_leaderboard.csv")).stdout)
        self.assertEqual((result["paper_tasks"],result["primary_tasks"],result["contracts"],result["leaderboard_rows"]),(7,5,41,533))

    def test_manifest_rejects_task_action_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest=Path(directory)/"paper_task_manifest.csv"
            with (ROOT/"contracts/paper_task_manifest.csv").open(newline="",encoding="utf-8") as source:
                rows=list(csv.DictReader(source)); fields=list(rows[0])
            rows[0]["primary_action_view"]="wrong_action"
            for row in rows:
                row["config_path"]=str(ROOT/row["config_path"])
                row["clinical_packet_path"]=str(ROOT/row["clinical_packet_path"])
            with manifest.open("w",newline="",encoding="utf-8") as target:
                writer=csv.DictWriter(target,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
            result=_run("validate-manifest","--task-manifest",str(manifest),"--contract-manifest",str(ROOT/"contracts/paper_contract_manifest.csv"),"--evidence",str(ROOT/"evidence/core/contract_transition_leaderboard.csv"),expected=2)
        self.assertIn("action mismatch",result.stdout)

    def test_submission_template_validates(self):
        result=_json(_run("validate-submission","--submission",str(ROOT/"submission/leaderboard_submission_template.json"),"--config-dir",str(ROOT/"configs/tasks")).stdout)
        self.assertEqual(result["valid_rows"],1)

    def test_clinical_policy_submission_rejected(self):
        template=json.loads((ROOT/"submission/leaderboard_submission_template.json").read_text(encoding="utf-8")); template["rows"][0]["track"]="clinical_policy"
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"bad.json"; path.write_text(json.dumps(template),encoding="utf-8")
            result=_run("validate-submission","--submission",str(path),"--config-dir",str(ROOT/"configs/tasks"),expected=2)
        self.assertIn("Unsupported submission track",result.stdout)

    def test_submission_rejects_wrong_action_view(self):
        template=json.loads((ROOT/"submission/leaderboard_submission_template.json").read_text(encoding="utf-8")); template["rows"][0]["action_view"]="wrong"
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"bad.json"; path.write_text(json.dumps(template),encoding="utf-8")
            result=_run("validate-submission","--submission",str(path),"--config-dir",str(ROOT/"configs/tasks"),expected=2)
        self.assertIn("action view does not match",result.stdout)

    def test_split_is_deterministic(self):
        self.assertEqual(_run("split","--entity-key","1001").stdout,_run("split","--entity-key","1001").stdout)

    def test_privacy_scan_passes(self):
        result=_json(_run("scan-release","--root",str(ROOT)).stdout); self.assertTrue(result["pass"])

    def test_privacy_scan_rejects_identifier_header(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"unsafe.csv"; path.write_text("patient_id,metric\n1,0.1\n",encoding="utf-8")
            result=_run("scan-release","--root",directory,expected=2)
        self.assertIn("restricted_csv_header",result.stdout)

    def test_generated_bytecode_is_ignored_by_release_checks(self):
        cache=ROOT/"src/kdd2027_benchmark/__pycache__"; cache.mkdir(exist_ok=True)
        generated=cache/"generated.pyc"; generated.write_bytes(b"cache")
        try:
            self.assertTrue(_json(_run("scan-release","--root",str(ROOT)).stdout)["pass"])
        finally:
            generated.unlink(missing_ok=True)

    def test_manifest_checksums(self):
        for line in (ROOT/"MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
            digest,relative=line.split("  ",1); self.assertEqual(hashlib.sha256((ROOT/relative).read_bytes()).hexdigest(),digest)
        self.assertTrue(_json(_run("verify-checksums","--root",str(ROOT)).stdout)["pass"])

    def test_fixture_has_no_restricted_headers(self):
        with (ROOT/"fixtures/synthetic_small.csv").open(newline="",encoding="utf-8") as handle: headers=set(next(csv.reader(handle)))
        self.assertFalse(headers&{"subject_id","stay_id","hadm_id","patient_id","timestamp"})

    def test_credentialed_construction_is_public_but_outputs_are_restricted(self):
        for name in ("00_base_eligible_stays.sql","10_frozen_cohort_anchors.sql","20_four_hour_windows.sql","30_action_exposures.sql","40_observation_events.sql","45_static_context.sql"):
            self.assertTrue((ROOT/"credentialed/sql"/name).is_file())
        self.assertTrue((ROOT/"credentialed/build_local_contract.py").is_file())
        self.assertIn("restricted_outputs",(ROOT/"credentialed/build_local_contract.py").read_text(encoding="utf-8"))

    @unittest.skipUnless(importlib.util.find_spec("pandas"), "credentialed extra is not installed")
    def test_credentialed_action_encoders_preserve_frozen_levels(self):
        path=ROOT/"credentialed/build_local_contract.py"
        spec=importlib.util.spec_from_file_location("ehrdyn_credentialed_builder",path)
        if spec is None or spec.loader is None:
            self.fail("Could not load the credentialed builder module")
        module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        peep=module.pd.DataFrame({"peep":[float("nan"),4.0,5.5,7.0]})
        self.assertEqual(module._encode_action("kdd2027_respiratory_peep_5bin",peep).tolist(),[0,1,2,3])
        levels=[0.0,0.5,1.0]
        aki=module.pd.DataFrame({
            "diuretic":[value for value in levels for _ in levels],
            "rrt_crrt":levels*len(levels),
        })
        self.assertEqual(module._encode_action("kdd2027_aki_diuretic_rrt_factorized_3bin",aki).tolist(),list(range(9)))

    @unittest.skipUnless(importlib.util.find_spec("pandas"), "credentialed extra is not installed")
    def test_credentialed_builder_smoke(self):
        task_ids=[json.loads(path.read_text(encoding="utf-8"))["task_id"] for path in sorted((ROOT/"configs/tasks").glob("*.json"))]
        action_fields=["subject_id","stay_id","task_id","step_index","mortality_90d","fluid_bolus","vasopressor","diuretic","inotrope","rate_rhythm_control","anticoagulation","rrt_crrt","peep"]
        observation_fields=["subject_id","stay_id","task_id","step_index","feature_name","feature_value"]
        static_fields=["subject_id","stay_id","task_id","age","gender_male","readmission","elixhauser_score_proxy"]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); observations=root/"observations.csv"; actions=root/"actions.csv"; static=root/"static.csv"; expected=root/"expected.csv"; output=root/"output"
            with observations.open("w",newline="",encoding="utf-8") as handle:
                writer=csv.DictWriter(handle,fieldnames=observation_fields); writer.writeheader()
                for task_index,task_id in enumerate(task_ids):
                    stay=1000+task_index
                    for step in range(18):
                        for feature,value in (("heart_rate",80+step),("sbp",120),("mbp",75),("respiratory_rate",16),("temperature_c",37),("wbc",8),("pao2",90),("fio2",40)):
                            writer.writerow({"subject_id":2,"stay_id":stay,"task_id":task_id,"step_index":step,"feature_name":feature,"feature_value":value})
            with actions.open("w",newline="",encoding="utf-8") as handle:
                writer=csv.DictWriter(handle,fieldnames=action_fields); writer.writeheader()
                for task_index,task_id in enumerate(task_ids):
                    stay=1000+task_index
                    for step in range(18):
                        writer.writerow({"subject_id":2,"stay_id":stay,"task_id":task_id,"step_index":step,"mortality_90d":0,"fluid_bolus":step%2,"vasopressor":step%3,"diuretic":step%3,"inotrope":step%3,"rate_rhythm_control":step%3,"anticoagulation":step%3,"rrt_crrt":step%3,"peep":5+step%4})
            with static.open("w",newline="",encoding="utf-8") as handle:
                writer=csv.DictWriter(handle,fieldnames=static_fields); writer.writeheader()
                for task_index,task_id in enumerate(task_ids):
                    writer.writerow({"subject_id":2,"stay_id":1000+task_index,"task_id":task_id,"age":65,"gender_male":1,"readmission":0,"elixhauser_score_proxy":3})
            with expected.open("w",newline="",encoding="utf-8") as handle:
                fields=["benchmark_version","task_id","metric","expected_value","absolute_tolerance","validation_scope"]
                writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader()
                for task_id in task_ids:
                    writer.writerow({"benchmark_version":"smoke","task_id":task_id,"metric":"full_eligible_episodes","expected_value":1,"absolute_tolerance":0,"validation_scope":"synthetic_smoke"})
            result=subprocess.run([sys.executable,str(ROOT/"credentialed/build_local_contract.py"),"--observations",str(observations),"--actions",str(actions),"--static-context",str(static),"--expected",str(expected),"--output-dir",str(output)],cwd=ROOT,capture_output=True,text=True,check=False)
            self.assertEqual(result.returncode,0,msg=result.stdout+result.stderr)
            receipt=json.loads((output/"aggregate_receipt.json").read_text(encoding="utf-8"))
            self.assertTrue(receipt["restricted_outputs"]); self.assertTrue(receipt["parity_pass"])
            self.assertEqual(len(list(output.glob("*.restricted.npz"))),7)

    def test_policy_evidence_is_physically_quarantined(self):
        self.assertFalse((ROOT/"evidence/policy").exists())
        self.assertTrue((ROOT/"evidence/quarantine/policy/ope_wis_wpdis_clipping.csv").is_file())
        schema=json.loads((ROOT/"schemas/leaderboard_submission.schema.json").read_text(encoding="utf-8"))
        self.assertNotIn("policy_diagnostic",schema["properties"]["rows"]["items"]["properties"]["track"]["enum"])

    def test_clinical_review_is_explicitly_pending(self):
        with (ROOT/"clinical_review/core_review_status.csv").open(newline="",encoding="utf-8") as handle:
            rows=list(csv.DictReader(handle))
        self.assertEqual(len(rows),7)
        self.assertEqual({row["status"] for row in rows},{"pending"})
        self.assertTrue((ROOT/"clinical_review/reviewer_response_template.md").is_file())

    def test_compact_task_cards_have_current_policy_status(self):
        for path in sorted((ROOT/"task_cards").glob("kdd2027_*.md")):
            text=path.read_text(encoding="utf-8")
            self.assertIn("quarantined_not_public_leaderboard",text)
            self.assertNotIn("pending_KDD071",text)


def _run(*arguments:str,expected:int=0):
    env=dict(os.environ); env["PYTHONPATH"]=str(SOURCE); env["PYTHONDONTWRITEBYTECODE"]="1"
    result=subprocess.run([sys.executable,"-m","kdd2027_benchmark.cli",*arguments],cwd=ROOT,env=env,capture_output=True,text=True,check=False)
    if result.returncode!=expected: raise AssertionError(f"exit {result.returncode}, expected {expected}: {result.stdout} {result.stderr}")
    return result


def _json(text:str): return json.loads(text)


if __name__=="__main__": unittest.main()
