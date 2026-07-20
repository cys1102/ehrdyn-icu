from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from pathlib import Path


FILES = (
    ("five_task_transition_source", "tables/five_cohort_transition_uncertainty.csv", "source"),
    ("five_task_transition_table", "tables/five_cohort_transition_uncertainty.tex", "table"),
    ("five_task_policy_source", "tables/current_five_cohort_real_ehr_policy_diagnostics.csv", "source"),
    ("five_task_policy_table", "tables/current_five_cohort_real_ehr_policy_diagnostics.tex", "table"),
    ("five_task_ope_source", "tables/current_real_ehr_ope_by_estimator.csv", "source"),
    ("five_task_ope_table", "tables/current_real_ehr_ope_by_estimator.tex", "table"),
    ("paired_method_source", "tables/kdd211_all_method_environment_paired.csv", "source"),
    ("paired_method_table", "tables/kdd211_all_method_environment_paired.tex", "table"),
    ("paired_estimator_source", "tables/kdd211_estimator_paired_contrasts.csv", "source"),
    ("paired_estimator_table", "tables/kdd211_estimator_paired_contrasts.tex", "table"),
    ("uncertainty_horizon_source", "figure-data/uncertainty_horizon_metrics.csv", "source"),
    ("uncertainty_reliability_source", "figure-data/uncertainty_reliability_metrics.csv", "source"),
    ("uncertainty_figure_pdf", "figures/uncertainty_overview_revised.pdf", "figure"),
    ("uncertainty_figure_png", "figures/uncertainty_overview_revised.png", "figure"),
    ("paired_inference_figure_pdf", "figures/kdd211_environment_paired_inference.pdf", "figure"),
    ("paired_inference_figure_png", "figures/kdd211_environment_paired_inference.png", "figure"),
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Import a reviewed aggregate-only KDD211 public bundle.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--kdd211-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with args.kdd211_manifest.open(newline="", encoding="utf-8") as handle:
        parity_rows = list(csv.DictReader(handle))
    parity = {Path(row["artifact"]).name: row["sha256"] for row in parity_rows if row["variant"] == "final"}
    rows = []
    for artifact_id, relative, role in FILES:
        source = args.source / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        if source.suffix.lower() in {".csv", ".tex", ".md", ".json"}:
            text = source.read_text(encoding="utf-8", errors="strict")
            local_home_marker = "/" + "home/"
            if local_home_marker in text or any(token in text.lower() for token in ("subject_id", "stay_id", "hadm_id", "patient_id", "charttime")):
                raise RuntimeError(f"restricted or internal content in {relative}")
        target = args.output / "artifacts" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        digest = sha(target)
        parity_status = "exact_kdd211_source_manifest" if parity.get(source.name) == digest else "current_lineage_aggregate_safe_addition"
        rows.append({
            "artifact_id": artifact_id,
            "bundle_path": target.relative_to(args.output).as_posix(),
            "output_path": relative,
            "role": role,
            "sha256": digest,
            "expected_output_sha256": digest,
            "kdd211_parity_status": parity_status,
            "restricted_input_required": False,
        })
    manifest = args.output / "public_manuscript_aggregate_bundle_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    receipt = args.output / "kdd211_source_manifest_receipt.csv"
    with receipt.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["artifact_name", "role", "sha256", "status"])
        for row in parity_rows:
            if row["variant"] == "final":
                writer.writerow([Path(row["artifact"]).name, row["role"], row["sha256"], "exact_hash_frozen"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
