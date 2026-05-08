import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.core.config import settings
from app.schemas.runs import (
    AnalysisRunResponse,
    AnalysisRunSummary,
)


# =========================
# SCRIPT EXECUTION
# =========================
def run_external_pipeline_scripts():

    print("\n================ XAIFA PIPELINE START ================\n")

    project_root = settings.project_root

    scripts = [
        "scripts/train_model.py",
        "scripts/extract_failures.py",
        "scripts/run_experiments.py",
    ]

    for script in scripts:

        script_path = project_root / script

        print(f"\nRUNNING: {script}")

        if not script_path.exists():

            print(f"FILE NOT FOUND: {script_path}")

            continue

        try:

            result = subprocess.run(
                ["python", str(script_path)],
                capture_output=True,
                text=True,
                cwd=script_path.parent,
            )

            print(f"\n===== OUTPUT: {script} =====\n")

            print(result.stdout)

            if result.stderr:

                print(f"\n===== ERRORS: {script} =====\n")

                print(result.stderr)

            print(f"\n✅ FINISHED: {script}")

        except Exception as e:

            print(f"\n FAILED: {script}")

            print(e)

    print("\n================ XAIFA PIPELINE END ================\n")


# =========================
# MAIN ANALYSIS PIPELINE
# =========================
def run_prediction_analysis(
    model_id: str,
    dataset_id: str,
    limit: int | None = None,
):

    # ===================================
    # RUN ACTUAL XAIFA PIPELINE
    # ===================================
    run_external_pipeline_scripts()

    run_id = str(uuid4())

    # ===================================
    # RESPONSE
    # ===================================
    response = AnalysisRunResponse(
        run_id=run_id,
        model_id=model_id,
        dataset_id=dataset_id,
        total_samples=0,
        correct_predictions=0,
        failed_predictions=0,
        accuracy=0.0,
        labels=[],
        confusion_matrix=[],
        class_accuracy=[],
        predictions=[],
        failures=[],
        report_path="",
        created_at=datetime.now(timezone.utc).isoformat(),
        status="completed",
    )

    return response


# =========================
# REPORT PATH
# =========================
def _report_path(run_id: str):

    return (
        settings.outputs_dir
        / "predictions"
        / run_id
        / "analysis_report.json"
    )


# =========================
# GET RUN
# =========================
def get_analysis_run(run_id: str):

    path = _report_path(run_id)

    if not path.exists():
        raise ValueError(f"Analysis run not found: {run_id}")

    return AnalysisRunResponse.model_validate_json(
        path.read_text(encoding="utf-8")
    )


# =========================
# LIST RUNS
# =========================
def list_analysis_runs():

    root = settings.outputs_dir / "predictions"

    if not root.exists():
        return []

    summaries = []

    for path in sorted(
        root.glob("*/analysis_report.json"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    ):

        data = json.loads(
            path.read_text(encoding="utf-8")
        )

        summaries.append(
            AnalysisRunSummary(
                run_id=data["run_id"],
                model_id=data["model_id"],
                dataset_id=data["dataset_id"],
                total_samples=data["total_samples"],
                correct_predictions=data["correct_predictions"],
                failed_predictions=data["failed_predictions"],
                accuracy=data["accuracy"],
                created_at=data["created_at"],
                status=data["status"],
            )
        )

    return summaries