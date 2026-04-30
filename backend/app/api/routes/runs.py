from fastapi import APIRouter, HTTPException

from app.schemas.runs import (
    AnalysisRunPlanResponse,
    AnalysisRunRequest,
    AnalysisRunResponse,
    AnalysisRunSummary,
    FailureRecord,
    PipelineStep,
)
from app.services.dataset_registry import DatasetUploadError
from app.services.model_loader import ModelLoadError
from app.services.model_registry import ModelUploadError
from app.services.prediction_runner import get_analysis_run, list_analysis_runs, run_prediction_analysis


router = APIRouter()


@router.get("/pipeline", response_model=AnalysisRunPlanResponse)
def analysis_pipeline() -> AnalysisRunPlanResponse:
    return AnalysisRunPlanResponse(
        steps=[
            PipelineStep(order=1, name="Load model", status="planned"),
            PipelineStep(order=2, name="Preprocess dataset", status="planned"),
            PipelineStep(order=3, name="Run predictions", status="planned"),
            PipelineStep(order=4, name="Collect failed cases", status="planned"),
            PipelineStep(order=5, name="Generate Grad-CAM, SHAP, and LIME", status="planned"),
            PipelineStep(order=6, name="Create fusion explanations", status="planned"),
            PipelineStep(order=7, name="Score XAI methods", status="planned"),
            PipelineStep(order=8, name="Cluster failures", status="planned"),
            PipelineStep(order=9, name="Generate recommendations", status="planned"),
        ]
    )


@router.get("", response_model=list[AnalysisRunSummary])
def list_runs() -> list[AnalysisRunSummary]:
    return list_analysis_runs()


@router.post("/analyze", response_model=AnalysisRunResponse)
def analyze(request: AnalysisRunRequest) -> AnalysisRunResponse:
    try:
        return run_prediction_analysis(
            model_id=request.model_id,
            dataset_id=request.dataset_id,
            limit=request.limit,
        )
    except (ModelUploadError, ModelLoadError, DatasetUploadError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{run_id}", response_model=AnalysisRunResponse)
def get_run(run_id: str) -> AnalysisRunResponse:
    try:
        return get_analysis_run(run_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{run_id}/failures", response_model=list[FailureRecord])
def get_run_failures(run_id: str) -> list[FailureRecord]:
    try:
        return get_analysis_run(run_id).failures
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
