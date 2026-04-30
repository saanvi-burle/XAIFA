from pydantic import BaseModel


class PipelineStep(BaseModel):
    order: int
    name: str
    status: str


class AnalysisRunPlanResponse(BaseModel):
    steps: list[PipelineStep]


class AnalysisRunRequest(BaseModel):
    model_id: str
    dataset_id: str
    limit: int | None = None


class PredictionRecord(BaseModel):
    sample_id: str
    source_path: str
    true_label: str
    predicted_label: str
    predicted_index: int
    confidence: float
    is_correct: bool


class FailureRecord(BaseModel):
    failure_id: str
    sample_id: str
    source_path: str
    true_label: str
    predicted_label: str
    confidence: float


class ConfusionMatrixRow(BaseModel):
    true_label: str
    predicted_counts: dict[str, int]


class ClassAccuracyRecord(BaseModel):
    label: str
    total: int
    correct: int
    failed: int
    accuracy: float


class XAIExplanationRecord(BaseModel):
    method: str
    failure_id: str
    heatmap_path: str
    min: float
    max: float
    mean: float


class FailureWithXAI(BaseModel):
    failure_id: str
    sample_id: str
    source_path: str
    true_label: str
    predicted_label: str
    confidence: float
    explanations: dict[str, XAIExplanationRecord]


class AnalysisRunResponse(BaseModel):
    run_id: str
    model_id: str
    dataset_id: str
    total_samples: int
    correct_predictions: int
    failed_predictions: int
    accuracy: float
    labels: list[str]
    confusion_matrix: list[ConfusionMatrixRow]
    class_accuracy: list[ClassAccuracyRecord]
    predictions: list[PredictionRecord]
    failures: list[FailureRecord]
    report_path: str
    created_at: str
    status: str


class AnalysisRunSummary(BaseModel):
    run_id: str
    model_id: str
    dataset_id: str
    total_samples: int
    correct_predictions: int
    failed_predictions: int
    accuracy: float
    created_at: str
    status: str
