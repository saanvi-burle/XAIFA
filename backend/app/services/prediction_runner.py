import json
from datetime import datetime, timezone
from uuid import uuid4

import torch
from PIL import Image
from torchvision import transforms

from app.core.config import settings
from app.schemas.runs import (
    AnalysisRunResponse,
    AnalysisRunSummary,
    ClassAccuracyRecord,
    ConfusionMatrixRow,
    FailureRecord,
    PredictionRecord,
)
from app.services.dataset_loader import DatasetSample, load_dataset_samples
from app.services.model_loader import load_uploaded_model


def _preprocess(image: Image.Image, width: int, height: int, channels: int) -> torch.Tensor:
    mode = "L" if channels == 1 else "RGB"
    pipeline = transforms.Compose(
        [
            transforms.Resize((height, width)),
            transforms.Grayscale(num_output_channels=1) if channels == 1 else transforms.Lambda(lambda img: img.convert(mode)),
            transforms.ToTensor(),
        ]
    )
    return pipeline(image).unsqueeze(0)


def _predict_sample(model: torch.nn.Module, sample: DatasetSample, metadata: dict, labels: list[str]) -> PredictionRecord:
    input_width = int(metadata["input_width"])
    input_height = int(metadata["input_height"])
    channels = int(metadata["channels"])

    tensor = _preprocess(sample.image, input_width, input_height, channels)
    with torch.no_grad():
        output = model(tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        confidence, predicted_index = torch.max(probabilities, dim=0)

    predicted_idx = int(predicted_index.item())
    predicted_label = labels[predicted_idx] if predicted_idx < len(labels) else str(predicted_idx)
    confidence_value = float(confidence.item())

    return PredictionRecord(
        sample_id=sample.sample_id,
        source_path=sample.source_path,
        true_label=sample.true_label,
        predicted_label=predicted_label,
        predicted_index=predicted_idx,
        confidence=round(confidence_value, 6),
        is_correct=predicted_label == sample.true_label,
    )


def _build_confusion_matrix(predictions: list[PredictionRecord], labels: list[str]) -> list[ConfusionMatrixRow]:
    all_labels = sorted(set(labels) | {p.true_label for p in predictions} | {p.predicted_label for p in predictions})
    rows: list[ConfusionMatrixRow] = []
    for true_label in all_labels:
        counts = {label: 0 for label in all_labels}
        for prediction in predictions:
            if prediction.true_label == true_label:
                counts[prediction.predicted_label] = counts.get(prediction.predicted_label, 0) + 1
        rows.append(ConfusionMatrixRow(true_label=true_label, predicted_counts=counts))
    return rows


def _build_class_accuracy(predictions: list[PredictionRecord], labels: list[str]) -> list[ClassAccuracyRecord]:
    all_labels = sorted(set(labels) | {p.true_label for p in predictions})
    records: list[ClassAccuracyRecord] = []
    for label in all_labels:
        class_predictions = [prediction for prediction in predictions if prediction.true_label == label]
        total = len(class_predictions)
        correct = sum(1 for prediction in class_predictions if prediction.is_correct)
        failed = total - correct
        records.append(
            ClassAccuracyRecord(
                label=label,
                total=total,
                correct=correct,
                failed=failed,
                accuracy=round(correct / total, 6) if total else 0.0,
            )
        )
    return records


def _report_path(run_id: str):
    return settings.outputs_dir / "predictions" / run_id / "analysis_report.json"


def run_prediction_analysis(model_id: str, dataset_id: str, limit: int | None = None) -> AnalysisRunResponse:
    model, model_metadata = load_uploaded_model(model_id)
    samples, labels, dataset_metadata = load_dataset_samples(dataset_id, limit=limit)

    predictions = [_predict_sample(model, sample, model_metadata, labels) for sample in samples]
    failures = [
        FailureRecord(
            failure_id=f"failure_{index + 1}",
            sample_id=prediction.sample_id,
            source_path=prediction.source_path,
            true_label=prediction.true_label,
            predicted_label=prediction.predicted_label,
            confidence=prediction.confidence,
        )
        for index, prediction in enumerate(predictions)
        if not prediction.is_correct
    ]

    total = len(predictions)
    correct = sum(1 for prediction in predictions if prediction.is_correct)
    failed = total - correct
    accuracy = correct / total if total else 0.0

    run_id = str(uuid4())
    report_dir = _report_path(run_id).parent
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = _report_path(run_id)

    response = AnalysisRunResponse(
        run_id=run_id,
        model_id=model_id,
        dataset_id=dataset_id,
        total_samples=total,
        correct_predictions=correct,
        failed_predictions=failed,
        accuracy=round(accuracy, 6),
        labels=labels,
        confusion_matrix=_build_confusion_matrix(predictions, labels),
        class_accuracy=_build_class_accuracy(predictions, labels),
        predictions=predictions,
        failures=failures,
        report_path=str(report_path),
        created_at=datetime.now(timezone.utc).isoformat(),
        status="completed",
    )

    report_path.write_text(json.dumps(response.model_dump(), indent=2), encoding="utf-8")
    return response


def get_analysis_run(run_id: str) -> AnalysisRunResponse:
    path = _report_path(run_id)
    if not path.exists():
        raise ValueError(f"Analysis run not found: {run_id}")
    return AnalysisRunResponse.model_validate_json(path.read_text(encoding="utf-8"))


def list_analysis_runs() -> list[AnalysisRunSummary]:
    root = settings.outputs_dir / "predictions"
    if not root.exists():
        return []

    summaries: list[AnalysisRunSummary] = []
    for path in sorted(root.glob("*/analysis_report.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        data = json.loads(path.read_text(encoding="utf-8"))
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
