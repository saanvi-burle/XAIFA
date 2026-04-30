import csv
import json
import zipfile
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from app.core.config import settings
from app.schemas.datasets import DatasetInspectResponse, DatasetUploadResponse


SUPPORTED_DATASET_FORMATS = {"folder_zip", "csv_zip"}
SUPPORTED_DATASET_SUFFIXES = {".zip"}
SUPPORTED_LABEL_SUFFIXES = {".json", ".txt", ".csv"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class DatasetUploadError(ValueError):
    pass


async def save_uploaded_dataset(
    dataset_file: UploadFile,
    dataset_format: str,
    labels_file: UploadFile | None,
) -> DatasetUploadResponse:
    original_name = dataset_file.filename or "uploaded_dataset.zip"
    suffix = Path(original_name).suffix.lower()
    normalized_format = dataset_format.strip().lower()

    if normalized_format not in SUPPORTED_DATASET_FORMATS:
        raise DatasetUploadError("dataset_format must be either folder_zip or csv_zip.")

    if suffix not in SUPPORTED_DATASET_SUFFIXES:
        raise DatasetUploadError("Dataset upload must be a .zip file.")

    dataset_id = str(uuid4())
    dataset_dir = settings.uploads_dir / "datasets" / dataset_id
    dataset_dir.mkdir(parents=True, exist_ok=True)

    saved_path = dataset_dir / original_name
    saved_path.write_bytes(await dataset_file.read())

    labels_path = None
    labels_name = None
    if labels_file is not None and labels_file.filename:
        labels_name = labels_file.filename
        labels_suffix = Path(labels_name).suffix.lower()
        if labels_suffix not in SUPPORTED_LABEL_SUFFIXES:
            raise DatasetUploadError("Labels file must be .json, .txt, or .csv.")

        labels_path = dataset_dir / labels_name
        labels_path.write_bytes(await labels_file.read())

    metadata = {
        "dataset_id": dataset_id,
        "filename": original_name,
        "dataset_format": normalized_format,
        "labels_filename": labels_name,
        "saved_path": str(saved_path),
        "labels_path": str(labels_path) if labels_path else None,
    }
    metadata_path = dataset_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return DatasetUploadResponse(
        dataset_id=dataset_id,
        filename=original_name,
        dataset_format=normalized_format,
        labels_filename=labels_name,
        saved_path=str(saved_path),
        labels_path=str(labels_path) if labels_path else None,
        status="uploaded",
        next_step="Start an analysis run with the uploaded model and dataset.",
    )


def get_dataset_metadata(dataset_id: str) -> dict:
    metadata_path = settings.uploads_dir / "datasets" / dataset_id / "metadata.json"
    if not metadata_path.exists():
        raise DatasetUploadError(f"Dataset metadata not found for dataset_id: {dataset_id}")

    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _zip_image_entries(zf: zipfile.ZipFile) -> list[str]:
    return [
        info.filename
        for info in zf.infolist()
        if not info.is_dir() and Path(info.filename).suffix.lower() in IMAGE_SUFFIXES
    ]


def _inspect_folder_zip(zf: zipfile.ZipFile) -> tuple[int, list[str]]:
    images = _zip_image_entries(zf)
    classes = sorted(
        {
            Path(name).parts[0]
            for name in images
            if len(Path(name).parts) >= 2 and not Path(name).parts[0].startswith("__")
        }
    )
    return len(images), classes


def _inspect_csv_zip(zf: zipfile.ZipFile) -> tuple[int, list[str], int]:
    csv_names = [
        info.filename
        for info in zf.infolist()
        if not info.is_dir() and Path(info.filename).suffix.lower() == ".csv"
    ]
    if not csv_names:
        raise DatasetUploadError("csv_zip datasets must contain a CSV file.")

    with zf.open(csv_names[0]) as csv_file:
        rows = list(csv.DictReader(line.decode("utf-8") for line in csv_file))

    if not rows:
        raise DatasetUploadError("The dataset CSV file is empty.")

    label_column = "true_label" if "true_label" in rows[0] else "label"
    if label_column not in rows[0]:
        raise DatasetUploadError("The dataset CSV must contain true_label or label column.")

    classes = sorted({row[label_column] for row in rows if row.get(label_column)})
    return len(_zip_image_entries(zf)), classes, len(rows)


def inspect_uploaded_dataset(dataset_id: str) -> DatasetInspectResponse:
    metadata = get_dataset_metadata(dataset_id)
    dataset_path = Path(metadata["saved_path"])
    dataset_format = metadata["dataset_format"]

    if not dataset_path.exists():
        raise DatasetUploadError(f"Uploaded dataset file is missing: {dataset_path}")

    try:
        with zipfile.ZipFile(dataset_path) as zf:
            if dataset_format == "folder_zip":
                image_count, classes = _inspect_folder_zip(zf)
                csv_rows = None
            elif dataset_format == "csv_zip":
                image_count, classes, csv_rows = _inspect_csv_zip(zf)
            else:
                raise DatasetUploadError(f"Unsupported dataset format: {dataset_format}")
    except zipfile.BadZipFile as exc:
        raise DatasetUploadError("Dataset file is not a valid ZIP archive.") from exc

    if image_count == 0:
        raise DatasetUploadError("Dataset ZIP does not contain supported image files.")

    return DatasetInspectResponse(
        dataset_id=dataset_id,
        dataset_format=dataset_format,
        image_count=image_count,
        class_count=len(classes),
        classes=classes,
        csv_rows=csv_rows,
        status="inspected",
        message="Dataset ZIP structure is readable.",
    )
