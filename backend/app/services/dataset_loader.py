import csv
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from app.services.dataset_registry import DatasetUploadError, IMAGE_SUFFIXES, get_dataset_metadata
from app.services.label_parser import parse_labels


@dataclass
class DatasetSample:
    sample_id: str
    image: Image.Image
    true_label: str
    source_path: str


def _read_image(zf: zipfile.ZipFile, name: str) -> Image.Image:
    with zf.open(name) as f:
        return Image.open(io.BytesIO(f.read())).convert("RGB")


def _load_folder_zip(zf: zipfile.ZipFile, limit: int | None) -> list[DatasetSample]:
    samples: list[DatasetSample] = []
    image_names = [
        info.filename
        for info in zf.infolist()
        if not info.is_dir() and Path(info.filename).suffix.lower() in IMAGE_SUFFIXES
    ]

    for name in sorted(image_names):
        parts = Path(name).parts
        if len(parts) < 2:
            continue
        samples.append(
            DatasetSample(
                sample_id=f"sample_{len(samples) + 1}",
                image=_read_image(zf, name),
                true_label=parts[0],
                source_path=name,
            )
        )
        if limit and len(samples) >= limit:
            break

    return samples


def _load_csv_zip(zf: zipfile.ZipFile, limit: int | None) -> list[DatasetSample]:
    csv_names = [
        info.filename
        for info in zf.infolist()
        if not info.is_dir() and Path(info.filename).suffix.lower() == ".csv"
    ]
    if not csv_names:
        raise DatasetUploadError("csv_zip datasets must contain a CSV file.")

    with zf.open(csv_names[0]) as csv_file:
        rows = list(csv.DictReader(line.decode("utf-8") for line in csv_file))

    samples: list[DatasetSample] = []
    label_column = "true_label" if rows and "true_label" in rows[0] else "label"
    for row in rows:
        image_path = row.get("image_path")
        true_label = row.get(label_column)
        if not image_path or not true_label:
            continue
        samples.append(
            DatasetSample(
                sample_id=f"sample_{len(samples) + 1}",
                image=_read_image(zf, image_path),
                true_label=true_label,
                source_path=image_path,
            )
        )
        if limit and len(samples) >= limit:
            break

    return samples


def load_dataset_samples(dataset_id: str, limit: int | None = None) -> tuple[list[DatasetSample], list[str] | None, dict]:
    metadata = get_dataset_metadata(dataset_id)
    dataset_path = Path(metadata["saved_path"])
    dataset_format = metadata["dataset_format"]
    labels = parse_labels(metadata.get("labels_path"))

    if not dataset_path.exists():
        raise DatasetUploadError(f"Uploaded dataset file is missing: {dataset_path}")

    try:
        with zipfile.ZipFile(dataset_path) as zf:
            if dataset_format == "folder_zip":
                samples = _load_folder_zip(zf, limit)
            elif dataset_format == "csv_zip":
                samples = _load_csv_zip(zf, limit)
            else:
                raise DatasetUploadError(f"Unsupported dataset format: {dataset_format}")
    except zipfile.BadZipFile as exc:
        raise DatasetUploadError("Dataset file is not a valid ZIP archive.") from exc

    if not samples:
        raise DatasetUploadError("No labelled image samples were found in the dataset.")

    if labels is None:
        labels = sorted({sample.true_label for sample in samples})

    return samples, labels, metadata
