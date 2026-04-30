import json
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from app.core.config import settings
from app.schemas.models import ModelUploadResponse


SUPPORTED_ARCHITECTURES = [
    "torchscript",
    "simple_cnn",
    "resnet18",
    "resnet50",
    "mobilenet_v2",
    "vgg16",
    "efficientnet_b0",
]

SUPPORTED_FORMATS = {".pt", ".pth"}


class ModelUploadError(ValueError):
    pass


def get_supported_architectures() -> list[str]:
    return SUPPORTED_ARCHITECTURES


async def save_uploaded_model(
    model_file: UploadFile,
    model_format: str,
    architecture: str | None,
    input_width: int,
    input_height: int,
    channels: int,
    num_classes: int,
) -> ModelUploadResponse:
    original_name = model_file.filename or "uploaded_model"
    suffix = Path(original_name).suffix.lower()
    normalized_format = model_format.strip().lower()

    if suffix not in SUPPORTED_FORMATS:
        raise ModelUploadError("Only .pt and .pth model files are supported in this version.")

    if normalized_format not in {"torchscript", "pytorch_state_dict"}:
        raise ModelUploadError("model_format must be either torchscript or pytorch_state_dict.")

    if normalized_format == "torchscript" and suffix != ".pt":
        raise ModelUploadError("TorchScript uploads must use a .pt file.")

    if normalized_format == "pytorch_state_dict" and suffix != ".pth":
        raise ModelUploadError("PyTorch state dict uploads must use a .pth file.")

    if normalized_format == "pytorch_state_dict" and not architecture:
        raise ModelUploadError("Architecture is required for .pth model uploads.")

    if input_width <= 0 or input_height <= 0 or channels <= 0 or num_classes <= 0:
        raise ModelUploadError("Input width, input height, channels, and num_classes must be positive.")

    model_id = str(uuid4())
    model_dir = settings.uploads_dir / "models" / model_id
    model_dir.mkdir(parents=True, exist_ok=True)

    saved_path = model_dir / original_name
    content = await model_file.read()
    saved_path.write_bytes(content)

    metadata = {
        "model_id": model_id,
        "filename": original_name,
        "model_format": normalized_format,
        "architecture": architecture,
        "input_width": input_width,
        "input_height": input_height,
        "channels": channels,
        "num_classes": num_classes,
        "saved_path": str(saved_path),
    }
    metadata_path = model_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return ModelUploadResponse(
        model_id=model_id,
        filename=original_name,
        model_format=normalized_format,
        architecture=architecture,
        input_width=input_width,
        input_height=input_height,
        channels=channels,
        num_classes=num_classes,
        saved_path=str(saved_path),
        metadata_path=str(metadata_path),
        status="uploaded",
        next_step="Upload a labelled test dataset and start analysis.",
    )


def get_model_metadata(model_id: str) -> dict:
    metadata_path = settings.uploads_dir / "models" / model_id / "metadata.json"
    if not metadata_path.exists():
        raise ModelUploadError(f"Model metadata not found for model_id: {model_id}")

    return json.loads(metadata_path.read_text(encoding="utf-8"))
