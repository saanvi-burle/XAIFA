from pathlib import Path

import torch

from app.ml.architectures import build_architecture
from app.schemas.models import ModelValidationResponse
from app.services.model_registry import get_model_metadata


class ModelLoadError(ValueError):
    pass


def _load_state_dict(path: Path) -> dict:
    loaded = torch.load(path, map_location="cpu")
    if isinstance(loaded, dict) and "state_dict" in loaded:
        return loaded["state_dict"]
    if isinstance(loaded, dict):
        return loaded
    raise ModelLoadError("The .pth file must contain a PyTorch state_dict.")


def load_uploaded_model(model_id: str) -> tuple[torch.nn.Module, dict]:
    metadata = get_model_metadata(model_id)
    model_path = Path(metadata["saved_path"])
    if not model_path.exists():
        raise ModelLoadError(f"Uploaded model file is missing: {model_path}")

    model_format = metadata["model_format"]
    num_classes = int(metadata["num_classes"])
    channels = int(metadata["channels"])

    try:
        if model_format == "torchscript":
            model = torch.jit.load(str(model_path), map_location="cpu")
        elif model_format == "pytorch_state_dict":
            architecture = metadata.get("architecture")
            if not architecture:
                raise ModelLoadError("Architecture is required for .pth validation.")
            model = build_architecture(architecture, num_classes=num_classes, channels=channels)
            model.load_state_dict(_load_state_dict(model_path), strict=True)
        else:
            raise ModelLoadError(f"Unsupported model format: {model_format}")

        model.eval()
        return model, metadata
    except ModelLoadError:
        raise
    except Exception as exc:
        raise ModelLoadError(f"Model loading failed: {exc}") from exc


def validate_uploaded_model(model_id: str) -> ModelValidationResponse:
    model, metadata = load_uploaded_model(model_id)
    model_format = metadata["model_format"]
    channels = int(metadata["channels"])
    input_height = int(metadata["input_height"])
    input_width = int(metadata["input_width"])
    input_shape = [1, channels, input_height, input_width]

    try:
        with torch.no_grad():
            dummy_input = torch.zeros(*input_shape)
            output = model(dummy_input)

        output_shape = list(output.shape) if hasattr(output, "shape") else None
    except Exception as exc:
        raise ModelLoadError(f"Model validation failed: {exc}") from exc

    return ModelValidationResponse(
        model_id=model_id,
        status="validated",
        architecture=metadata.get("architecture"),
        model_format=model_format,
        input_shape=input_shape,
        output_shape=output_shape,
        message="Model loaded successfully and completed a dummy forward pass.",
    )
