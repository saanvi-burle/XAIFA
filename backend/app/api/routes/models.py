from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.schemas.models import ModelUploadResponse, ModelValidationResponse, SupportedModelResponse
from app.services.model_loader import ModelLoadError, validate_uploaded_model
from app.services.model_registry import ModelUploadError, get_supported_architectures, save_uploaded_model


router = APIRouter()


@router.get("/supported", response_model=SupportedModelResponse)
def supported_models() -> SupportedModelResponse:
    return SupportedModelResponse(architectures=get_supported_architectures())


@router.post("/upload", response_model=ModelUploadResponse)
async def upload_model(
    model_file: UploadFile = File(...),
    model_format: str = Form(...),
    architecture: str | None = Form(default=None),
    input_width: int = Form(...),
    input_height: int = Form(...),
    channels: int = Form(...),
    num_classes: int = Form(...),
) -> ModelUploadResponse:
    print("[DEBUG] upload_model endpoint called")
    print(f"model_file: {getattr(model_file, 'filename', None)}")
    print(f"model_format: {model_format}, architecture: {architecture}, input_width: {input_width}, input_height: {input_height}, channels: {channels}, num_classes: {num_classes}")
    try:
        return await save_uploaded_model(
            model_file=model_file,
            model_format=model_format,
            architecture=architecture,
            input_width=input_width,
            input_height=input_height,
            channels=channels,
            num_classes=num_classes,
        )
    except ModelUploadError as exc:
        print(f"ModelUploadError: {exc}")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        import traceback
        print("Unexpected error in upload_model:", exc)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error. See backend logs.") from exc


@router.post("/{model_id}/validate", response_model=ModelValidationResponse)
def validate_model(model_id: str) -> ModelValidationResponse:
    try:
        return validate_uploaded_model(model_id)
    except (ModelUploadError, ModelLoadError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
