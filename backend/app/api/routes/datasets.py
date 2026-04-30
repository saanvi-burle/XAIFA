from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.schemas.datasets import DatasetInspectResponse, DatasetUploadResponse
from app.services.dataset_registry import (
    DatasetUploadError,
    inspect_uploaded_dataset,
    save_uploaded_dataset,
)


router = APIRouter()


@router.post("/upload", response_model=DatasetUploadResponse)
async def upload_dataset(
    dataset_file: UploadFile = File(...),
    dataset_format: str = Form(...),
    labels_file: UploadFile | None = File(default=None),
) -> DatasetUploadResponse:
    try:
        return await save_uploaded_dataset(
            dataset_file=dataset_file,
            dataset_format=dataset_format,
            labels_file=labels_file,
        )
    except DatasetUploadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{dataset_id}/inspect", response_model=DatasetInspectResponse)
def inspect_dataset(dataset_id: str) -> DatasetInspectResponse:
    try:
        return inspect_uploaded_dataset(dataset_id)
    except DatasetUploadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
