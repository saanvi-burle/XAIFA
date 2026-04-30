from pydantic import BaseModel


class DatasetUploadResponse(BaseModel):
    dataset_id: str
    filename: str
    dataset_format: str
    labels_filename: str | None = None
    saved_path: str
    labels_path: str | None = None
    status: str
    next_step: str


class DatasetInspectResponse(BaseModel):
    dataset_id: str
    dataset_format: str
    image_count: int
    class_count: int
    classes: list[str]
    csv_rows: int | None = None
    status: str
    message: str
