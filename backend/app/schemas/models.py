from pydantic import BaseModel, Field


class SupportedModelResponse(BaseModel):
    architectures: list[str]


class ModelUploadResponse(BaseModel):
    model_id: str
    filename: str
    model_format: str
    architecture: str | None = None
    input_width: int = Field(gt=0)
    input_height: int = Field(gt=0)
    channels: int = Field(gt=0)
    num_classes: int = Field(gt=0)
    saved_path: str
    metadata_path: str
    status: str
    next_step: str


class ModelValidationResponse(BaseModel):
    model_id: str
    status: str
    architecture: str | None = None
    model_format: str
    input_shape: list[int]
    output_shape: list[int] | None = None
    message: str
