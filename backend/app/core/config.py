from pathlib import Path

from pydantic import BaseModel


class Settings(BaseModel):
    app_name: str = "XAIFA"
    app_version: str = "0.1.0"
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    backend_dir: Path = Path(__file__).resolve().parents[2]
    project_root: Path = backend_dir.parent
    storage_dir: Path = project_root / "storage"
    uploads_dir: Path = storage_dir / "uploads"
    outputs_dir: Path = storage_dir / "outputs"


settings = Settings()
