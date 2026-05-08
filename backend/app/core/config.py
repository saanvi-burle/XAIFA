from pathlib import Path
from pydantic import BaseModel


class Settings(BaseModel):
    app_name: str = "XAIFA"
    app_version: str = "0.1.0"

    cors_origins: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    # Base paths
    backend_dir: Path = Path(__file__).resolve().parents[2]
    project_root: Path = backend_dir.parent

    # Storage
    storage_dir: Path = project_root / "storage"
    uploads_dir: Path = storage_dir / "uploads"
    outputs_dir: Path = storage_dir / "outputs"

    # 🔥 ADD THIS (CRITICAL FIX)
    heatmaps_dir: Path = outputs_dir / "heatmaps"

    # Optional but useful structure
    gradcam_dir: Path = heatmaps_dir / "gradcam"
    shap_dir: Path = heatmaps_dir / "shap"
    lime_dir: Path = heatmaps_dir / "lime"
    fusion_dir: Path = heatmaps_dir / "fusion"


settings = Settings()


# ✅ CREATE FOLDERS AUTOMATICALLY
def ensure_dirs():
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    settings.outputs_dir.mkdir(parents=True, exist_ok=True)

    settings.heatmaps_dir.mkdir(parents=True, exist_ok=True)
    settings.gradcam_dir.mkdir(parents=True, exist_ok=True)
    settings.shap_dir.mkdir(parents=True, exist_ok=True)
    settings.lime_dir.mkdir(parents=True, exist_ok=True)
    settings.fusion_dir.mkdir(parents=True, exist_ok=True)


ensure_dirs()