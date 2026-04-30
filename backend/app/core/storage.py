from app.core.config import settings


STORAGE_DIRS = [
    settings.storage_dir,
    settings.uploads_dir,
    settings.uploads_dir / "models",
    settings.uploads_dir / "datasets",
    settings.outputs_dir,
    settings.outputs_dir / "predictions",
    settings.outputs_dir / "failures",
    settings.outputs_dir / "heatmaps" / "gradcam",
    settings.outputs_dir / "heatmaps" / "shap",
    settings.outputs_dir / "heatmaps" / "lime",
    settings.outputs_dir / "heatmaps" / "fusion",
    settings.outputs_dir / "clusters",
    settings.outputs_dir / "reports",
]


def ensure_storage_dirs() -> None:
    for path in STORAGE_DIRS:
        path.mkdir(parents=True, exist_ok=True)
