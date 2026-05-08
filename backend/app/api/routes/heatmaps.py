from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.core.config import settings

router = APIRouter()


@router.get("/{method}/{failure_id}")
def get_heatmap(method: str, failure_id: str):

    allowed_methods = [
        "gradcam",
        "shap",
        "lime",
        "fusion",
    ]

    if method not in allowed_methods:
        raise HTTPException(
            status_code=400,
            detail="Invalid heatmap method",
        )

    heatmap_path = (
        settings.outputs_dir
        / "heatmaps"
        / method
        / f"{failure_id}.png"
    )

    print("\n🔥 REQUESTED:")
    print("Method:", method)
    print("Failure ID:", failure_id)
    print("Path:", heatmap_path)

    if not heatmap_path.exists():

        print("❌ FILE NOT FOUND")

        raise HTTPException(
            status_code=404,
            detail=f"Heatmap not found: {heatmap_path}",
        )

    print("✅ FILE FOUND")

    return FileResponse(
        path=heatmap_path,
        media_type="image/png",
    )