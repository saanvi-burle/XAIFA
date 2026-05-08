from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import datasets, health, models, runs
from app.core.config import settings
from app.core.storage import ensure_storage_dirs
from app.api.routes import heatmaps

def create_app() -> FastAPI:
    ensure_storage_dirs()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Explainable AI Failure Analyzer API",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers
    app.include_router(health.router, prefix="/api", tags=["health"])
    app.include_router(models.router, prefix="/api/models", tags=["models"])
    app.include_router(datasets.router, prefix="/api/datasets", tags=["datasets"])
    app.include_router(runs.router, prefix="/api/runs", tags=["analysis-runs"])

    app.include_router(heatmaps.router, prefix="/api/heatmaps", tags=["heatmaps"],)

    return app


# App instance
app = create_app()
from pathlib import Path
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# CORRECT RESULTS FOLDER
RESULTS_DIR = BASE_DIR / "results"

# CREATE IF NOT EXISTS
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"[DEBUG] Serving results from: {RESULTS_DIR}")

# MOUNT STATIC FILES
app.mount(
    "/results",
    StaticFiles(directory=str(RESULTS_DIR)),
    name="results"
)