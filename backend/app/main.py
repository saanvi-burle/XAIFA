from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import datasets, health, models, runs
from app.core.config import settings
from app.core.storage import ensure_storage_dirs


def create_app() -> FastAPI:
    ensure_storage_dirs()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Explainable AI Failure Analyzer API",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router, prefix="/api", tags=["health"])
    app.include_router(models.router, prefix="/api/models", tags=["models"])
    app.include_router(datasets.router, prefix="/api/datasets", tags=["datasets"])
    app.include_router(runs.router, prefix="/api/runs", tags=["analysis-runs"])

    return app


app = create_app()
