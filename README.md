# XAIFA

Explainable AI Failure Analyzer.

XAIFA is a web-based platform for analyzing failures in uploaded image classification
models. A user uploads a trained PyTorch model, a labelled test dataset, and class
labels. XAIFA then runs predictions, collects failed cases, generates Grad-CAM, SHAP,
and LIME explanations, compares solo and fused explanation methods, clusters similar
failures, and shows improvement recommendations on a dashboard.

## Current Development Status

The original prototype scripts are still available in `scripts/` and `models/`.
The new production-style app is being built in phases:

1. Backend API foundation.
2. Model upload and loading.
3. Dataset upload and preprocessing.
4. Prediction and failure collection.
5. Grad-CAM, SHAP, and LIME explanations.
6. XAI scoring and fusion comparison.
7. Failure clustering and recommendations.
8. Frontend dashboard.

## Backend

The FastAPI backend lives in `backend/`.

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

API docs:

```text
http://127.0.0.1:8000/docs
```

Initial endpoints:

- `GET /api/health`
- `GET /api/models/supported`
- `POST /api/models/upload`
- `POST /api/models/{model_id}/validate`
- `POST /api/datasets/upload`
- `GET /api/datasets/{dataset_id}/inspect`
- `GET /api/runs/pipeline`
- `GET /api/runs`
- `POST /api/runs/analyze`
- `GET /api/runs/{run_id}`
- `GET /api/runs/{run_id}/failures`

## Documentation

Project workflow and development plan:

- `docs/XAIFA_User_Workflow_and_Development_Plan.md`
- `docs/XAIFA_User_Workflow_and_Development_Plan.docx`
