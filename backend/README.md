# XAIFA Backend

FastAPI backend for the XAIFA dashboard.

## Current Purpose

This backend is the foundation for:

- model upload and validation
- dataset upload and preprocessing
- prediction analysis
- failure collection
- Grad-CAM, SHAP, and LIME explanation generation
- XAI fusion and comparison
- clustering and recommendations

The first development milestone is API structure. ML logic will be added service-by-service.

## Initial Endpoints

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

## Run Locally

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Then open:

```text
http://127.0.0.1:8000/docs
```
