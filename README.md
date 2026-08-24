# CT Kidney Disease Classification

Educational deep learning + MLOps project classifying kidney CT scans into
Normal / Cyst / Tumor / Stone.

> ⚠️ **This is an educational/portfolio project. It is NOT a medical
> diagnostic device and must not be used for clinical decision-making.**

## Quick Start

### 1. Start the API and Interactive Web UI
```powershell
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

- **Interactive Diagnostic Web UI:** [http://127.0.0.1:8000/ui](http://127.0.0.1:8000/ui)
- **FastAPI Interactive Docs (Swagger):** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Health Check Endpoint:** [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

### 2. Run Test Suite
```powershell
pytest -v
```