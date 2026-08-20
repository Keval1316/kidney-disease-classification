# Project State & Handoff Document: Kidney Disease Classification

> **Purpose:** This document is the single source of truth for the project state. Any AI assistant or developer can read this file to understand completed phases, project architecture, verified modules, and exact instructions on how to continue.

---

## 1. Project Overview & Architecture

- **Project Goal:** Build an end-to-end Deep Learning & MLOps system for **CT Kidney Disease Classification** into 4 classes:
  1. `Normal`
  2. `Cyst`
  3. `Tumor`
  4. `Stone`
- **Dataset:** Kaggle CT Kidney Dataset (Normal, Cyst, Tumor, Stone) - 12,446 images total.
- **Backbone Model:** `EfficientNetB0` (Pretrained on ImageNet, Transfer Learning with trainable classification head, opt-in fine-tuning).
- **Core Stack:**
  - **DL Framework:** TensorFlow / Keras (Python 3.11)
  - **Data & Pipeline Versioning:** DVC (DagsHub remote storage)
  - **Experiment Tracking:** MLflow (DagsHub hosted MLflow)
  - **Backend Inference API:** FastAPI (Uvicorn)
  - **Frontend UI:** Streamlit
  - **Containerization:** Docker (FastAPI inference image)
  - **CI/CD:** GitHub Actions (`ci.yml` & `cd.yml`)
  - **Deployment Targets (100% Free-Tier):**
    - FastAPI Backend $\rightarrow$ Render Free Tier (Docker)
    - Streamlit UI $\rightarrow$ Streamlit Community Cloud (Calling Render FastAPI)

---

## 2. Phase-by-Phase Progress Tracker

| Phase | Description | Key Artifacts / Scripts | Status |
|---|---|---|:---:|
| **Phase 0** | **Architecture & Design** | Complete MLOps blueprint, free-tier cost design | **DONE** |
| **Phase 1** | **Project Setup** | `pyproject.toml`, `requirements.txt`, `.gitignore`, `config/` | **DONE** |
| **Phase 2** | **Environment Setup** | Python 3.11 `.venv`, pinned dependencies | **DONE** |
| **Phase 3** | **Dataset Ingestion** | `src/data/download.py` (Kaggle download & check) | **DONE** |
| **Phase 4** | **Data Validation** | `src/data/validate.py` $\rightarrow$ `reports/metrics/dataset_summary.json` | **DONE** |
| **Phase 5** | **Data Splitting** | `src/data/split.py` (Stratified 70/15/15) $\rightarrow$ `data/processed/splits.json` | **DONE** |
| **Phase 6** | **Preprocessing** | `src/preprocessing/preprocessing.py` (EfficientNet normalization & tf.data) | **DONE** |
| **Phase 7** | **Model Architecture** | `src/model/model.py`, `src/model/callbacks.py` | **DONE** |
| **Phase 8** | **Training Pipeline** | `src/training/train.py` $\rightarrow$ `artifacts/model/best_model.keras` | **DONE** |
| **Phase 9** | **MLflow Integration** | MLflow logging with DagsHub integration | **DONE** |
| **Phase 10** | **Model Evaluation** | `src/evaluation/evaluate.py` $\rightarrow$ `metrics.json`, `confusion_matrix.png` | **DONE** |
| **Phase 11** | **Error Analysis** | `src/evaluation/error_analysis.py` $\rightarrow$ `reports/figures/error_analysis_examples/` | **DONE** |
| **Phase 12** | **DVC Pipeline** | `dvc.yaml`, `dvc.lock` (Stages: download, validate, split, train, evaluate) | **DONE** |
| **Phase 13** | **Reproducibility** | `dvc repro` verified working | **DONE** |
| **Phase 14** | **Logging Setup** | `src/utils/logger.py` (Console + Rotating File `logs/kidney_clf.log`) | **DONE** |
| **Phase 15** | **Inference Module** | `src/inference/predict.py` (Model caching, probability outputs) | **DONE** |
| **Phase 16** | **FastAPI Service** | `api/main.py` (`/`, `/health`, `/predict`, CORS, Pydantic) | **DONE** |
| **Phase 17** | **Streamlit UI** | `app/streamlit_app.py` (Interactive UI calling FastAPI) | **DONE** |
| **Phase 18** | **Dockerization** | `Dockerfile`, `.dockerignore` for FastAPI service | **DONE** |
| **Phase 19** | **Local Docker Test** | Docker execution guide & health/predict test procedure | **DONE** |
| **Phase 20** | **Automated Tests** | 15 passing tests across `test_data.py`, `test_model.py`, `test_api.py`, `test_inference.py` | **DONE** |
| **Phase 21** | **CI (GitHub Actions)** | `.github/workflows/ci.yml` (Lint, test, smoke train, docker build) | **DONE** |
| **Phase 22** | **CD (GitHub Actions)** | `.github/workflows/cd.yml` (Deploy to Render via webhook/secrets) | **DONE** |
| **Phase 23** | **Render Deployment** | Deploy FastAPI Docker container to Render Free Tier | **PENDING** |
| **Phase 23b** | **Streamlit Deployment** | Deploy Streamlit UI to Streamlit Community Cloud | **PENDING** |
| **Phase 24** | **Model Strategy** | Lightweight packaging & artifact strategy | **PENDING** |
| **Phase 25** | **Model Versioning** | Versioning tag & tracking strategy documentation | **PENDING** |
| **Phase 26** | **Configuration** | `.env`, `.env.example`, environment variables | **PENDING** |
| **Phase 27** | **Security Audit** | Verifying zero hardcoded secrets / credentials | **PENDING** |
| **Phase 28** | **Comprehensive README** | Full professional portfolio README with diagrams & metrics | **PENDING** |
| **Phase 29** | **Final Acceptance** | Complete verification checklist against Master Prompt | **PENDING** |

---

## 3. Directory Structure & Important Files Reference

```text
kidney-disease-classification/
├── .github/
│   └── workflows/
│       ├── ci.yml                  # [CRITICAL] Lint, unit tests, smoke train & Docker build
│       └── cd.yml                  # [CRITICAL] Automatic deployment trigger for Render
├── api/
│   ├── __init__.py
│   └── main.py                     # [CRITICAL] FastAPI application with /, /health, /predict
├── app/
│   └── streamlit_app.py            # [CRITICAL] Streamlit frontend (Phase 17)
├── artifacts/
│   └── model/
│       ├── best_model.keras        # [CRITICAL] Trained EfficientNetB0 Keras model weights
│       └── final_model.keras
├── config/
│   └── config.yaml                 # Central paths & project configuration
├── data/
│   ├── raw/                        # CT scan images partitioned by class
│   └── processed/
│       └── splits.json             # Stratified dataset split file paths
├── logs/
│   └── kidney_clf.log              # Rotating log file
├── reports/
│   ├── figures/
│   │   ├── confusion_matrix.png    # Evaluation confusion matrix
│   │   ├── training_history.png    # Loss & accuracy curves
│   │   └── error_analysis_examples/
│   └── metrics/
│       ├── dataset_summary.json
│       ├── metrics.json            # Test set accuracy, precision, recall, f1
│       ├── classification_report.json
│       └── training_history.json
├── src/
│   ├── data/
│   │   ├── download.py             # Kaggle dataset ingestion
│   │   ├── validate.py             # Data integrity & class balance validator
│   │   └── split.py                # Train/Val/Test stratified partitioner
│   ├── preprocessing/
│   │   └── preprocessing.py        # Image resizing, tf.data pipeline & augmentation
│   ├── model/
│   │   ├── model.py                # EfficientNetB0 architecture definition
│   │   └── callbacks.py            # EarlyStopping, ReduceLROnPlateau, Checkpoints
│   ├── training/
│   │   └── train.py                # Model training loop & DagsHub MLflow logger
│   ├── evaluation/
│   │   ├── evaluate.py             # Test set evaluation & metric calculation
│   │   └── error_analysis.py       # Best/worst confidence error analyzer
│   ├── inference/
│   │   └── predict.py              # [CRITICAL] Image preprocessing & cached predictor
│   └── utils/
│       ├── common.py               # YAML & JSON helper utilities
│       └── logger.py               # Logging configuration
├── tests/
│   ├── __init__.py
│   ├── test_data.py                # Data validation & split leakage tests
│   ├── test_model.py               # Architecture & smoke-training test
│   ├── test_inference.py           # Inference engine & probability tests
│   └── test_api.py                 # FastAPI endpoints & error handling tests
├── .dockerignore                    # Excludes datasets, local caches, tests from image
├── .env.example
├── .gitignore
├── Dockerfile                      # [CRITICAL] Single-service FastAPI container definition
├── dvc.yaml                        # 5-stage DVC pipeline
├── dvc.lock
├── params.yaml                     # Hyperparameters (epochs, batch_size, lr, etc.)
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt            # Dev dependencies: pytest, flake8, ruff, httpx
└── PROJECT_STATE.md                # THIS FILE
```

---

## 4. Key Verified Commands

- **Run Full Automated Test Suite:**
  ```powershell
  pytest -v
  ```
- **Run Inference directly via Python:**
  ```powershell
  python -m src.inference.predict "data/raw/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/Tumor/Tumor- (1).jpg"
  ```
- **Start FastAPI inference backend:**
  ```powershell
  uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
  ```
- **Start Streamlit UI:**
  ```powershell
  streamlit run app/streamlit_app.py
  ```
- **Build Docker Container:**
  ```powershell
  docker build -t kidney-classifier .
  ```
- **Run Docker Container Locally:**
  ```powershell
  docker run -p 8000:8000 -e PORT=8000 kidney-classifier
  ```
