# Trace Finder (Image Residuals Detector)

Trace Finder is an experimental project for scanner-source identification and image residual analysis. It provides a FastAPI backend for inference, helper scripts and notebooks for training/analysis, and a small React + Vite frontend for demonstrations.

This README gives a short, practical guide to get the project running locally.

**Project layout (high level)**
- **Backend:** `backend/` — FastAPI app and inference modules (`backend/api/main.py`).
- **Inference:** `backend/inference/` — feature extraction and prediction utilities.
- **Models:** `backend/models/` and `models/` — saved model weights (not committed by default).
- **Frontend:** `frontend/` — Vite + React demo app.
- **Notebooks:** `Notebooks/` — training and analysis notebooks (kept separate; large artifacts live here).
- **Data / Output:** `Data/`, `Output/` — datasets and exported CSVs (ignored by default).

**Prerequisites**
- Python 3.8+ (3.10 recommended)
- Node.js 16+ and npm (for frontend dev)
- (Optional) GPU-enabled TensorFlow for faster model inference on large images

**Local setup (PowerShell)**

1) Create and activate a Python virtual environment from the repo root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Install Python dependencies:

```powershell
pip install -r requirements.txt
```

3) Start the backend API (option A: helper script; option B: uvicorn):

```powershell
# Option A (provided helper)
.\start-backend.ps1

# Option B (manual)
cd backend
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

4) Run the frontend (optional, from repo root):

```powershell
cd frontend
npm install
npm run dev
```

Open the frontend dev URL printed by Vite (usually `http://localhost:3000`) and point the UI to the backend at `http://localhost:8000` if needed.

**Where to place models & data**
- Model weights: put Keras/TensorFlow model files under `backend/models/` or `models/` (example filenames: `cnn_residual_best.keras`, `scanner_hybrid_best.keras`).
- Data: place datasets in `Data/` and outputs in `Output/`.

**Quick commands**
- Create & activate venv: `` `python -m venv .venv` `` then `` `.\.venv\Scripts\Activate.ps1` ``
- Install deps: `` `pip install -r requirements.txt` ``
- Start backend: `` `.\start-backend.ps1` `` or `` `uvicorn api.main:app --reload` ``

**Notes & recommendations**
- Large datasets and model files are intentionally ignored by Git. Use `.gitignore` and `.gitattributes` provided in the repo root.
- If you need exact dependency versions for reproducibility, I can pin the current environment versions to `requirements.txt` for you.

If you'd like, I can now:
- Pin dependency versions in `requirements.txt` from your active `.venv`.
- Stage and commit the initial files (`README.md`, `backend/`, `frontend/`, `requirements.txt`).
# Trace Finder (Image Residuals Detector)

Trace Finder is an experimental project for scanner-source identification and image residual analysis. It provides a FastAPI backend for inference, helper scripts and notebooks for training/analysis, and a small React + Vite frontend for demonstrations.

This README gives a short, practical guide to get the project running locally.

**Project layout (high level)**
- **Backend:** `backend/` — FastAPI app and inference modules (`backend/api/main.py`).
- **Inference:** `backend/inference/` — feature extraction and prediction utilities.
- **Models:** `backend/models/` and `models/` — saved model weights (not committed by default).
- **Frontend:** `frontend/` — Vite + React demo app.
- **Notebooks:** `Notebooks/` — training and analysis notebooks (kept separate; large artifacts live here).
- **Data / Output:** `Data/`, `Output/` — datasets and exported CSVs (ignored by default).

**Prerequisites**
- Python 3.8+ (3.10 recommended)
- Node.js 16+ and npm (for frontend dev)
- (Optional) GPU-enabled TensorFlow for faster model inference on large images

**Local setup (PowerShell)**

1) Create and activate a Python virtual environment from the repo root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Install Python dependencies:

```powershell
pip install -r requirements.txt
```

3) Start the backend API (option A: helper script; option B: uvicorn):

```powershell
# Option A (provided helper)
.\start-backend.ps1

# Option B (manual)
cd backend
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

4) Run the frontend (optional, from repo root):

```powershell
cd frontend
npm install
npm run dev
```

Open the frontend dev URL printed by Vite (usually `http://localhost:3000`) and point the UI to the backend at `http://localhost:8000` if needed.

**Where to place models & data**
- Model weights: put Keras/TensorFlow model files under `backend/models/` or `models/` (example filenames: `cnn_residual_best.keras`, `scanner_hybrid_best.keras`).
- Data: place datasets in `Data/` and outputs in `Output/`.

**Quick commands**
- Create & activate venv: `` `python -m venv .venv` `` then `` `.\.venv\Scripts\Activate.ps1` ``
- Install deps: `` `pip install -r requirements.txt` ``
- Start backend: `` `.\start-backend.ps1` `` or `` `uvicorn api.main:app --reload` ``

**Notes & recommendations**
- Large datasets and model files are intentionally ignored by Git. Use `.gitignore` and `.gitattributes` provided in the repo root.
- If you need exact dependency versions for reproducibility, I can pin the current environment versions to `requirements.txt` for you.

If you'd like, I can now:
- Pin dependency versions in `requirements.txt` from your active `.venv`.
- Stage and commit the initial files (`README.md`, `backend/`, `frontend/`, `requirements.txt`).
# Trace Finder

Trace Finder is a small experimental project for scanner-source identification. It contains:

- A FastAPI backend that exposes prediction endpoints and serves a few static assets.
- A React + Vite frontend (in `frontend/`) used to explore models and upload images for prediction.
- Pretrained models and training artifacts stored under `Notebooks/` and `backend/models/`.
- Datasets and residual images under `Data/`.

This README documents the project's structure, the files required to run the app locally, and quick run instructions.

---

## Important folders & files (brief inventory)

- `backend/`
	- `backend/api/main.py` — FastAPI application and endpoints (`/predict-file`, `/predict-path`, `/assets/*`). This is the backend entrypoint.
	- `backend/inference/` — Python modules that implement feature extraction and `predict_single_image`. (Essential for server-side predictions.)
	- `backend/models/` — model files (e.g. `cnn_residual_best.keras`, `scanner_hybrid_best.keras`) used by the inference code.

- `frontend/`
	- `frontend/package.json` — contains frontend dependencies and `dev` script (Vite).
	- `frontend/src/` — React app source (pages: `Home.jsx`, `ModelGallery.jsx`, `ModelDetail.jsx`, components: `UploadPanel.jsx`, etc.).
	- `frontend/public/` — static assets (background image `bg-page.jpg`, `assets/` confusion images).

- `Notebooks/` (research/experiment files)
	- Training scripts and notebooks (for example: `CNN_model_training.py`, `hybrid_cnn.ipynb`) and many saved artifacts: `*.keras`, `*.h5`, label encoders, feature pickles.
	- These files are useful for model retraining and reproducing experiments but are not required for running the frontend dev server.

- `Data/` — raw and processed datasets, residual images and intermediate files used for training and analysis. Large and not required to run the UI locally unless you want to reproduce training or offline predictions.

- `Output/` — CSVs and exported outputs from earlier runs and analyses.

---

## Files required to run the application (minimum)

1. backend/api/main.py (the FastAPI app)
2. backend/inference/models.py (or equivalent) — must provide `predict_single_image(path, model=None, verbose=False)`
3. backend/models/* (at least one saved model file that your inference code can load). Without models the API can still run but predictions will fail.
# Trace Finder (Image Residuals Detector)

Trace Finder is an experimental project for scanner-source identification and image residual analysis. It provides a FastAPI backend for inference, helper scripts and notebooks for training/analysis, and a small React + Vite frontend for demonstrations.

This README gives a short, practical guide to get the project running locally.

**Project layout (high level)**
- **Backend:** `backend/` — FastAPI app and inference modules (`backend/api/main.py`).
- **Inference:** `backend/inference/` — feature extraction and prediction utilities.
- **Models:** `backend/models/` and `models/` — saved model weights (not committed by default).
- **Frontend:** `frontend/` — Vite + React demo app.
- **Notebooks:** `Notebooks/` — training and analysis notebooks (kept separate; large artifacts live here).
- **Data / Output:** `Data/`, `Output/` — datasets and exported CSVs (ignored by default).

**Prerequisites**
- Python 3.8+ (3.10 recommended)
- Node.js 16+ and npm (for frontend dev)
- (Optional) GPU-enabled TensorFlow for faster model inference on large images

**Local setup (PowerShell)**

1) Create and activate a Python virtual environment from the repo root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Install Python dependencies:

```powershell
pip install -r requirements.txt
```

3) Start the backend API (option A: helper script; option B: uvicorn):

```powershell
# Option A (provided helper)
.\start-backend.ps1

# Option B (manual)
cd backend
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

4) Run the frontend (optional, from repo root):

```powershell
cd frontend
npm install
npm run dev
```

Open the frontend dev URL printed by Vite (usually `http://localhost:3000`) and point the UI to the backend at `http://localhost:8000` if needed.

**Where to place models & data**
- Model weights: put Keras/TensorFlow model files under `backend/models/` or `models/` (example filenames: `cnn_residual_best.keras`, `scanner_hybrid_best.keras`).
- Data: place datasets in `Data/` and outputs in `Output/`.

**Quick commands**
- Create & activate venv: `` `python -m venv .venv` `` then `` `.\.venv\Scripts\Activate.ps1` ``
- Install deps: `` `pip install -r requirements.txt` ``
- Start backend: `` `.\start-backend.ps1` `` or `` `uvicorn api.main:app --reload` ``

