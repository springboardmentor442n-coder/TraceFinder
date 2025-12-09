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
4. frontend/package.json and `frontend/src/` — to run the frontend dev server.
5. `frontend/public/bg-page.jpg` (optional) and `frontend/public/assets/*` (confusion images) — used by the UI.

If you only want to run the UI without local predictions, you can start the frontend without any backend running; the UI currently has a mock prediction flow used for development.

---

## Quick start (development)

Prerequisites:

- Node.js (LTS) and npm installed for the frontend.
- Python 3.10+ for the backend (a virtual environment is recommended).

Backend (Python):

1. Create and activate a virtual environment from the repo root:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install Python requirements (from repo root `requirements.txt`):

```powershell
pip install -r ..\requirements.txt
```

3. Start the API (from `backend`):

```powershell
uvicorn api.main:app --reload --port 8000
```

OR use the provided helper from the repo root:

```powershell
.\start-backend.ps1
```

Frontend (Node + Vite):

1. From repo root:

```powershell
cd frontend
npm install
npm run dev
```

2. Open the dev server URL printed by Vite (usually `http://localhost:3000`).

Note: the frontend has mock prediction behavior for development; to get real predictions wire the frontend to the running backend (defaults to `http://localhost:8000`).

---

## Requirements / Dependencies

Python (root `requirements.txt`) — packages used by the backend and inference code (recommended minimum):

```
fastapi
uvicorn[standard]
numpy
pillow
scikit-learn
tensorflow
aiofiles
python-multipart
pydantic
```

Frontend (from `frontend/package.json`):

```
react
react-dom
react-router-dom
vite
tailwindcss
postcss
autoprefixer
@vitejs/plugin-react
```

You can install Node deps with `npm install` inside `frontend/`.

---

## Project notes & suggestions

- The heavy model files and `Data/` folder contain many large binary artifacts; they are not required to run the UI but are necessary if you want to run accurate local inference.
- If you retain the `Notebooks/` folder, you can find training scripts and serialized artifacts used during model development.
- To reduce repository size for distribution, consider moving large datasets and model weights to a separate storage (S3, Google Drive) and keep only the code and small sample models in the git repo.

---

If you want, I can now:

1. Expand this README with a full file listing (per-folder) — note this can be very large (many thousands of files). I can produce a trimmed listing with the most relevant files.
2. Update the `requirements.txt` with pinned versions from your current Python environment (if you want exact reproducibility I can inspect the `.venv` and pin versions).

Tell me which of the two you'd like and I'll proceed.

---

## Additional files discovered and suggested README entries

While scanning the repository I found a number of files and modules used by the backend and inference pipeline that are useful to list explicitly in the README so new contributors know which files are required for predictions and which are optional research artifacts.

- Backend entry & inference modules (required for running the API with real predictions):
	- `backend/api/main.py` — FastAPI application and endpoints.
	- `backend/inference/models.py` — loads saved models and exposes `predict_single_image`.
	- `backend/inference/features.py` — feature-extraction helpers (residuals, handcrafted features).
	- `backend/inference/singleimage_prediction.py` — example single-image prediction logic used in notebooks.
	- `backend/inference/utils.py` — helper utilities for image resizing/padding and other helpers.
	- `backend/models/` — contains Keras model files like `cnn_residual_best.keras` and `scanner_hybrid_best.keras` (required for real predictions).

- Frontend (UI):
	- `frontend/package.json` — frontend dependencies and scripts (Vite dev server).
	- `frontend/src/pages/Home.jsx`, `frontend/src/pages/ModelGallery.jsx`, `frontend/src/pages/ModelDetail.jsx` — main pages used for navigation and UI.
	- `frontend/public/assets/` — static images used in the UI (confusion matrices, icons).

- Notebooks and training artifacts (optional, for reproduction and retraining):
	- `Notebooks/CNN_model_training.py`, `Notebooks/hybrid_cnn.py`, `Notebooks/preprocess.py`, `Notebooks/singleimage_prediction.py` — training and prediction scripts used for research.
	- Many large serialized artifacts (`*.keras`, `*.h5`, `*.npy`, pickles) are stored across `Notebooks/` and `backend/models/`.

- Helpers and utilities found at repo root:
	- `start-backend.ps1` — helper PowerShell script to start the backend.
	- `test.py` — a small test harness that calls `backend.inference.predict_single_image` (useful for quick sanity checks).
	- `checklist.py` / `checklist_output.json` — repository inspection helpers used by the project.

## Updated requirements (additional packages added)

I updated `requirements.txt` to include additional packages discovered while scanning Python files (used by the notebooks and backend inference code). These packages are not all strictly required for the minimal API, but they are necessary if you want to run training, preprocessing, or the complete inference pipeline locally:

```
fastapi
uvicorn[standard]
numpy
pillow
scikit-learn
tensorflow
aiofiles
python-multipart
pydantic
opencv-python
scikit-image
scipy
pandas
matplotlib
seaborn
joblib
tqdm
PyWavelets
```

---