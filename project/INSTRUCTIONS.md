# TraceFinder — Setup & Run Instructions

These instructions show how to create a local environment, install dependencies, and run the backend and frontend for the TraceFinder project on Windows (PowerShell).

## Prerequisites

- Python 3.10+ installed
- Git (for cloning / updating the repo)
- (Optional) Git LFS if you plan to store large model files in the remote repository

## 1 — Create & activate a virtual environment (PowerShell)

```powershell
python -m venv .venv
. .venv/Scripts/activate
```

Note: The activation command above is for PowerShell. For cmd.exe use `.venv\\Scripts\\activate.bat`.

## 2 — Install dependencies

```powershell
pip install -r requirements.txt
```

## 3 — Run the backend API

The backend uses FastAPI (served by Uvicorn). Start it with:

```powershell
uvicorn backend.main:app --reload
```

This will start the API at `http://127.0.0.1:8000` by default.

## 4 — Run the frontend (Streamlit)

Open a new terminal (keep the backend running) and run:

```powershell
streamlit run frontend/app.py
```

The Streamlit UI will open in your browser (usually at `http://localhost:8501`).

## Notes about model files

- The repository does not include large model artifacts by default (these can exceed GitHub's file size limits).
- Locally the project expects models under `project/models/` (for example `cnn_final_model.keras`). If you have a model file keep it in that folder locally. Do not add it to Git unless you use Git LFS.

If you want to add large models to the remote using Git LFS:

```powershell
git lfs install
git lfs track "project/models/*.keras"
git add .gitattributes
git add project/models/<your-model>.keras
git commit -m "Add model via Git LFS"
git push origin Hanvith-Sai-Alla
```

Or alternatively upload model files to cloud storage (OneDrive / Google Drive / Azure Blob) and include a small download script or instructions in the repo.

## Troubleshooting

- If you see errors about missing packages, re-run `pip install -r requirements.txt` inside the activated `.venv`.
- If a port is already in use, pass `--port <num>` to `uvicorn` or stop the process using that port.

---

If you want, I can also replace the existing `instructions.txt` with this Markdown and commit the change. Would you like me to do that now?
