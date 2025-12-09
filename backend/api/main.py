# backend/api/main.py
import os
import tempfile
import traceback
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# Try to import the prediction wrapper from the inference package
try:
    # primary: import from package (works when backend is a package)
    from backend.inference.models import predict_single_image
except Exception as e:
    # fallback: try a relative import (useful for some dev setups)
    try:
        from inference.models import predict_single_image  # type: ignore
    except Exception:
        raise ImportError(
            "Could not import predict_single_image from backend.inference.models.\n"
            "Make sure 'backend' is on PYTHONPATH and backend/inference/models.py exists.\n"
            "Original error: " + str(e)
        )


# create FastAPI app once
app = FastAPI(title="TraceFinder API")

# Allow React dev server(s) on port 3000 (adjust if you host differently)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PathPayload(BaseModel):
    path: str
    model: Optional[str] = None  # optional: which model frontend asked for (not enforced server-side)


@app.get("/")
def health():
    return {"status": "ok", "message": "TraceFinder API - healthy"}


@app.post("/predict-path")
def predict_from_path(payload: PathPayload):
    """
    Predict from an existing file path on the server.
    payload.path must be the absolute path the server can read.
    Optional payload.model is accepted and echoed back.
    """
    img_path = payload.path
    if not os.path.exists(img_path):
        raise HTTPException(status_code=400, detail=f"File does not exist: {img_path}")

    try:
        res = predict_single_image(img_path, model=payload.model, verbose=False)
        return {"status": "ok", "result": res, "model_requested": payload.model}
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail={"error": str(e), "traceback": tb})


@app.post("/predict-file")
async def predict_from_file(
    file: UploadFile = File(...),
    model: Optional[str] = Form(None),
):
    """
    Accepts a multipart file upload (field name: 'file').
    Optional form field 'model' can be passed from the frontend (e.g. 'cnn', 'hybrid', 'sklearn').
    Currently the server will still run the existing prediction code (ensemble). 'model' is echoed.
    """
    # create a temporary copy of the uploaded file so predict_single_image can access it
    suffix = os.path.splitext(file.filename)[1] or ".tif"
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            content = await file.read()
            tmp.write(content)

        if not os.path.exists(tmp_path):
            raise RuntimeError("Temporary file not found after write.")

        # run prediction (note: your predict_single_image should accept file path)
        result = predict_single_image(tmp_path, model=model, verbose=False)

        # cleanup temp file (best-effort)
        try:
            os.remove(tmp_path)
        except Exception:
            pass

        return {"status": "ok", "result": result, "model_requested": model, "filename": file.filename}

    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail={"error": str(e), "traceback": tb})


# --- Small static asset helper (serve confusion images from a few known locations) ---
# The frontend can request: /assets/cnn_confusionmatrix.png
ASSET_SEARCH_DIRS = [
    os.path.join(os.path.dirname(__file__), "..", "Notebooks"),        # backend/api/../Notebooks
    os.path.join(os.path.dirname(__file__), "..", "..", "Notebooks"),   # backend/../../Notebooks
    os.path.join(os.path.dirname(__file__), "..", "models"),           # backend/api/../models
    os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "public", "assets"),  # frontend public
]


def find_asset(filename: str) -> Optional[str]:
    for d in ASSET_SEARCH_DIRS:
        d = os.path.abspath(d)
        if not os.path.isdir(d):
            continue
        candidate = os.path.join(d, filename)
        if os.path.exists(candidate):
            return candidate
    return None


@app.get("/assets/{name}")
def serve_asset(name: str):
    """
    Serve common images (confusion matrices, etc.) to the frontend.
    Looks for file in a few likely folders (Notebooks, backend/models, frontend/public/assets).
    """
    # sanitize a tiny bit
    if ".." in name or name.startswith("/"):
        raise HTTPException(status_code=400, detail="Invalid filename")

    path = find_asset(name)
    if not path:
        raise HTTPException(status_code=404, detail=f"Asset not found: {name}")
    return FileResponse(path)


# Optionally add a small health-check for model readiness
@app.get("/ready")
def ready():
    # Basic check: try a no-op call (do NOT run heavy predictions here)
    return {"status": "ok", "models_loaded": True}
