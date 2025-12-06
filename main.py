# backend/main.py

# ------------------ IMPORTS ------------------
import os
import uuid
import traceback
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from backend.random_forest_model import RandomForestScannerModel
from backend.cnn_model import CNNScannerModel
from backend.hybrid_cnn_model import HybridCNNScannerModel

# ------------------ APP SETUP ------------------
app = FastAPI(title="AI Trace Finder API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ LOAD MODELS ------------------
rf_model = RandomForestScannerModel()      # Random Forest model
cnn_model = CNNScannerModel()              # CNN model
hybrid_model = HybridCNNScannerModel()     # Hybrid CNN model

# Temporary directory for uploaded files
TMP_DIR = "tmp"
os.makedirs(TMP_DIR, exist_ok=True)

# ------------------ ROUTES ------------------
@app.get("/")
def root():
    """Root endpoint to check API status"""
    return {"message": "AI Trace Finder API is up."}


@app.post("/predict")
async def predict(model_choice: str = Form(...), file: UploadFile = File(...)):
    """
    Predict uploaded file using selected model.
    
    Parameters:
    - model_choice: 'Random Forest', 'CNN Model', 'Hybrid CNN Model'
    - file: uploaded image file
    """
    # ------------------ VALIDATE FILE ------------------
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".tif", ".tiff", ".png", ".jpg", ".jpeg"]:
        raise HTTPException(status_code=400, detail="Invalid file format")

    # Save file temporarily
    tmp_path = os.path.join(TMP_DIR, uuid.uuid4().hex + ext)
    with open(tmp_path, "wb") as f:
        f.write(await file.read())

    # ------------------ PREDICTION ------------------
    try:
        model_choice_lower = model_choice.lower()

        if model_choice_lower == "random forest":
            result = rf_model.predict(tmp_path)
        elif model_choice_lower == "cnn model":
            result = cnn_model.predict(tmp_path)
        elif model_choice_lower == "hybrid cnn model":
            result = hybrid_model.predict(tmp_path)
        else:
            raise HTTPException(
                status_code=400,
                detail="Choose a valid model: Random Forest, CNN Model, Hybrid CNN Model"
            )

        return JSONResponse(content={"prediction": result})

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

    finally:
        # ------------------ CLEANUP ------------------
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
