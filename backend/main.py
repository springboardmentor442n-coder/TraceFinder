from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
import os, shutil

from backend.models import (   # ✅ Use this if running as module
    registry,
    predict_cnn,
    predict_svm,
    predict_rf,
    predict_hybrid
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.on_event("startup")
def load_all_models():
    print("🚀 Loading models…")
    registry.load_all()


@app.post("/predict")
async def predict(
    model: str = Query(...),
    file: UploadFile = File(...)
):
    temp_path = "temp_input_image.tif"

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # ---- route to correct model ----
    if model == "cnn":
        pred, conf = predict_cnn(temp_path)

    elif model == "svm":
        pred, conf = predict_svm(temp_path)

    elif model == "rf":
        pred, conf = predict_rf(temp_path)

    elif model == "hybrid":
        pred, conf = predict_hybrid(temp_path)

    else:
        return {"error": "Invalid model name"}

    os.remove(temp_path)

    return {"prediction": pred, "confidence": conf}
