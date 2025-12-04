import os
import uuid
from fastapi import FastAPI, UploadFile, File, Form, HTTPException

from .models_xgb import XGBScannerModel
from .models_cnn import CNNScannerModel

app = FastAPI(title="Scanner API")

xgb = XGBScannerModel()
cnn = CNNScannerModel()

TMP_DIR = "tmp"
os.makedirs(TMP_DIR, exist_ok=True)

@app.post("/predict")
async def predict(model_choice: str = Form(...), file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()

    if ext not in [".tif", ".tiff", ".png", ".jpg", ".jpeg"]:
        raise HTTPException(status_code=400, detail="Invalid file format")

    tmp_path = os.path.join(TMP_DIR, uuid.uuid4().hex + ext)
    with open(tmp_path, "wb") as f:
        f.write(await file.read())

    try:
        if model_choice.lower() == "xgboost":
            result = xgb.predict(tmp_path)
        elif model_choice.lower() == "cnn":
            result = cnn.predict(tmp_path)
        else:
            raise HTTPException(status_code=400, detail="Choose xgboost or cnn")
    finally:
        os.remove(tmp_path)

    return result