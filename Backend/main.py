import sys
import os
import pickle
import numpy as np
import cv2
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
import io
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import uvicorn
import shutil
from fastapi import Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import timedelta

# Handle imports for both running from Backend/ or from root
try:
    # Try relative imports (when running from root with uvicorn Backend.main:app)
    from . import models, schemas, auth, database
    from .database import get_db
    from .auth import get_current_user
except ImportError:
    # Fall back to direct imports (when running from Backend/ directory)
    import models, schemas, auth, database
    from database import get_db
    from auth import get_current_user

models.Base.metadata.create_all(bind=database.engine)

# Add src to path to import utils and config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import config
import utils

app = FastAPI(title="Printer Source Identification API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Variables for Models
xgb_model = None
xgb_le = None
cnn_model = None
cnn_le = None
hybrid_model = None
hybrid_le = None
hybrid_scaler = None
scanner_fps = None
fp_keys = None

@app.on_event("startup")
def load_models():
    global xgb_model, xgb_le, cnn_model, cnn_le, hybrid_model, hybrid_le, hybrid_scaler, scanner_fps, fp_keys
    
    # Load XGBoost
    if os.path.exists(config.XGB_MODEL_PATH):
        print("Loading XGBoost...")
        with open(config.XGB_MODEL_PATH, "rb") as f:
            xgb_model = pickle.load(f)
        with open(config.LABEL_ENCODER_PATH, "rb") as f:
            xgb_le = pickle.load(f)
            
    # Load CNN
    if os.path.exists(config.CNN_MODEL_PATH):
        print("Loading CNN...")
        cnn_model = tf.keras.models.load_model(config.CNN_MODEL_PATH, compile=False)
        cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        with open(config.CNN_LABEL_ENCODER_PATH, "rb") as f:
            cnn_le = pickle.load(f)

    # Load Hybrid
    if os.path.exists(config.HYBRID_MODEL_PATH):
        print("Loading Hybrid...")
        hybrid_model = tf.keras.models.load_model(config.HYBRID_MODEL_PATH, compile=False)
        hybrid_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        with open(config.HYBRID_LABEL_ENCODER_PATH, "rb") as f:
            hybrid_le = pickle.load(f)
        with open(config.HYBRID_SCALER_PATH, "rb") as f:
            hybrid_scaler = pickle.load(f)
        with open(config.SCANNER_FINGERPRINTS_PATH, "rb") as f:
            scanner_fps = pickle.load(f)
        fp_keys = np.load(config.FP_KEYS_PATH, allow_pickle=True).tolist()

@app.get("/")
def read_root():
    return {"message": "Printer Source Identification API is running"}

def save_upload_file(upload_file: UploadFile, destination: str):
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    try:
        with open(destination, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
    finally:
        upload_file.file.close()

@app.post("/predict/xgboost")
async def predict_xgboost(file: UploadFile = File(...), current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not xgb_model:
        raise HTTPException(status_code=503, detail="XGBoost model not loaded")
    
    temp_file = f"temp/temp_{file.filename}"
    save_upload_file(file, temp_file)
    
    try:
        img = utils.load_gray(temp_file)
        if img is None:
             raise HTTPException(status_code=400, detail="Invalid image")
             
        # Extract features (dummy class/density as they are not needed for prediction if we ignore them or handle properly)
        # Note: The training script used 'scanner_folder' and 'dpi' which are not available at inference.
        # We need to adapt extract_features to handle this or pass dummy values.
        # For now, passing dummy values.
        features = utils.extract_features(img, temp_file, "unknown", "unknown")
        
        # Create DataFrame with same columns as training (excluding target)
        # Training columns: file_name, pixel_density, width, height, aspect_ratio, file_size_kb, mean_intensity, ...
        # The training script dropped: file_name, width, height, aspect_ratio, class_label
        # So we need: pixel_density, file_size_kb, mean_intensity, ...
        
        # IMPORTANT: The training script used 'pixel_density' as a feature. 
        # If we don't know it, we might have issues. 
        # Assuming the user might provide it or we try to infer from metadata, but for now let's use a default or try to parse from filename if possible.
        # The notebook parsed it from path. Here we might not have it.
        # Let's assume 150 as a fallback or 0.
        
        # Re-creating the feature vector expected by the model
        # We need to match the columns X_train had.
        # X = df.drop(columns=['file_name', 'width', 'height', 'aspect_ratio', 'class_label'])
        # Columns: pixel_density, file_size_kb, mean_intensity, std_intensity, skewness, kurtosis, entropy, edge_density
        
        # We need to ensure pixel_density is numeric. utils.extract_features passes it as is.
        # If it's "unknown", it will fail.
        # Let's try to extract from filename or default to 0.
        pixel_density = 0
        
        feature_vector = [
            pixel_density,
            features['file_size_kb'],
            features['mean_intensity'],
            features['std_intensity'],
            features['skewness'],
            features['kurtosis'],
            features['entropy'],
            features['edge_density']
        ]
        
        # Predict
        pred_idx = xgb_model.predict([feature_vector])[0]
        pred_label = xgb_le.inverse_transform([pred_idx])[0]
        
        # Confidence (XGBoost predict_proba)
        probs = xgb_model.predict_proba([feature_vector])[0]
        confidence = float(probs[pred_idx])
        
        # Save Prediction
        db_prediction = models.Prediction(
            filename=file.filename,
            model_used="XGBoost",
            prediction_result=str(pred_label),
            confidence=confidence,
            user_id=current_user.id
        )
        db.add(db_prediction)
        db.commit()

        return {"model": "XGBoost", "prediction": str(pred_label), "confidence": confidence}
        
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)



@app.post("/predict/cnn")

async def predict_cnn(file: UploadFile = File(...), current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not cnn_model:
        raise HTTPException(status_code=503, detail="CNN model not loaded")
        
    temp_file = f"temp/temp_{file.filename}"
    save_upload_file(file, temp_file)
    
    try:
        print(f"Processing CNN for file: {temp_file}")
        res = utils.preprocess_image(temp_file, size=(512, 512))
        if res is None:
            print("Preprocessing returned None")
            raise HTTPException(status_code=400, detail="Invalid image")
            
        print(f"Preprocessed shape: {res.shape}")
        res = res.reshape(1, 512, 512, 1)
        print(f"Reshaped for model: {res.shape}")
        
        # cnn_model.summary()
        
        probs = cnn_model.predict(res)[0]
        print(f"Prediction probs: {probs}")
        pred_idx = np.argmax(probs)
        pred_label = cnn_le.inverse_transform([pred_idx])[0]
        confidence = float(probs[pred_idx])
        print(f"Predicted: {pred_label}, Confidence: {confidence}")
        
        # Save Prediction
        db_prediction = models.Prediction(
            filename=file.filename,
            model_used="CNN",
            prediction_result=str(pred_label),
            confidence=confidence,
            user_id=current_user.id
        )
        db.add(db_prediction)
        db.commit()

        return {"model": "CNN", "prediction": str(pred_label), "confidence": confidence}
        
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

@app.post("/predict/hybrid")
async def predict_hybrid(file: UploadFile = File(...), current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not hybrid_model:
        raise HTTPException(status_code=503, detail="Hybrid model not loaded")
        
    temp_file = f"temp/temp_{file.filename}"
    save_upload_file(file, temp_file)
    
    try:
        # Preprocess Image (Residual)
        res = utils.preprocess_image(temp_file)
        if res is None:
            raise HTTPException(status_code=400, detail="Invalid image")
            
        # Extract Handcrafted Features
        v_corr = [utils.corr2d(res, scanner_fps[k]) for k in fp_keys]
        v_fft  = utils.fft_radial_energy(res)
        v_lbp  = utils.lbp_hist_safe(res)
        feat_vec = v_corr + v_fft + v_lbp
        
        # Scale Features
        feat_vec = np.array(feat_vec).reshape(1, -1)
        feat_vec_scaled = hybrid_scaler.transform(feat_vec)
        
        # Prepare Inputs
        img_in = res.reshape(1, 256, 256, 1)
        
        # Predict
        probs = hybrid_model.predict([img_in, feat_vec_scaled])[0]
        pred_idx = np.argmax(probs)
        pred_label = hybrid_le.inverse_transform([pred_idx])[0]
        confidence = float(probs[pred_idx])
        
        # Save Prediction
        db_prediction = models.Prediction(
            filename=file.filename,
            model_used="Hybrid CNN",
            prediction_result=str(pred_label),
            confidence=confidence,
            user_id=current_user.id
        )
        db.add(db_prediction)
        db.commit()

        return {"model": "Hybrid CNN", "prediction": str(pred_label), "confidence": confidence}
        
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

@app.post("/signup", response_model=schemas.User)
def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    try:
        db_user = db.query(models.User).filter(models.User.username == user.username).first()
        if db_user:
            raise HTTPException(status_code=400, detail="Username already registered")
        hashed_password = auth.get_password_hash(user.password)
        db_user = models.User(username=user.username, email=user.email, hashed_password=hashed_password)
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        return db_user
    except Exception as e:
        with open("logs/debug_log.txt", "a") as log:
            import traceback
            log.write(f"Signup Error: {e}\n")
            traceback.print_exc(file=log)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/token", response_model=schemas.Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.username == form_data.username).first()
    if not user or not auth.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=auth.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = auth.create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = auth.jwt.decode(token, auth.SECRET_KEY, algorithms=[auth.ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = schemas.TokenData(username=username)
    except auth.JWTError:
        raise credentials_exception
    user = db.query(models.User).filter(models.User.username == token_data.username).first()
    if user is None:
        raise credentials_exception
    return user

@app.get("/users/me", response_model=schemas.User)
async def read_users_me(current_user: models.User = Depends(get_current_user)):
    return current_user

@app.get("/users/me/stats")
async def read_user_stats(current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    total_scans = db.query(models.Prediction).filter(models.Prediction.user_id == current_user.id).count()
    return {"total_scans": total_scans}

@app.get("/users/me/history", response_model=List[schemas.Prediction])
async def read_user_history(current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    history = db.query(models.Prediction).filter(models.Prediction.user_id == current_user.id).order_by(models.Prediction.timestamp.desc()).all()
    return history

@app.post("/utils/convert-preview")
async def convert_preview(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            # Try reading with tifffile or other methods if cv2 fails, but cv2 usually handles tiff
            # If it's a multi-page tiff, cv2 might only read the first page which is fine for preview
            raise HTTPException(status_code=400, detail="Could not decode image")

        # Resize for preview if too large (optional, but good for performance)
        height, width = img.shape[:2]
        max_dim = 800
        if height > max_dim or width > max_dim:
            scale = max_dim / max(height, width)
            img = cv2.resize(img, None, fx=scale, fy=scale)

        # Encode as PNG
        _, buffer = cv2.imencode(".png", img)
        return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/png")
    except Exception as e:
        print(f"Error converting preview: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
