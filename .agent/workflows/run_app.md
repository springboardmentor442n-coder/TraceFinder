---
description: How to run the Printer Source Identification Application
---

# Run the Application

Follow these steps to run the full-scale application.

## 1. Install Dependencies

### Backend
```bash
pip install -r Backend/requirements.txt
```

### Frontend
```bash
cd Frontend
npm install
npm install react-dropzone
cd ..
```

## 2. Train Models (Optional if models already exist)

Run the training scripts in order:

```bash
python src/train_xgboost.py
python src/train_cnn.py
python src/train_hybrid.py
```

## 3. Start the Application

### Start Backend Server
Open a terminal and run (choose one method):

**Method 1 (Recommended):**
```bash
cd Backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Method 2 (From root directory):**
```bash
uvicorn Backend.main:app --reload --host 0.0.0.0 --port 8000
```

### Start Frontend Client
Open another terminal and run:
```bash
cd Frontend
npm run dev
```

## 4. Access the App
Open your browser and navigate to the URL shown in the Frontend terminal (usually `http://localhost:5173`).
