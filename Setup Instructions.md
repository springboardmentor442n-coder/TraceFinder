# AI Trace Finder - Setup Instructions

## 1. Create Virtual Environment
Create your own virtual environment in the project root:
```
python -m venv .venv
```

## 2. Activate Virtual Environment
- **Windows**:
```
.venv\Scripts\activate
```
- **Linux / Mac**:
```
source .venv/bin/activate
```

## 3. Install Requirements
Install all necessary packages:
```
pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Start Backend Server (FastAPI)
```
uvicorn backend.main:app --reload
```
- Runs on: http://127.0.0.1:8000  
- Ensure all `.pkl` model files are in the `models/` folder.

## 5. Open Frontend Application (Streamlit)
Open a **new terminal** and run:
```
streamlit run frontend/app.py
```
- Frontend will connect to backend automatically if it’s running.  
- Use the UI to test AI Trace Finder functionalities.

## Notes
- TensorFlow 2.13 requires `typing-extensions==4.15.0`.  
- Keep all model `.pkl` files in the `models/` folder.  
- If backend fails due to version conflicts, recreate `.venv` and reinstall requirements.  
- Temporary files are in `tmp/` and can be deleted safely.  

## Submission
- Submit the entire `AI_TRACE_FINDER/` folder.  
- `.venv/` is included for testing purposes but can be recreated if necessary.  
- This demonstrates full effort, though some backend models may require exact version setup to run.  
