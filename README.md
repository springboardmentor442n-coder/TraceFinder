# DocumentFounder - Printer Source Identification

A machine learning application for identifying the source printer of printed documents using advanced image processing and deep learning techniques.

## 🌟 Features

- **Multi-Model Prediction**: Choose from XGBoost, CNN, or Hybrid CNN models
- **Advanced Image Processing**: Flatfield correction and residual analysis
- **FastAPI Backend**: High-performance REST API for predictions
- **React Frontend**: Modern, responsive web interface
- **User Authentication**: Secure login and registration system
- **MySQL Database**: Robust data storage and management

## 🛠️ Tech Stack

### Backend
- **FastAPI** - Modern Python web framework
- **TensorFlow/Keras** - Deep learning models (CNN, Hybrid CNN)
- **XGBoost** - Gradient boosting classifier
- **OpenCV** - Image processing
- **scikit-learn** - Feature extraction and preprocessing
- **MySQL** - Database (via SQLAlchemy)

### Frontend
- **React** - UI framework
- **Vite** - Build tool and dev server
- **React Router** - Client-side routing
- **Axios** - HTTP requests

## 📋 Prerequisites

- Python 3.8+
- Node.js 16+
- MySQL Server (optional for full functionality)

## 🚀 Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/DocumentFounder.git
cd DocumentFounder
```

### 2. Backend Setup
```bash
cd Backend
pip install -r requirements.txt
```

### 3. Frontend Setup
```bash
cd Frontend
npm install
```

### 4. Database Setup (Optional)
```bash
# Run the MySQL setup script
python scripts/setup_mysql_db.py
```

## 🎮 Usage

### Start the Backend
```bash
cd Backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### Start the Frontend
```bash
cd Frontend
npm run dev
```

The web interface will be available at `http://localhost:5173`

## 📁 Project Structure

```
DocumentFounder/
├── Backend/              # FastAPI backend
│   ├── main.py          # Main application & endpoints
│   ├── models.py        # Database models
│   ├── schemas.py       # Pydantic schemas
│   ├── auth.py          # Authentication logic
│   └── requirements.txt # Python dependencies
├── Frontend/            # React frontend
│   ├── src/
│   │   ├── components/  # Reusable components
│   │   ├── pages/       # Page components
│   │   └── main.jsx     # Entry point
│   └── package.json     # Node dependencies
├── models/              # Trained ML models
│   ├── xgboost_model.pkl
│   ├── scanner_hybrid_final.keras
│   └── cnn_residual_model.keras
├── scripts/             # Utility scripts
│   └── setup_mysql_db.py
├── src/                 # Shared utilities
│   ├── config.py        # Configuration
│   └── utils.py         # Image processing utilities
└── .gitignore
```

## 🔬 Models

### 1. XGBoost Classifier
- Traditional ML approach using gradient boosting
- Fast inference time
- Good for baseline comparisons

### 2. CNN Residual Model
- Convolutional Neural Network trained on flatfield residuals
- Captures printer-specific artifacts
- High accuracy for known printers

### 3. Hybrid CNN Model
- Combines image features and metadata
- Best overall performance
- Robust to variations in print quality

## 📊 API Endpoints

- `POST /predict/xgboost` - Predict using XGBoost model
- `POST /predict/cnn` - Predict using CNN model
- `POST /predict/hybrid` - Predict using Hybrid model
- `POST /register` - User registration
- `POST /login` - User authentication
- `GET /history` - Get prediction history

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 👥 Authors

- Your Name

## 🙏 Acknowledgments

- Dataset and research methodology based on printer forensics literature
- Built with modern ML and web development best practices
