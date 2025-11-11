# TraceFinder - Forensic Scanner Identification

This project implements a machine learning system for identifying the source scanner device used to scan documents by analyzing unique patterns and artifacts left during the scanning process. This tool is valuable for forensic investigations, copyright authentication, and document verification tasks.

## Features

- **Scanner-Specific Feature Extraction**
  - Noise pattern analysis
  - Frequency domain features (FFT)
  - Photo Response Non-Uniformity (PRNU)
  - Texture descriptors
  - Edge pattern analysis

- **Multiple Classification Models**
  - Random Forest classifier
  - Support Vector Machine (SVM)
  - Convolutional Neural Network (CNN)

- **Model Explainability**
  - SHAP values for feature importance
  - Confidence scores
  - Performance visualization

- **Web Interface**
  - Easy-to-use Streamlit interface
  - Upload and analyze scanned documents
  - View confidence scores and predictions

## Project Structure

```
scanned-docs-preprocessing/
├── data/
│   ├── raw/                # Raw scanned document samples
│   ├── processed/          # Preprocessed images
│   ├── features/          # Extracted features
│   └── annotations/       # Labels and metadata
├── src/
│   ├── __init__.py
│   ├── collect_samples.py  # Data collection utilities
│   ├── analyze_images.py   # Image analysis tools
│   ├── features.py        # Feature extraction
│   ├── models.py          # ML model implementations
│   ├── preprocess.py      # Image preprocessing
│   ├── dataset.py         # Dataset management
│   └── utils.py           # Utility functions
├── scripts/
│   ├── generate_sample_data.py
│   ├── train_models.py    # Model training pipeline
│   └── run_preprocess.py
├── notebooks/
│   └── 01_exploration.ipynb
├── tests/
│   └── test_preprocess.py
├── app.py                 # Streamlit web interface
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd scanned-docs-preprocessing
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate     # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Data Collection

```bash
python -m src.collect_samples
```

This will:
- Collect scanned document samples
- Extract scanner-specific features
- Save features and metadata

### 2. Model Training

```bash
python scripts/train_models.py
```

This script will:
- Train multiple models (RF, SVM, CNN)
- Generate performance metrics
- Create visualization plots
- Save results and models

### 3. Web Interface

```bash
streamlit run app.py
```

This will launch the web interface where you can:
- Upload scanned documents
- Get scanner predictions
- View confidence scores
- Analyze feature importance

## Model Performance

The system aims to achieve:
- Classification accuracy > 85%
- Support for 3-5 different scanner models
- Robustness to image format and resolution variations

## Documentation

Detailed documentation is available in the following locations:
- `notebooks/01_exploration.ipynb`: Dataset exploration and analysis
- `output/*/results.json`: Model performance metrics
- `output/*/`: Visualization plots and metrics

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License - see LICENSE file for details