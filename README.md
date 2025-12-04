# TraceFinder - Forensic Scanner Identification

## 1. Project Statement
The aim of this project is to **identify the source scanner device** used to scan a document or image by analyzing the unique patterns or artifacts left behind during the scanning process.  

Each scanner (brand/model) introduces specific noise, texture, or compression traces that can be learned by a machine learning model.  

This project is important in **forensic investigations**, **copyright authentication**, and **document verification tasks**.

---
## Project Structure
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

## 2. Use Cases

### **1. Digital Forensics**
- **Description:** Determine which scanner was used to forge or duplicate legal documents.  
- **Example:** Detect whether a fake certificate was created using a specific scanner model.

### **2. Document Authentication**
- **Description:** Identify the source of printed and scanned images to detect tampering or fraudulent claims.  
- **Example:** Differentiate between scans from authorized and unauthorized departments.

### **3. Legal Evidence Verification**
- **Description:** Ensure scanned copies submitted in court/legal matters came from known and approved devices.  
- **Example:** Verify that scanned agreements originated from the company’s official scanner.

---

## 3. Expected Outcomes
By the end of this project, students will:
- Understand the concept of **source device identification**.  
- Extract **scanner-specific features** such as noise patterns, frequency domain signals, and artifacts.  
- Train a **classification model** to distinguish among multiple scanners.  
- Evaluate **model accuracy** and visualize **feature importance**.  
- Optionally **deploy a simple app** to upload and identify the scanner source.

---

## 4. Dataset
- **Source:** Kaggle  
*(Additional manually collected samples may be added if needed.)*

---

## 5. System Architecture
*(Insert system architecture diagram here)*

---

## 6. Modules to Be Implemented

### **1. Data Collection & Labeling**
- Manually scan sample images using multiple scanners.  
- Assign proper labels based on source device.

### **2. Image Preprocessing**
- Resize, denoise, and convert to grayscale if needed.  
- Normalize pixels and remove non-artifact content.

### **3. Feature Extraction**
- Extract noise patterns using filters (e.g., Wavelet, FFT).  
- Compute **PRNU**, texture descriptors, and edge patterns.

### **4. Model Training**
- Train classifiers such as:
  - **CNN** (if using deep features)
  - **Random Forest**, **SVM** (if using extracted features)  
- Evaluate performance on validation set.

### **5. Output System**
- Upload an image → Return probable scanner model.  
- Optional: Display **confidence score** and **feature map**.

---

## 7. Week-wise Implementation Roadmap

### **Milestone 1: Dataset Collection & Preprocessing**

#### Week 1:
- Collect scanned document samples from different scanner devices (minimum **3–5 models/brands**).  
- Create a labeled dataset (e.g., `scanner_model`, `file_name`, etc.).  
- Analyze basic image properties such as resolution, format, and color channel.

#### Week 2:
- Perform image preprocessing:
  - Resize all images to a fixed dimension.  
  - Convert to grayscale if needed.  
  - Denoise (optional).  
- Normalize and structure the dataset for model training.

---

### **Milestone 2: Feature Engineering & Baseline Modeling**

#### Week 3:
- Extract hand-crafted features such as:
  - Noise patterns.  
  - Frequency domain features (e.g., FFT).  
  - Texture descriptors (e.g., LBP).  
- Visualize differences between scanner outputs (e.g., noise maps).

#### Week 4:
- Train baseline ML models (e.g., Logistic Regression, SVM, Random Forest).  
- Evaluate using **accuracy** and **confusion matrix**.  
- Log performance and identify limitations of hand-crafted features.

---

### **Milestone 3: Deep Learning Model + Explainability**

#### Week 5:
- Build and train a **CNN model** on the raw image dataset.  
- Use **image augmentation** for generalization (brightness, rotation).  
- Tune **hyperparameters** and track training curves (accuracy/loss).

#### Week 6:
- Evaluate model performance using **accuracy**, **F1-score**, and **confusion matrix**.  
- Apply explainability tools such as **SHAP** or **Grad-CAM** to visualize how the model identifies scanner-specific patterns.

---

### **Milestone 4: Deployment & Final Report**

#### Week 7:
- Create a simple UI using **Streamlit** (or any frontend):
  - Upload scanned image.  
  - Get predicted scanner brand/model with confidence score.  
- Log predictions and allow download of results.

#### Week 8:
- Final documentation and formatting.  
- Add system architecture, training results, model comparison, and screenshots.  
- Prepare final presentation slides and demonstrate the working model.

---

## 8. Evaluation Criteria

### **Completion of Tasks**
- Data collected and labeled correctly.  
- Feature engineering and modeling completed.  
- UI integration and testing done.

### **Model Quality**
- Classification accuracy **above baseline (ideally >85%)**.  
- Model should distinguish between at least **3–5 scanners**.  
- Robustness to image format and scan resolution.

### **Documentation & Demo**
- Clear explanation of methodology.  
- Model performance charts (accuracy, confusion matrix).  
- Insights into feature importance (explainability).

---

## 9. Model Performance – Quantitative Metrics

### **Classification Metrics**
- **Accuracy:** Correct scanner prediction rate.  
- **Precision:** Correct predictions for each scanner class.  
- **Recall:** Ability to detect all true scanner outputs.  
- **F1-Score:** Balance of precision and recall.  
- **Confusion Matrix:** Visualization of misclassification between scanner types.

### **Feature Analysis**
- Use **SHAP** or **Grad-CAM** (if CNN used) to show key areas.  
- Validate that the model is **not biased** toward scan brightness or layout.

---

## 10. Conclusion
This project aims to bridge forensic analysis and machine learning by identifying **scanner device signatures** embedded in scanned images.  

Through advanced feature extraction, classification, and explainable AI, the **TraceFinder** system can significantly aid digital forensics, legal investigations, and document authenticity verification.
