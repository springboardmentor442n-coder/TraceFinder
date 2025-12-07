# train_model.py

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import joblib

from preprocessing import load_and_preprocess  # import preprocessing file

# ---------------------------
# File Paths
# ---------------------------
csv_path = r"D:\AI_TraceFinder\metadata_all.csv"
model_path = r"D:\AI_TraceFinder\ai_trace_finder_model.pkl"
target_column = "class_label"

# ---------------------------
# 1. Load + Preprocess
# ---------------------------
X, y = load_and_preprocess(csv_path, target_column)

# ---------------------------
# 2. Train/Test Split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# 3. Train Random Forest
# ---------------------------
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    random_state=42
)

model.fit(X_train, y_train)
print("Random Forest Trained!")

# ---------------------------
# 4. Predictions + Evaluation
# ---------------------------
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f" Accuracy: {acc * 100:.2f}%")
print("\n Classification Report:\n", classification_report(y_test, y_pred))

# ---------------------------
# 5. Confusion Matrix
# ---------------------------
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(xticks_rotation=45)
plt.title("Confusion Matrix")
plt.show()

# ---------------------------
# 6. Feature Importance
# ---------------------------
importances = model.feature_importances_
indices = importances.argsort()[::-1]
feature_names = X.columns

plt.figure(figsize=(12, 6))
plt.title("Feature Importances")
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), feature_names[indices], rotation=90)
plt.tight_layout()
plt.show()

# ---------------------------
# 7. Save Model
# ---------------------------
joblib.dump(model, model_path)
print(" Model saved at:", model_path)
