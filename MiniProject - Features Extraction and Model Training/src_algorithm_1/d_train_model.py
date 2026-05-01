import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==============================
# PATHS
# ==============================
DATA_PATH = r"D:\VI\IP & CV\Mini Project\processed_data\landmarks_dataset.csv"
MODEL_PATH = r"D:\VI\IP & CV\Mini Project\models\svm_model_1.pkl"
SCALER_PATH = r"D:\VI\IP & CV\Mini Project\models\scaler_1.pkl"

# ==============================
# LOAD DATA
# ==============================
df = pd.read_csv(DATA_PATH)

X = df.drop("label", axis=1).values
y = df["label"].values

print("Dataset shape:", X.shape)

# ==============================
# TRAIN-TEST SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train samples:", len(X_train))
print("Test samples:", len(X_test))

# ==============================
# FEATURE SCALING (IMPORTANT)
# ==============================
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==============================
# TRAIN SVM MODEL
# ==============================
model = SVC(
    kernel='rbf',
    C=10,
    gamma='scale',
    probability=True
)

print("\nTraining model...")
model.fit(X_train, y_train)

# ==============================
# EVALUATION
# ==============================
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"\n✅ Accuracy: {accuracy * 100:.2f}%\n")

print("Classification Report:\n")
print(classification_report(y_test, y_pred))

# Optional: Confusion matrix
print("Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# ==============================
# SAVE MODEL + SCALER
# ==============================
os.makedirs("../model", exist_ok=True)

joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

print("\n✅ Model saved at:", MODEL_PATH)
print("✅ Scaler saved at:", SCALER_PATH)