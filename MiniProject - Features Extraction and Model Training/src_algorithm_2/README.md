# 🖐️ ASL Sign Language to Text (Real-Time)

### MediaPipe Hands + Feature-Engineered SVM (Algorithm 2)

---

## 📌 Overview

This project implements a real-time **American Sign Language (ASL) recognition system** using computer vision and machine learning.

The system captures hand gestures through a webcam, extracts hand landmarks using MediaPipe, applies feature engineering, and classifies gestures into ASL letters using a **Support Vector Machine (SVM)**.

---

## 🚀 Algorithm Used

### 🔷 Pipeline

Camera → MediaPipe Hands → Landmark Extraction → Feature Engineering → SVM → Prediction

---

## 🧠 Core Idea

The model enhances raw hand landmarks by computing **geometric relationships** such as:

* distances between fingers
* angles between joints
* finger states (open/closed)

This improves class separability and overall accuracy.

---

## 📊 Features

Total features: **~90**

### 🔹 Landmark Features (63)

* 21 hand landmarks × (x, y, z)

### 🔹 Distance Features

* Fingertip-to-fingertip distances
* Wrist-to-fingertip distances

### 🔹 Angle Features

* Finger bending angles
* Joint orientation

### 🔹 Finger State Features

* Binary values indicating finger positions

---

## ⚙️ Preprocessing

* Translation normalization (wrist as origin)
* Scale normalization (size invariance)
* StandardScaler (mean = 0, std = 1)

---

## 🧠 Model

* Algorithm: **Support Vector Machine (SVM)**
* Kernel: RBF (non-linear)
* Probability enabled for confidence scoring

---

## 📂 Project Structure

```id="projstruct"
src_algorithm_2/
│
├── a_subset_creation.py      # Dataset sampling
├── b_feature_extraction.py  # Feature engineering
├── c_build_dataset.py       # Dataset creation
├── d_train_model.py         # Model training
├── e_realtime_predict.py    # Real-time prediction
├── f_word_predict.py        # Word formation
└── README.md
```

---

## ⚙️ Installation

```bash id="installcmd"
pip install opencv-python mediapipe numpy pandas scikit-learn joblib
```

---

## 🏗️ Workflow

### 1️⃣ Create Dataset Subset

```bash id="step1cmd"
python a_subset_creation.py
```

---

### 2️⃣ Build Dataset

```bash id="step2cmd"
python c_build_dataset.py
```

* Extracts ~90 features per image
* Saves dataset for training

---

### 3️⃣ Train Model

```bash id="step3cmd"
python d_train_model.py
```

* Trains SVM model
* Saves:

  * `svm_model.pkl`
  * `scaler.pkl`

---

### 4️⃣ Real-Time Prediction

```bash id="step4cmd"
python e_realtime_predict.py
```

* Detects hand
* Displays:

  * Hand skeleton
  * Predicted letter
  * Confidence score

---

### 5️⃣ Word Formation

```bash id="step5cmd"
python f_word_predict.py
```

* Converts letters → words
* Uses:

  * stability buffer
  * cooldown logic
  * space/delete gestures

---

## 🎯 Output

```id="exampleout"
Prediction: A (0.92)
Word: HELLO
```

---

## 📈 Performance

| Metric   | Value     |
| -------- | --------- |
| Accuracy | ~97–98%   |
| Features | ~90       |
| Model    | SVM (RBF) |

---

## 🧠 Techniques Used

* MediaPipe Hands (landmark detection)
* Feature engineering (distance + angle + states)
* Support Vector Machine (SVM)
* Kernel Trick (RBF kernel)
* Confidence filtering
* Majority voting (stability buffer)
* State-based word formation

---

## ⚠️ Limitations

* Sensitive to lighting conditions
* Similar gestures (M, N) may confuse
* Limited support for dynamic gestures (J, Z)

---

## 🔮 Future Improvements

* Deep learning models (CNN/LSTM)
* Gesture sequence modeling
* Language correction system
* Mobile/web deployment

---

## 🧠 Key Learnings

* Feature engineering significantly improves performance
* Geometric relationships enhance classification
* SVM works well for structured feature data
* Real-time systems require smoothing and filtering

---

## 👨‍💻 Author

Monish D.Y.
B.E. Computer Science and Engineering

---

## 📌 Conclusion

This project demonstrates how combining **computer vision, feature engineering, and machine learning** enables an efficient and accurate real-time ASL recognition system.

---
