# 🖐️ ASL Sign Language to Text (Real-Time)

### MediaPipe Hands + Landmark-Based SVM (Algorithm 1)

---

## 📌 Overview

This project implements a real-time **American Sign Language (ASL) recognition system** using **MediaPipe hand landmarks** and a **Support Vector Machine (SVM)** classifier.

The system captures hand gestures from a webcam, extracts **21 hand landmarks**, and classifies them into ASL letters using a trained ML model.

---

## 🚀 Algorithm Used (Algorithm 1)

### 🔷 Pipeline

Camera → MediaPipe Hands → Landmark Extraction → SVM → Prediction

---

## 🧠 Core Idea

Instead of complex feature engineering, this system uses:

* Raw **hand landmarks (x, y, z)**
* Normalization (translation + scaling)
* SVM for classification

---

## 📊 Features

* 21 landmarks × 3 coordinates = **63 features**
* No distance/angle features (pure landmark-based model)

---

## 📂 Project Structure

```
src_algorithm_1/
│
├── a_subset_creation.py      # Creates dataset subset
├── b_feature_extraction.py  # Extracts 63 landmark features
├── c_build_dataset.py       # Converts images → dataset.csv
├── d_train_model.py         # Trains SVM model
├── e_realtime_predict.py    # Real-time prediction using webcam
```

---

## ⚙️ Installation

```bash
pip install opencv-python mediapipe numpy pandas scikit-learn joblib
```

---

## 🏗️ Workflow

### 1️⃣ Subset Creation

```bash
python a_subset_creation.py
```

* Selects representative images from dataset
* Reduces redundancy

---

### 2️⃣ Build Dataset

```bash
python c_build_dataset.py
```

* Uses MediaPipe to extract landmarks
* Generates dataset with **63 features + labels**

---

### 3️⃣ Train Model

```bash
python d_train_model.py
```

* Trains SVM classifier
* Uses:

  * RBF kernel
  * StandardScaler
  * probability=True (for confidence)

---

### 4️⃣ Real-Time Prediction

```bash
python e_realtime_predict.py
```

* Captures webcam feed
* Detects hand landmarks
* Predicts ASL letters in real-time
* Displays:

  * Hand skeleton
  * Predicted letter
  * Confidence score

---

## 🎯 Output

Example:

```
Prediction: A (0.92)
```

or

```
Low Confidence
```

---

## 📈 Performance

| Metric   | Value     |
| -------- | --------- |
| Accuracy | ~97%      |
| Features | 63        |
| Model    | SVM (RBF) |

---

## 🧠 Techniques Used

* MediaPipe Hands (landmark detection)
* Landmark normalization
* Support Vector Machine (SVM)
* Confidence filtering
* Majority voting (stability buffer)

---

## ⚠️ Limitations

* Sensitive to lighting conditions
* Similar gestures (M, N) may confuse
* Static model (no motion detection for J/Z)

---

## 🔮 Future Enhancements

* Word formation system
* Temporal modeling (LSTM)
* Gesture smoothing improvements
* Mobile/Web deployment

---

## 👨‍💻 Author

Monish D.Y.
B.E. Computer Science and Engineering

---

## 📌 Note

This project demonstrates a **lightweight and efficient ASL recognition system** using only landmark-based features, achieving high accuracy without complex feature engineering.

---
