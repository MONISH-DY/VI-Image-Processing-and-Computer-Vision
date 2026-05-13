# Observation — ASL Sign Language to Text Recognition System

### Abstract
This project implements a real-time **ASL Sign Language to Text Recognition System** using a hybrid pipeline that combines **MediaPipe Hands** for landmark extraction and **Support Vector Machines (SVM)** for classification. The system processes live webcam feeds, extracts 2D/3D hand landmarks, and classifies them into ASL letters. A multi-stage architecture is used: hand detection and landmark localization via MediaPipe, followed by advanced geometric feature engineering (distances, angles, and states), and finally classification via an SVM model with an RBF kernel. The system supports dual algorithms (Raw Landmarks vs. Engineered Features) and achieves approximately **98% accuracy**. It is served through a **Flask-based dashboard** with real-time MJPEG streaming and a word-formation engine with stability buffering.

### Models Used
| # | Model / Algorithm | Type | Role in Pipeline |
|---|---|---|---|
| 1 | **MediaPipe Hands** | Pre-trained ML (BlazePalm) | Detects hand bounding boxes and extracts 21 precise 3D hand landmarks in each frame. |
| 2 | **Geometric Feature Engine** | Rule-based CV | Computes secondary features like inter-finger distances, joint angles, and finger states (open/closed). |
| 3 | **Support Vector Machine (SVM)** | Classical Machine Learning | Performs multi-class classification on normalized hand features to predict ASL letters. |
| 4 | **Stability Buffer (Voting)** | Logic Layer | Uses a deque-based majority voting system to filter out transient noise and ensure stable predictions. |
| 5 | **Word Formation Engine** | State Logic | Implements cooldown and stability checks to concatenate predicted letters into meaningful words and sentences. |

### How the Model is Optimized
The core classification system utilizes a **Support Vector Machine (SVM)** optimized for high-dimensional feature spaces with the following approach:

**1. Feature Normalization & Invariance**
* **Translation Invariance:** All raw landmarks are translated to a local coordinate system with the wrist as the origin (0,0,0).
* **Scale Invariance:** Features are normalized by the hand's bounding box size, ensuring the model works regardless of how close the hand is to the camera.
* **Standardization:** A `StandardScaler` is applied to ensure all features (coordinates, angles, and distances) have a mean of 0 and a standard deviation of 1.

**2. Feature Engineering (Algorithm 2)**
* Instead of relying solely on raw coordinates, Algorithm 2 computes **~30 geometric features**:
    * **Euclidean Distances:** Distance between fingertips (e.g., Thumb to Index).
    * **Bending Angles:** The angle of flexion at each knuckle joint.
    * **Binary States:** Boolean flags indicating whether specific fingers are "extended" or "folded."
* This significantly improves class separability for similar-looking signs like 'M', 'N', and 'T'.

**3. Decision Boundary Optimization**
* An **RBF (Radial Basis Function)** kernel is used to handle non-linear decision boundaries between complex hand gestures.
* The model is trained with a regularization parameter **C=10**, balancing the trade-off between a smooth decision boundary and classifying training points correctly.

### Training Configuration
| Parameter | Value |
|---|---|
| **Algorithm** | Support Vector Machine (SVM) |
| **Kernel** | RBF (Radial Basis Function) |
| **Regularization (C)** | 10.0 |
| **Gamma** | 'scale' |
| **Feature Count** | 63 (Algo 1) / ~90 (Algo 2) |
| **Train/Test Split** | 80/20 |
| **Validation Accuracy** | ~98.1% |
| **Input Type** | Landmark Coordinate Vectors (Normalized) |

### Why This Approach Works
* **Computational Efficiency:** By using landmarks instead of raw pixel data (CNNs), the input dimensionality is reduced from millions of pixels to just ~90 features. This allows the system to run at high FPS even on standard CPUs without a GPU.
* **Feature Essence:** Geometric features (angles/distances) capture the "anatomy" of a sign rather than just its visual appearance, making it more robust to different hand sizes and orientations.
* **MediaPipe Robustness:** Leveraging MediaPipe's pre-trained hand tracker provides stable landmark detection across various lighting conditions and backgrounds, which would be difficult to train from scratch with a small dataset.
* **Stability Buffer:** The majority voting logic prevents "flickering" between predictions, ensuring that only confident and stable gestures are translated into words.
