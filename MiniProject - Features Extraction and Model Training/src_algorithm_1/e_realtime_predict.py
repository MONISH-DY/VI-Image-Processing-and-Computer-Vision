import cv2
import numpy as np
import joblib
from collections import deque, Counter
import mediapipe as mp

from b_feature_extraction import extract_landmark_features

# ==============================
# LOAD MODEL + SCALER
# ==============================
model = joblib.load(r"D:\VI\IP & CV\Mini Project\models\svm_model_1.pkl")
scaler = joblib.load(r"D:\VI\IP & CV\Mini Project\models\scaler_1.pkl")

# ==============================
# MEDIAPIPE SETUP
# ==============================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

# ==============================
# STABILITY BUFFER
# ==============================
buffer = deque(maxlen=10)

# ==============================
# CONFIDENCE THRESHOLD
# ==============================
CONF_THRESHOLD = 0.5

# ==============================
# WEBCAM
# ==============================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Camera not accessible")
    exit()

print("Press ESC to exit")

# ==============================
# MAIN LOOP
# ==============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    display_text = "No Hand"

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            # DRAW HAND
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # ==============================
            # FEATURE EXTRACTION (ALGO A)
            # ==============================
            features = extract_landmark_features(hand_landmarks)

            # Scale
            features = scaler.transform([features])

            # ==============================
            # CONFIDENCE PREDICTION
            # ==============================
            probs = model.predict_proba(features)[0]

            max_index = np.argmax(probs)
            prediction = model.classes_[max_index]
            confidence = probs[max_index]

            # ==============================
            # APPLY CONFIDENCE FILTER
            # ==============================
            if confidence > CONF_THRESHOLD:
                buffer.append(prediction)
                most_common = Counter(buffer).most_common(1)[0][0]

                display_text = f"{most_common} ({confidence:.2f})"
            else:
                display_text = "Low Confidence"

    # ==============================
    # DISPLAY
    # ==============================
    cv2.putText(frame, f"Prediction: {display_text}",
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3)

    cv2.imshow("ASL Real-Time Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# ==============================
# CLEANUP
# ==============================
cap.release()
cv2.destroyAllWindows()