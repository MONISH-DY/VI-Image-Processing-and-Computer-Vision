import cv2
import joblib
import mediapipe as mp
from collections import deque, Counter
import time

from b_feature_extraction import extract_features_from_landmarks

# ==============================
# LOAD MODEL
# ==============================
model = joblib.load(r"D:\VI\IP & CV\Mini Project\models\svm_model_2.pkl")
scaler = joblib.load(r"D:\VI\IP & CV\Mini Project\models\scaler_2.pkl")

# ==============================
# MEDIAPIPE
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
# BUFFERS
# ==============================
buffer = deque(maxlen=10)

# ==============================
# WORD LOGIC VARIABLES
# ==============================
current_word = ""
last_letter = ""
last_added_time = 0

COOLDOWN = 1.2   # seconds between letters
STABILITY_THRESHOLD = 7  # frames

# ==============================
# WEBCAM
# ==============================
cap = cv2.VideoCapture(0)

print("Press ESC to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    display_letter = "None"

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            # Draw skeleton
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract features
            features = extract_features_from_landmarks(hand_landmarks)
            features = scaler.transform([features])

            prediction = model.predict(features)[0]

            buffer.append(prediction)

            # Stable prediction
            most_common, count = Counter(buffer).most_common(1)[0]

            display_letter = most_common

            # ==============================
            # LETTER COMMIT LOGIC
            # ==============================
            current_time = time.time()

            if count >= STABILITY_THRESHOLD:
                
                # Avoid repeating same letter
                if most_common != last_letter and (current_time - last_added_time) > COOLDOWN:

                    if most_common == "space":
                        current_word += " "

                    elif most_common == "del":
                        current_word = current_word[:-1]

                    else:
                        current_word += most_common

                    last_letter = most_common
                    last_added_time = current_time

    # ==============================
    # DISPLAY
    # ==============================
    cv2.putText(frame, f"Letter: {display_letter}",
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0), 3)

    cv2.putText(frame, f"Word: {current_word}",
                (30, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (255, 0, 0), 3)

    cv2.imshow("ASL Word Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()