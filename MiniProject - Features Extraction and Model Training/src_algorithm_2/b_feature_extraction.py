import cv2
import numpy as np
import mediapipe as mp

# ==============================
# MEDIAPIPE SETUP (TWO MODES)
# ==============================
mp_hands = mp.solutions.hands

# For dataset (high accuracy)
mp_hands_static = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

# For real-time (stable tracking)
mp_hands_video = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

# ==============================
# HELPER FUNCTIONS
# ==============================

def get_landmarks(image, mode="static"):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if mode == "static":
        result = mp_hands_static.process(img_rgb)
    else:
        result = mp_hands_video.process(img_rgb)

    if result.multi_hand_landmarks is None:
        return None

    landmarks = result.multi_hand_landmarks[0]

    coords = []
    for lm in landmarks.landmark:
        coords.append([lm.x, lm.y, lm.z])

    return np.array(coords)


def normalize_landmarks(coords):
    coords = coords - coords[0]

    max_val = np.max(np.abs(coords))
    if max_val != 0:
        coords = coords / max_val

    return coords


def compute_distance(p1, p2):
    return np.linalg.norm(p1 - p2)


def compute_angle(a, b, c):
    ba = a - b
    bc = c - b

    cos_angle = np.dot(ba, bc) / (
        np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6
    )

    return np.arccos(np.clip(cos_angle, -1.0, 1.0))


# ==============================
# MAIN FEATURE FUNCTION
# ==============================

def extract_features(image, mode="static"):
    coords = get_landmarks(image, mode)

    if coords is None:
        return None

    coords = normalize_landmarks(coords)

    features = []

    # 1. LANDMARKS (63)
    features.extend(coords.flatten())

    # 2. DISTANCES
    fingertips = [4, 8, 12, 16, 20]

    for i in range(len(fingertips)):
        for j in range(i + 1, len(fingertips)):
            d = compute_distance(coords[fingertips[i]], coords[fingertips[j]])
            features.append(d)

    for tip in fingertips:
        d = compute_distance(coords[0], coords[tip])
        features.append(d)

    # 3. ANGLES
    finger_joints = [
        (0, 5, 8),
        (0, 9, 12),
        (0, 13, 16),
        (0, 17, 20)
    ]

    for a, b, c in finger_joints:
        features.append(compute_angle(coords[a], coords[b], coords[c]))

    thumb_joints = [
        (0, 2, 4),
        (1, 2, 3),
        (2, 3, 4)
    ]

    for a, b, c in thumb_joints:
        features.append(compute_angle(coords[a], coords[b], coords[c]))

    # 4. FINGER STATES
    finger_states = []

    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]

    for tip, pip in zip(tips, pips):
        finger_states.append(1 if coords[tip][1] < coords[pip][1] else 0)

    # Thumb
    finger_states.append(1 if coords[4][0] > coords[3][0] else 0)

    features.extend(finger_states)

    return np.array(features)

def extract_features_from_landmarks(landmarks):
    coords = []

    for lm in landmarks.landmark:
        coords.append([lm.x, lm.y, lm.z])

    coords = np.array(coords)

    # Normalize
    coords = coords - coords[0]
    max_val = np.max(np.abs(coords))
    if max_val != 0:
        coords = coords / max_val

    features = []

    # LANDMARKS
    features.extend(coords.flatten())

    # DISTANCES
    fingertips = [4, 8, 12, 16, 20]

    for i in range(len(fingertips)):
        for j in range(i + 1, len(fingertips)):
            features.append(np.linalg.norm(coords[fingertips[i]] - coords[fingertips[j]]))

    for tip in fingertips:
        features.append(np.linalg.norm(coords[0] - coords[tip]))

    # ANGLES
    def angle(a, b, c):
        ba = a - b
        bc = c - b
        cos = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc) + 1e-6)
        return np.arccos(np.clip(cos, -1.0, 1.0))

    finger_joints = [(0,5,8),(0,9,12),(0,13,16),(0,17,20)]
    for a,b,c in finger_joints:
        features.append(angle(coords[a], coords[b], coords[c]))

    thumb_joints = [(0,2,4),(1,2,3),(2,3,4)]
    for a,b,c in thumb_joints:
        features.append(angle(coords[a], coords[b], coords[c]))

    # FINGER STATES
    tips = [8,12,16,20]
    pips = [6,10,14,18]

    for t,p in zip(tips,pips):
        features.append(1 if coords[t][1] < coords[p][1] else 0)

    features.append(1 if coords[4][0] > coords[3][0] else 0)

    return np.array(features)