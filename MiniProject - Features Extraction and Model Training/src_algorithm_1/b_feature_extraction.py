import numpy as np

# ==============================
# EXTRACT LANDMARK FEATURES ONLY
# ==============================
def extract_landmark_features(landmarks):
    """
    Input: MediaPipe hand_landmarks object
    Output: Flattened normalized landmark features (63 values)
    """

    coords = []

    # Extract (x, y, z) for all 21 landmarks
    for lm in landmarks.landmark:
        coords.append([lm.x, lm.y, lm.z])

    coords = np.array(coords)  # shape (21, 3)

    # ==============================
    # NORMALIZATION (IMPORTANT)
    # ==============================

    # 1. Translation → move wrist to origin
    coords = coords - coords[0]

    # 2. Scale normalization → make size invariant
    max_val = np.max(np.abs(coords))
    if max_val != 0:
        coords = coords / max_val

    # ==============================
    # FLATTEN → 63 FEATURES
    # ==============================
    features = coords.flatten()

    return features