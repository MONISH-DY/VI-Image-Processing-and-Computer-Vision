import numpy as np

def extract_features(landmarks):
    """
    Algorithm 1: Flattened normalized landmark features (63 values)
    Input: MediaPipe hand_landmarks object
    Output: Flattened normalized landmark features
    """
    coords = []
    for lm in landmarks.landmark:
        coords.append([lm.x, lm.y, lm.z])

    coords = np.array(coords)  # shape (21, 3)

    # Translation -> move wrist to origin
    coords = coords - coords[0]

    # Scale normalization -> make size invariant
    max_val = np.max(np.abs(coords))
    if max_val != 0:
        coords = coords / max_val

    return coords.flatten()
