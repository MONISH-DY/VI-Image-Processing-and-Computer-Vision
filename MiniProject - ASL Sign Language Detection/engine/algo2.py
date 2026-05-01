import numpy as np

def extract_features(landmarks):
    """
    Algorithm 2: Landmarks + Distances + Angles + Finger States
    Input: MediaPipe hand_landmarks object
    Output: Flattened feature vector
    """
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

    # 1. LANDMARKS (63)
    features.extend(coords.flatten())

    # 2. DISTANCES
    fingertips = [4, 8, 12, 16, 20]
    for i in range(len(fingertips)):
        for j in range(i + 1, len(fingertips)):
            features.append(np.linalg.norm(coords[fingertips[i]] - coords[fingertips[j]]))

    for tip in fingertips:
        features.append(np.linalg.norm(coords[0] - coords[tip]))

    # 3. ANGLES
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

    # 4. FINGER STATES
    tips = [8,12,16,20]
    pips = [6,10,14,18]

    for t,p in zip(tips,pips):
        features.append(1 if coords[t][1] < coords[p][1] else 0)

    # Thumb state
    features.append(1 if coords[4][0] > coords[3][0] else 0)

    return np.array(features)
