import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import mediapipe as mp

from b_feature_extraction import extract_landmark_features

# ==============================
# MEDIAPIPE SETUP (STATIC MODE)
# ==============================
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

# ==============================
# PATHS
# ==============================
DATASET_PATH = r"D:\VI\IP & CV\Mini Project\dataset_subset\asl_alphabet_train"
OUTPUT_FILE = r"D:\VI\IP & CV\Mini Project\processed_data\landmarks_dataset.csv"

# ==============================
# MAIN
# ==============================
def main():
    X = []
    y = []

    classes = os.listdir(DATASET_PATH)

    for label in classes:
        class_path = os.path.join(DATASET_PATH, label)

        if not os.path.isdir(class_path):
            continue

        print(f"\nProcessing class: {label}")

        for img_name in tqdm(os.listdir(class_path)):
            img_path = os.path.join(class_path, img_name)

            image = cv2.imread(img_path)
            if image is None:
                continue

            # Convert to RGB
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                hand_landmarks = result.multi_hand_landmarks[0]

                # 🔥 Extract landmark features
                features = extract_landmark_features(hand_landmarks)

                X.append(features)
                y.append(label)

    X = np.array(X)
    y = np.array(y)

    print("\nTotal samples:", len(X))
    print("Feature size:", X.shape)

    df = pd.DataFrame(X)
    df["label"] = y

    os.makedirs("../data_processed", exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print("\n✅ Dataset saved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()