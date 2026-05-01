import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from b_feature_extraction import extract_features

# ==============================
# PATH CONFIG (IMPORTANT)
# ==============================
DATASET_PATH = r"D:\VI\IP & CV\Mini Project\dataset_subset\asl_alphabet_train"
OUTPUT_FILE = r"D:\VI\IP & CV\Mini Project\processed_data\features_dataset.csv"

# ==============================
# MAIN
# ==============================

def main():
    X = []
    y = []

    classes = os.listdir(DATASET_PATH)

    print("Classes found:", classes)

    for label in classes:
        class_path = os.path.join(DATASET_PATH, label)

        if not os.path.isdir(class_path):
            continue

        print(f"\nProcessing class: {label}")

        for img_name in tqdm(os.listdir(class_path)):
            img_path = os.path.join(class_path, img_name)

            # Read image
            image = cv2.imread(img_path)

            if image is None:
                continue

            # Extract features
            features = extract_features(image, mode="static")

            # Skip if no hand detected
            if features is None:
                continue

            X.append(features)
            y.append(label)

    # Convert to numpy
    X = np.array(X)
    y = np.array(y)

    print("\nTotal samples:", len(X))
    print("Feature size:", X.shape)

    # Create DataFrame
    df = pd.DataFrame(X)
    df["label"] = y

    # Ensure directory exists
    os.makedirs("../data_processed", exist_ok=True)

    # Save CSV
    df.to_csv(OUTPUT_FILE, index=False)

    print("\n✅ Dataset saved at:", OUTPUT_FILE)


if __name__ == "__main__":
    main()