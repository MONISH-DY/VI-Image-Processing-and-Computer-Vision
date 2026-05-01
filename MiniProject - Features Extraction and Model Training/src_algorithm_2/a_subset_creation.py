import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import imagehash
from sklearn.cluster import KMeans
import mediapipe as mp
import shutil

# ==============================
# CONFIG
# ==============================
DATASET_DIR = r"D:\VI\IP & CV\Mini Project\dataset\asl_alphabet_train\asl_alphabet_train"        # root folder (contains class folders)
OUTPUT_DIR = r"D:\VI\IP & CV\Mini Project\dataset_subset\asl_alphabet_train"         # where subset will be saved
IMAGES_PER_CLASS = 250

# ==============================
# MEDIAPIPE SETUP
# ==============================
mp_hands = mp.solutions.hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

# ==============================
# STEP 1: REMOVE DUPLICATES
# ==============================
def remove_duplicates(image_paths):
    hash_dict = {}
    filtered = []

    for path in tqdm(image_paths, desc="Removing duplicates"):
        try:
            img = Image.open(path)
            h = imagehash.phash(img)

            # Check similarity with existing hashes
            is_duplicate = False
            for existing_hash in hash_dict:
                if abs(h - existing_hash) < 5:  # threshold
                    is_duplicate = True
                    break

            if not is_duplicate:
                hash_dict[h] = path
                filtered.append(path)

        except:
            continue

    return filtered

# ==============================
# STEP 2: EXTRACT LANDMARKS
# ==============================
def extract_landmarks(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = mp_hands.process(img_rgb)

    if not result.multi_hand_landmarks:
        return None

    landmarks = result.multi_hand_landmarks[0]

    coords = []
    for lm in landmarks.landmark:
        coords.extend([lm.x, lm.y, lm.z])

    coords = np.array(coords)

    # Normalize (VERY IMPORTANT)
    coords = coords - coords[0]  # wrist as origin
    max_val = np.max(np.abs(coords))
    if max_val != 0:
        coords = coords / max_val

    return coords

# ==============================
# STEP 3: BUILD FEATURE MATRIX
# ==============================
def build_features(image_paths):
    X = []
    valid_paths = []

    for path in tqdm(image_paths, desc="Extracting landmarks"):
        features = extract_landmarks(path)
        if features is not None:
            X.append(features)
            valid_paths.append(path)

    return np.array(X), valid_paths

# ==============================
# STEP 4: CLUSTER & SELECT
# ==============================
def select_diverse_subset(X, image_paths, k):
    print("Clustering into", k, "groups...")

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_

    selected = []

    for i in range(k):
        cluster_indices = np.where(labels == i)[0]

        if len(cluster_indices) == 0:
            continue

        center = centers[i]

        best_idx = min(
            cluster_indices,
            key=lambda idx: np.linalg.norm(X[idx] - center)
        )

        selected.append(image_paths[best_idx])

    return selected

# ==============================
# MAIN PIPELINE
# ==============================
def process_class(class_name):
    print(f"\nProcessing class: {class_name}")

    class_path = os.path.join(DATASET_DIR, class_name)
    image_paths = [
        os.path.join(class_path, img)
        for img in os.listdir(class_path)
    ]

    # Step 1: Remove duplicates
    filtered_paths = remove_duplicates(image_paths)
    print(f"After duplicate removal: {len(filtered_paths)} images")

    # Step 2: Extract features
    X, valid_paths = build_features(filtered_paths)
    print(f"Valid landmark images: {len(valid_paths)}")

    # If less than required, just return all
    if len(valid_paths) <= IMAGES_PER_CLASS:
        return valid_paths

    # Step 3: Clustering
    selected = select_diverse_subset(X, valid_paths, IMAGES_PER_CLASS)

    return selected

# ==============================
# SAVE SUBSET
# ==============================
def save_subset(class_name, selected_paths):
    output_class_dir = os.path.join(OUTPUT_DIR, class_name)
    os.makedirs(output_class_dir, exist_ok=True)

    for i, path in enumerate(selected_paths):
        ext = os.path.splitext(path)[1]
        new_name = f"{class_name}_{i}{ext}"
        shutil.copy(path, os.path.join(output_class_dir, new_name))

# ==============================
# RUN FOR ALL CLASSES
# ==============================
def main():
    classes = os.listdir(DATASET_DIR)

    for cls in classes:
        selected_paths = process_class(cls)
        save_subset(cls, selected_paths)

    print("\n✅ Subset creation completed!")

if __name__ == "__main__":
    main()