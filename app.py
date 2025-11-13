import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import random

# -----------------------------
# 1️⃣ Load dataset folders
# -----------------------------
DATA_DIR = "data/Clothes_Dataset"
folders = os.listdir(DATA_DIR)
print("Detected folders:", folders)

# Separate tops & bottoms (simple heuristic)
TOP_CATEGORIES = ["tshirt", "shirt", "top", "blouse", "hoodie"]
BOTTOM_CATEGORIES = ["jeans", "pants", "trouser", "short", "skirt"]

tops, bottoms = [], []

for folder in folders:
    folder_path = os.path.join(DATA_DIR, folder)
    if not os.path.isdir(folder_path):
        continue
    if any(cat in folder.lower() for cat in TOP_CATEGORIES):
        tops.append(folder_path)
    elif any(cat in folder.lower() for cat in BOTTOM_CATEGORIES):
        bottoms.append(folder_path)

print(f"Tops found: {tops}")
print(f"Bottoms found: {bottoms}")

# -----------------------------
# 2️⃣ Feature extractor (ResNet50)
# -----------------------------
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

def extract_features(img_path):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = model.predict(x, verbose=0)
        return features.flatten()
    except Exception as e:
        print(f"Error loading {img_path}: {e}")
        return None

# -----------------------------
# 3️⃣ Build feature database
# -----------------------------
def build_feature_db(paths, label):
    features, names = [], []
    for path in paths:
        for img_file in tqdm(os.listdir(path), desc=f"Processing {label}"):
            img_path = os.path.join(path, img_file)
            feat = extract_features(img_path)
            if feat is not None:
                features.append(feat)
                names.append(img_path)
    return np.array(features), names

top_features, top_names = build_feature_db(tops, "tops")
bottom_features, bottom_names = build_feature_db(bottoms, "bottoms")

# -----------------------------
# 4️⃣ Simple Recommender
# -----------------------------
def recommend_bottom_for_top(top_img_path, num_results=3):
    top_feat = extract_features(top_img_path)
    if top_feat is None:
        print("Error: Could not extract features from input image.")
        return []
    similarities = cosine_similarity([top_feat], bottom_features)[0]
    best_idx = similarities.argsort()[-num_results:][::-1]
    recommendations = [bottom_names[i] for i in best_idx]
    return recommendations

# -----------------------------
# 5️⃣ Test with random top
# -----------------------------
random_top = random.choice(top_names)
print(f"\nRandom top image: {random_top}")

recommendations = recommend_bottom_for_top(random_top)
print("\nRecommended bottoms:")
for rec in recommendations:
    print("→", rec)
