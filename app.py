import os
import random
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


DATA_DIR = "data/Clothes_Dataset"
EMBED_MODEL_PATH = "model/fashion_embedding_resnet50.keras"


W_STYLE = 0.6
W_COLOR = 0.4

TOP_CATEGORIES = ["tshirt", "shirt", "top", "blouse", "hoodie", "kaos", "kemeja"]
BOTTOM_CATEGORIES = ["jeans", "pants", "trouser", "short", "skirt", "celana_panjang", "celana_pendek"]


OCCASION_PREFS = {
    "party":  ["hoodie", "kaos", "jeans", "rok", "jaket_denim"],
    "office": ["kemeja", "blazer", "celana_panjang", "polo", "mantel"],
    "date":   ["gaun", "rok", "kemeja", "jeans", "sweter"],
    "casual": ["hoodie", "kaos", "celana_pendek", "jeans", "jaket"],
}

folders = [f for f in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, f))]
print("Detected folders:", folders)

tops, bottoms = [], []

for folder in folders:
    folder_path = os.path.join(DATA_DIR, folder)
    name = folder.lower()
    if any(cat in name for cat in TOP_CATEGORIES):
        tops.append(folder_path)
    elif any(cat in name for cat in BOTTOM_CATEGORIES):
        bottoms.append(folder_path)

print(f"Tops found: {tops}")
print(f"Bottoms found: {bottoms}")

if not tops or not bottoms:
    print("❌ No tops or bottoms detected. Check TOP_CATEGORIES/BOTTOM_CATEGORIES and folder names.")
    raise SystemExit


print(f"Loading embedding model from: {EMBED_MODEL_PATH}")
embedding_model = load_model(EMBED_MODEL_PATH)
print("✅ Embedding model loaded.")



def extract_style_features(img_path):
    """
    Uses the fine-tuned embedding model to produce a style / visual feature vector.
    """
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        feat = embedding_model.predict(x, verbose=0)
        return feat.flatten()
    except Exception as e:
        print(f"Error loading {img_path} (style): {e}")
        return None


def extract_color_features(img_path, bins=8):
    """
    Simple RGB color histogram (normalized).
    Returns a 3 * bins vector.
    """
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)  # (H, W, 3), values 0-255
        x = x / 255.0

        hist_channels = []
        for c in range(3):  # R, G, B
            channel = x[:, :, c].flatten()
            hist, _ = np.histogram(channel, bins=bins, range=(0.0, 1.0), density=True)
            hist_channels.append(hist)

        hist_vec = np.concatenate(hist_channels)  # length = 3 * bins
        return hist_vec
    except Exception as e:
        print(f"Error loading {img_path} (color): {e}")
        return None




def build_feature_db(paths, label, limit_per_folder=500):
    """
    For each image in the given folders, compute:
      - style embedding (from embedding_model)
      - color histogram
    Returns:
      - style_features: (N, D_style)
      - color_features: (N, D_color)
      - names: list of file paths
    """
    style_features = []
    color_features = []
    names = []

    for path in paths:
        files = [f for f in os.listdir(path) if not f.startswith(".")]
        if limit_per_folder:
            files = files[:limit_per_folder]

        for img_file in tqdm(files, desc=f"Processing {label}"):
            img_path = os.path.join(path, img_file)

            style_feat = extract_style_features(img_path)
            color_feat = extract_color_features(img_path)

            if style_feat is None or color_feat is None:
                continue

            style_features.append(style_feat)
            color_features.append(color_feat)
            names.append(img_path)

    if len(names) == 0:
        print(f"❌ No valid images found for {label}.")
    else:
        print(f"✅ {label}: extracted {len(names)} items.")

    return np.array(style_features), np.array(color_features), names


top_style, top_color, top_names = build_feature_db(tops, "tops")
bottom_style, bottom_color, bottom_names = build_feature_db(bottoms, "bottoms")

if len(top_names) == 0 or len(bottom_names) == 0:
    print("❌ No embeddings extracted for tops or bottoms. Exiting.")
    raise SystemExit


def get_category_from_path(path):
    """
    Returns the folder name (category) for a given image path.
    e.g. data/Clothes_Dataset/Jeans/xxx.jpg -> "jeans"
    """
    return os.path.basename(os.path.dirname(path)).lower()


def occasion_bonus(top_path, bottom_path, occasion):
    """
    Returns a small bonus score based on whether the top/bottom
    categories match the occasion preferences.
    """
    occasion = (occasion or "").lower()
    prefs = OCCASION_PREFS.get(occasion, [])

    top_cat = get_category_from_path(top_path)
    bottom_cat = get_category_from_path(bottom_path)

    bonus = 0.0
    if top_cat in prefs:
        bonus += 0.05
    if bottom_cat in prefs:
        bonus += 0.05

    return bonus



def combined_similarity(top_style_vec, top_color_vec):
    """
    Compute combined similarity between a top and all bottoms.
    Returns an array of shape (num_bottoms,) with weighted similarity scores.
    """
    # Style similarity
    sims_style = cosine_similarity([top_style_vec], bottom_style)[0]

    # Color similarity
    sims_color = cosine_similarity([top_color_vec], bottom_color)[0]

    # Weighted combination
    sims_total = W_STYLE * sims_style + W_COLOR * sims_color
    return sims_total


def recommend_bottom_for_top(top_img_path, occasion="casual", num_results=3):
    """
    Given a top image path, recommend bottoms using:
      - style similarity (CNN embedding)
      - color similarity (histogram)
      - occasion-aware bonus (rule-based)
    """
    top_style_vec = extract_style_features(top_img_path)
    top_color_vec = extract_color_features(top_img_path)

    if top_style_vec is None or top_color_vec is None:
        print("Error: could not extract features from top image.")
        return []

    base_sims = combined_similarity(top_style_vec, top_color_vec)

    # Occasion bonus per bottom
    bonuses = np.array([
        occasion_bonus(top_img_path, b_path, occasion)
        for b_path in bottom_names
    ])

    total_scores = base_sims + bonuses
    best_idx = np.argsort(total_scores)[-num_results:][::-1]
    return [bottom_names[i] for i in best_idx]




random_top = random.choice(top_names)
print(f"\nRandom top image: {random_top}")
print(f"Occasion: {occasion}")

recommendations = recommend_bottom_for_top(random_top, occasion=occasion)
print("\nRecommended bottoms (style + color + occasion):")
for rec in recommendations:
    print("→", rec)
