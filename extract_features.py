import os
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tqdm import tqdm

DATA_DIR = "data"
OUTPUT_FILE = "features.npz"


def extract_features(img_path):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x) 
        features = embedding_model.predict(x, verbose=0)
        return features.flatten()
    except Exception as e:
        print(f"Error loading {img_path}: {e}")
        return None


def process_folder(folder_path):
    """Extract features for all images in a folder"""
    features_dict = {}
    for fname in tqdm(os.listdir(folder_path)):
        fpath = os.path.join(folder_path, fname)
        if os.path.isfile(fpath):
            feat = extract_features(fpath)
            if feat is not None:
                features_dict[fname] = feat
    return features_dict

if __name__ == "__main__":
    print("🚀 Extracting features...")

    jeans_features = process_folder(os.path.join(DATA_DIR, "jeans"))
    tops_features = process_folder(os.path.join(DATA_DIR, "tops"))

    np.savez(OUTPUT_FILE, jeans=jeans_features, tops=tops_features)
    print(f"✅ Features saved to {OUTPUT_FILE}")
