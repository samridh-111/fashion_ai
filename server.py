from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import os
import random
import numpy as np
from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity

# --- Setup ---
app = FastAPI()
DATA_DIR = "user_clothes"
TOP_DIR = os.path.join(DATA_DIR, "tops")
BOTTOM_DIR = os.path.join(DATA_DIR, "bottoms")

os.makedirs(TOP_DIR, exist_ok=True)
os.makedirs(BOTTOM_DIR, exist_ok=True)

# Load pretrained model (feature extractor)
model = ResNet50(weights="imagenet", include_top=False, pooling="avg")

# Memory store
wardrobe = {"tops": [], "bottoms": []}

# --- Helpers ---
def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    return features.flatten()

def get_best_fit(occasion):
    # Rough occasion-based preference
    occasion_map = {
        "party": ["jeans", "blazer", "shirt"],
        "office": ["kemeja", "polo", "celana_panjang"],
        "date": ["dress", "jeans", "hoodie"],
        "casual": ["kaos", "jeans", "celana_pendek"],
    }

    tops = wardrobe["tops"]
    bottoms = wardrobe["bottoms"]

    if not tops or not bottoms:
        return None

    best_top, best_bottom = None, None
    best_score = -1

    for top in tops:
        for bottom in bottoms:
            score = cosine_similarity([top["features"]], [bottom["features"]])[0][0]
            if score > best_score:
                best_score = score
                best_top, best_bottom = top, bottom

    return {
        "occasion": occasion,
        "top": best_top["filename"],
        "bottom": best_bottom["filename"],
        "similarity": float(best_score)
    }

# --- ROUTES ---

@app.post("/upload")
async def upload_clothing(
    file: UploadFile = File(...),
    category: str = Form(...)
):
    if category not in ["top", "bottom"]:
        return JSONResponse({"error": "category must be 'top' or 'bottom'"}, status_code=400)

    save_dir = TOP_DIR if category == "top" else BOTTOM_DIR
    save_path = os.path.join(save_dir, file.filename)

    with open(save_path, "wb") as f:
        f.write(await file.read())

    features = extract_features(save_path)
    wardrobe[category + "s"].append({
        "filename": file.filename,
        "path": save_path,
        "features": features
    })

    return {"message": f"{category} uploaded successfully!", "file": file.filename}

@app.get("/wardrobe")
def list_wardrobe():
    return {
        "tops": [x["filename"] for x in wardrobe["tops"]],
        "bottoms": [x["filename"] for x in wardrobe["bottoms"]]
    }

@app.post("/recommend")
def recommend_fit(occasion: str = Form(...)):
    outfit = get_best_fit(occasion)
    if outfit is None:
        return JSONResponse({"error": "Please upload at least one top and one bottom first!"}, status_code=400)
    return outfit
