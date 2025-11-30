# Fashion AI Project - Complete Beginner's Guide

## 🎯 What This Project Does

This is a **Fashion Recommendation System** that uses AI to:
- Analyze clothing images (tops and bottoms)
- Find matching outfits based on style, color, and patterns
- Recommend what goes well together

Think of it like a personal stylist that looks at your clothes and suggests combinations!

---

## 🧠 Core AI/ML Concepts Used

### 1. **Deep Learning / Convolutional Neural Networks (CNN)**
- Uses a pre-trained model (ResNet50 or VGG16) that was trained on millions of images
- This model can "understand" what's in an image without being specifically trained on fashion

### 2. **Feature Extraction**
- Converts images into numerical vectors (arrays of numbers)
- Each image becomes a unique "fingerprint" of numbers
- Similar-looking clothes will have similar numbers

### 3. **Similarity Matching**
- Uses cosine similarity to compare these number arrays
- Higher similarity = better match

---

## 📁 File-by-File Breakdown

### **1. `app.py` - Main Feature Extraction Script**

**What it does:**
- Processes your entire clothing dataset
- Extracts features from all images
- Tests recommendations on a random top

**Step-by-step:**

```python
# STEP 1: Load the dataset
DATA_DIR = "data/Clothes_Dataset"
folders = os.listdir(DATA_DIR)
```
- Scans the `data/Clothes_Dataset` folder
- Finds all clothing categories (Blazer, Jeans, Kaos, etc.)

```python
# STEP 2: Categorize into tops and bottoms
TOP_CATEGORIES = ["tshirt", "shirt", "top", "blouse", "hoodie"]
BOTTOM_CATEGORIES = ["jeans", "pants", "trouser", "short", "skirt"]
```
- Separates clothing into tops (upper body) and bottoms (lower body)
- Uses simple keyword matching on folder names

```python
# STEP 3: Load the AI model
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
```
- Loads ResNet50 (a famous deep learning model)
- `weights='imagenet'`: Uses pre-trained weights from ImageNet (millions of images)
- `include_top=False`: Removes the final classification layer (we don't need it)
- `pooling='avg'`: Averages features to get one vector per image

```python
# STEP 4: Extract features from each image
def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Resize to 224x224
    x = image.img_to_array(img)                              # Convert to numbers
    x = np.expand_dims(x, axis=0)                            # Add batch dimension
    x = preprocess_input(x)                                  # Normalize colors
    features = model.predict(x, verbose=0)                   # Get feature vector
    return features.flatten()                                 # Make it 1D array
```

**What happens:**
1. Load image and resize to 224×224 pixels (ResNet50 requirement)
2. Convert image to array of numbers (RGB values: 0-255)
3. Preprocess: normalize colors to match ImageNet format
4. Pass through ResNet50: image → 2048 numbers (feature vector)
5. Return the feature vector

```python
# STEP 5: Build feature database
top_features, top_names = build_feature_db(tops, "tops")
```
- Processes ALL top images
- Stores their feature vectors in memory
- Same for bottoms

```python
# STEP 6: Recommend matching bottoms
def recommend_bottom_for_top(top_img_path, num_results=3):
    top_feat = extract_features(top_img_path)                    # Get top's features
    similarities = cosine_similarity([top_feat], bottom_features)[0]  # Compare with all bottoms
    best_idx = similarities.argsort()[-num_results:][::-1]       # Find top 3 matches
    return [bottom_names[i] for i in best_idx]                   # Return file paths
```

**How recommendation works:**
1. Extract features from the input top image
2. Compare with ALL bottom features using cosine similarity
3. Find the 3 most similar bottoms
4. Return those file paths

---

### **2. `server.py` - Web API Server**

**What it does:**
- Creates a web server (using FastAPI)
- Lets you upload your own clothes
- Recommends outfits based on occasion

**Key Components:**

```python
app = FastAPI()
```
- Creates a web server that can receive HTTP requests

```python
wardrobe = {"tops": [], "bottoms": []}
```
- In-memory storage of your uploaded clothes
- Each item stores: filename, file path, and feature vector

**API Endpoints:**

1. **`POST /upload`** - Upload a clothing item
   - Accepts: image file + category (top/bottom)
   - Saves file to disk
   - Extracts features
   - Stores in wardrobe

2. **`GET /wardrobe`** - List all your clothes
   - Returns list of tops and bottoms you've uploaded

3. **`POST /recommend`** - Get outfit recommendation
   - Takes an occasion (party, office, date, casual)
   - Finds best matching top + bottom pair
   - Returns the outfit with similarity score

```python
def get_best_fit(occasion):
    # Tries every top with every bottom
    for top in tops:
        for bottom in bottoms:
            score = cosine_similarity([top["features"]], [bottom["features"]])[0][0]
            # Keeps track of the best match
```
- Brute force approach: tries all combinations
- Picks the pair with highest similarity score

---

### **3. `extract_features.py` - Batch Feature Extraction**

**What it does:**
- Processes images in bulk
- Saves features to a file (`features.npz`)
- Uses VGG16 instead of ResNet50 (another CNN model)

**Why save features?**
- Feature extraction is SLOW (takes time to process each image)
- Once extracted, you can reuse them without reprocessing
- `features.npz` is a compressed NumPy file format

**Process:**
1. Load VGG16 model
2. For each image in jeans folder → extract features → save
3. For each image in tops folder → extract features → save
4. Save everything to `features.npz`

---

### **4. `recommender.py` - Simple Recommendation Test**

**What it does:**
- Loads pre-saved features from `features.npz`
- Picks a random jeans image
- Finds matching tops

**Why this file exists:**
- Quick way to test recommendations without processing images again
- Uses the saved features from `extract_features.py`

---

### **5. `utils.py` - Helper Functions**

**What it does:**
- Provides reusable functions for loading features and finding similarities

```python
def load_features(file_path="features.npz"):
    # Loads the saved features from disk
```

```python
def get_similar_tops(jeans_vector, tops_features, top_n=3):
    # Compares one jeans vector with all tops
    # Returns top N matches
```

---

## 📦 Import-by-Import Explanation

### **Standard Library (Built-in Python)**
```python
import os
```
- **What:** File and directory operations
- **Used for:** Listing folders, joining file paths, checking if files exist
- **Example:** `os.listdir("data")` → gets list of files in folder

```python
import random
```
- **What:** Random number/choice generator
- **Used for:** Picking random images for testing
- **Example:** `random.choice([1,2,3])` → picks random item

---

### **NumPy - Numerical Computing**
```python
import numpy as np
```
- **What:** Library for working with arrays and matrices
- **Why needed:** Images are arrays of numbers, features are arrays
- **Key uses:**
  - `np.array()` - Create arrays
  - `np.expand_dims()` - Add dimensions (e.g., for batch processing)
  - `np.savez()` - Save arrays to file
  - `np.load()` - Load arrays from file
- **Example:** `[1, 2, 3]` → NumPy array → can do math operations efficiently

---

### **TensorFlow/Keras - Deep Learning**
```python
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
```

**TensorFlow:**
- **What:** Google's deep learning framework
- **Why:** Provides tools to build and use neural networks

**Keras:**
- **What:** High-level API built on TensorFlow (easier to use)
- **Why:** Makes loading pre-trained models simple

**`image.load_img()`:**
- **What:** Loads an image file
- **Returns:** PIL Image object
- **Example:** `load_img("shirt.jpg")` → image object

**`image.img_to_array()`:**
- **What:** Converts image to NumPy array
- **Returns:** Array of shape (height, width, 3) for RGB
- **Example:** Image → `[[[255, 0, 0], ...], ...]` (pixel values)

**`preprocess_input()`:**
- **What:** Normalizes image data for the specific model
- **Why:** Models expect data in a specific format
- **Does:** Adjusts color values (e.g., subtracts mean, scales)

**`ResNet50()`:**
- **What:** A 50-layer deep neural network
- **Trained on:** ImageNet (1.4 million images, 1000 categories)
- **Why use it:** Already knows how to recognize patterns, shapes, colors
- **`weights='imagenet'`:** Loads pre-trained weights (don't train from scratch)
- **`include_top=False`:** Removes final classification layer (we want features, not labels)
- **`pooling='avg'`:** Averages spatial features → one vector per image

**`VGG16()`:**
- **What:** Another CNN model (16 layers, older than ResNet50)
- **Similar to ResNet50:** Pre-trained, used for feature extraction
- **Difference:** Slightly different architecture, still works well

**`model.predict()`:**
- **What:** Runs the image through the neural network
- **Input:** Preprocessed image array
- **Output:** Feature vector (array of numbers representing the image)
- **Example:** Image → `[0.23, -0.45, 0.67, ..., 0.12]` (2048 numbers for ResNet50)

---

### **Scikit-learn - Machine Learning Tools**
```python
from sklearn.metrics.pairwise import cosine_similarity
```
- **What:** Library for machine learning algorithms and utilities
- **`cosine_similarity()`:** Measures how similar two vectors are
- **How it works:**
  - Takes two feature vectors
  - Calculates angle between them
  - Returns value between -1 and 1
  - **1.0** = identical, **0.0** = unrelated, **-1.0** = opposite
- **Example:**
  ```python
  vector1 = [1, 2, 3]
  vector2 = [2, 4, 6]  # Same direction, different magnitude
  cosine_similarity([vector1], [vector2])  # → ~1.0 (very similar)
  ```

---

### **tqdm - Progress Bars**
```python
from tqdm import tqdm
```
- **What:** Shows progress bars for loops
- **Why:** Processing thousands of images takes time - nice to see progress
- **Example:**
  ```python
  for img in tqdm(images):  # Shows: [████████░░] 80%
      process(img)
  ```

---

### **Pillow (PIL) - Image Processing**
```python
from PIL import Image
```
- **What:** Python Imaging Library
- **Used for:** Basic image operations (though TensorFlow's `image` module is used more here)
- **Alternative:** Can also load/manipulate images

---

### **FastAPI - Web Framework**
```python
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
```
- **What:** Modern Python web framework for building APIs
- **Why:** Easy way to create REST API endpoints
- **`FastAPI()`:** Creates the web application
- **`File`, `UploadFile`:** Handle file uploads
- **`Form`:** Handle form data (like category selection)
- **`JSONResponse`:** Return JSON data to client

---

## 🔄 How Everything Works Together

### **Workflow 1: Initial Setup (One-time)**
```
1. Run extract_features.py
   → Processes all images in dataset
   → Saves features to features.npz
   → Takes a while (processing thousands of images)

2. Run app.py (optional)
   → Loads dataset
   → Extracts features on-the-fly
   → Tests recommendations
```

### **Workflow 2: Using the Web Server**
```
1. Start server: python server.py (or uvicorn server:app)
   → Server starts on http://localhost:8000

2. Upload clothes via /upload endpoint
   → User uploads top1.jpg as "top"
   → Server saves file, extracts features, stores in wardrobe

3. Upload more clothes
   → User uploads bottom1.jpg as "bottom"
   → Same process

4. Get recommendation via /recommend endpoint
   → User requests outfit for "party"
   → Server compares all tops with all bottoms
   → Returns best matching pair
```

### **Workflow 3: Quick Testing**
```
1. Run recommender.py
   → Loads pre-saved features
   → Picks random jeans
   → Finds matching tops
   → Prints results
```

---

## 🎓 Key AI/ML Concepts Explained Simply

### **1. Transfer Learning**
- **What:** Using a model trained on one task for a different task
- **In this project:** ResNet50 was trained to classify objects (cat, dog, car, etc.)
- **We use it for:** Extracting features from fashion images
- **Why it works:** The model learned to recognize patterns, edges, colors, textures - these are useful for fashion too!

### **2. Feature Vectors**
- **What:** A list of numbers that represent an image
- **Analogy:** Like a fingerprint - unique to each image
- **Example:** A red shirt might be: `[0.8, -0.2, 0.5, ...]`
- **Why:** Can't directly compare images, but can compare numbers!

### **3. Cosine Similarity**
- **What:** Measures how similar two vectors are
- **Analogy:** Like measuring the angle between two arrows
- **If vectors point same direction:** High similarity (good match)
- **If vectors point different directions:** Low similarity (bad match)

### **4. Pre-trained Models**
- **What:** Neural networks already trained on huge datasets
- **Why use them:** Training from scratch takes weeks and needs millions of images
- **We use:** ResNet50 (trained on ImageNet) or VGG16
- **What we do:** Remove the final layer, use the middle layers to get features

---

## 🚀 How to Use This Project

1. **Extract features from dataset:**
   ```bash
   python extract_features.py
   ```

2. **Test recommendations:**
   ```bash
   python recommender.py
   ```

3. **Run web server:**
   ```bash
   python server.py
   # Or: uvicorn server:app --reload
   ```

4. **Use the API:**
   - Upload clothes: `POST http://localhost:8000/upload`
   - Get wardrobe: `GET http://localhost:8000/wardrobe`
   - Get recommendation: `POST http://localhost:8000/recommend`

---

## 💡 Why This Approach Works

1. **Pre-trained models** already understand visual patterns
2. **Feature extraction** converts images to comparable numbers
3. **Similarity matching** finds visually similar items
4. **No training needed** - we just use what's already learned!

---

## 🎯 Summary

This project is a **content-based recommendation system**:
- **Content-based:** Uses the actual image content (not user preferences)
- **Recommendation:** Suggests matching items
- **System:** Automated process

The magic happens because:
- Deep learning models can "see" patterns humans might miss
- Mathematical similarity can find good matches
- Pre-trained models save us from training from scratch

---

## 📚 Next Steps to Learn More

1. **Learn about CNNs:** How neural networks process images
2. **Learn about embeddings:** Converting data to vectors
3. **Learn about similarity metrics:** Cosine, Euclidean distance, etc.
4. **Experiment:** Try different models (ResNet101, EfficientNet, etc.)
5. **Improve:** Add color matching, style classification, etc.


