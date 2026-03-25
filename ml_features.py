"""
ML Feature Extraction Module
Extracts visual features using ResNet50, color features using OpenCV,
and text/occasion features using Sentence-BERT.
"""

import os
import uuid
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any

# Lazy imports - loaded on first use to speed up startup
_resnet_model = None
_sentence_model = None


def get_resnet_model():
    """Lazy-load ResNet50 model"""
    global _resnet_model
    if _resnet_model is None:
        try:
            from tensorflow.keras.applications import ResNet50
            _resnet_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
            print("✓ ResNet50 loaded")
        except Exception as e:
            print(f"Warning: ResNet50 unavailable: {e}")
            _resnet_model = "unavailable"
    return None if _resnet_model == "unavailable" else _resnet_model


def get_sentence_model():
    """Lazy-load Sentence-BERT model"""
    global _sentence_model
    if _sentence_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
            print("✓ Sentence-BERT loaded")
        except Exception as e:
            print(f"Warning: Sentence-BERT unavailable: {e}")
            _sentence_model = "unavailable"
    return None if _sentence_model == "unavailable" else _sentence_model


@dataclass
class ClothingFeatures:
    """Container for all extracted features of a clothing item"""
    visual_embedding: Optional[np.ndarray] = None   # ResNet50 2048-d
    color_histogram: Optional[np.ndarray] = None    # 96-d (32 bins × 3 channels)
    dominant_colors: Optional[List[List[int]]] = None  # Top-5 dominant RGB colors
    brightness: float = 0.5
    saturation: float = 0.5
    text_embedding: Optional[np.ndarray] = None     # Sentence-BERT 384-d


@dataclass
class ClothingItem:
    """Represents a single clothing item in the wardrobe"""
    id: str
    filename: str
    path: str
    category: str  # 'top' or 'bottom'
    features: ClothingFeatures
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "filename": self.filename,
            "path": self.path,
            "category": self.category,
            "tags": self.tags,
            "has_visual": self.features.visual_embedding is not None,
            "has_color": self.features.color_histogram is not None,
            "brightness": round(self.features.brightness, 3),
            "saturation": round(self.features.saturation, 3),
            "dominant_colors": self.features.dominant_colors or [],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], features: "ClothingFeatures") -> "ClothingItem":
        return cls(
            id=data["id"],
            filename=data["filename"],
            path=data["path"],
            category=data["category"],
            features=features,
            tags=data.get("tags", []),
        )


class FeatureExtractor:
    """Extracts multi-modal features from clothing images"""

    def extract_visual_features(self, image_path: str) -> Optional[np.ndarray]:
        """Extract 2048-d visual embedding using ResNet50"""
        model = get_resnet_model()
        if model is None:
            return None
        try:
            from tensorflow.keras.preprocessing import image as keras_image
            from tensorflow.keras.applications.resnet50 import preprocess_input

            img = keras_image.load_img(image_path, target_size=(224, 224))
            x = keras_image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            features = model.predict(x, verbose=0)
            return features.flatten()
        except Exception as e:
            print(f"Visual feature error for {image_path}: {e}")
            return None

    def extract_color_features(self, image_path: str):
        """Extract color histogram and dominant colors using OpenCV"""
        try:
            import cv2

            img = cv2.imread(image_path)
            if img is None:
                return None, None, 0.5, 0.5

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # Color histogram (32 bins per channel)
            hist_r = cv2.calcHist([img_rgb], [0], None, [32], [0, 256]).flatten()
            hist_g = cv2.calcHist([img_rgb], [1], None, [32], [0, 256]).flatten()
            hist_b = cv2.calcHist([img_rgb], [2], None, [32], [0, 256]).flatten()

            # Normalize
            total = img_rgb.shape[0] * img_rgb.shape[1]
            hist = np.concatenate([hist_r, hist_g, hist_b]) / total

            # Dominant colors via K-Means
            pixels = img_rgb.reshape(-1, 3).astype(np.float32)
            k = min(5, pixels.shape[0])
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            _, labels, centers = cv2.kmeans(
                pixels, k, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS
            )
            dominant = [[int(c[0]), int(c[1]), int(c[2])] for c in centers]

            # Brightness & saturation (mean V and S in HSV)
            brightness = float(np.mean(img_hsv[:, :, 2])) / 255.0
            saturation = float(np.mean(img_hsv[:, :, 1])) / 255.0

            return hist, dominant, brightness, saturation

        except Exception as e:
            print(f"Color feature error for {image_path}: {e}")
            return None, None, 0.5, 0.5

    def extract_text_features(self, text: str) -> Optional[np.ndarray]:
        """Extract 384-d text embedding using Sentence-BERT"""
        model = get_sentence_model()
        if model is None:
            return None
        try:
            embedding = model.encode([text])[0]
            return embedding
        except Exception as e:
            print(f"Text feature error: {e}")
            return None

    def extract_all_features(self, image_path: str, description: str = "") -> Optional[ClothingFeatures]:
        """Extract all features from an image"""
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return None

        visual = self.extract_visual_features(image_path)
        color_hist, dominant_colors, brightness, saturation = self.extract_color_features(image_path)

        # Text embedding on description or filename hint
        text = description or os.path.splitext(os.path.basename(image_path))[0]
        text_emb = self.extract_text_features(text)

        # Require at least visual or color features
        if visual is None and color_hist is None:
            print(f"Could not extract any features from {image_path}")
            return None

        return ClothingFeatures(
            visual_embedding=visual,
            color_histogram=color_hist,
            dominant_colors=dominant_colors,
            brightness=brightness,
            saturation=saturation,
            text_embedding=text_emb,
        )
