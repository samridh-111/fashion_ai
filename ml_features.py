"""
ML Feature Extraction Module
Extracts visual features using the fine-tuned fashion embedding model (preferred)
or falls back to raw ResNet50, plus color features via OpenCV and text/occasion
features via Sentence-BERT.

KEY FIXES (vs original):
  1. Loads model/fashion_embedding_resnet50.keras (256-d, fashion-specific) by
     default instead of raw ImageNet ResNet50 (2048-d, generic). The fine-tuned
     model was trained on 15 fashion categories and produces embeddings that
     capture fashion-relevant style signals.
  2. Falls back to stock ResNet50 if the fine-tuned model file is not found.
  3. extract_all_features() now accepts a `category_label` parameter so the
     Sentence-BERT embedding is computed on a meaningful string like
     "casual top shirt" rather than a UUID filename stem.
  4. Preprocessing is consistent (preprocess_input) for both models since the
     fine-tuned model was trained with the same preprocessing pipeline.
"""

import os
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any

# ---------------------------------------------------------------------------
# Model file paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FINETUNED_EMBED_PATH = os.path.join(_SCRIPT_DIR, "model", "fashion_embedding_resnet50.keras")

# Lazy singletons
_embed_model = None      # fine-tuned 256-d fashion embedding model (preferred)
_resnet_model = None     # fallback: raw ResNet50 2048-d
_sentence_model = None


def get_embed_model():
    """
    Load the fine-tuned fashion embedding model (256-d, preferred).
    Falls back gracefully to stock ResNet50 if the .keras file is not found.
    """
    global _embed_model
    if _embed_model is None:
        if os.path.exists(FINETUNED_EMBED_PATH):
            try:
                from tensorflow.keras.models import load_model
                _embed_model = load_model(FINETUNED_EMBED_PATH)
                print(f"✓ Fine-tuned fashion embedding model loaded (256-d, {FINETUNED_EMBED_PATH})")
            except Exception as e:
                print(f"Warning: could not load fine-tuned embedding model: {e}")
                _embed_model = "unavailable"
        else:
            print(f"Warning: fine-tuned model not found at {FINETUNED_EMBED_PATH}; "
                  "falling back to stock ResNet50.")
            _embed_model = "unavailable"
    return None if _embed_model == "unavailable" else _embed_model


def get_resnet_model():
    """Fallback: lazy-load stock ResNet50 (imagenet, 2048-d)."""
    global _resnet_model
    if _resnet_model is None:
        try:
            from tensorflow.keras.applications import ResNet50
            _resnet_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
            print("✓ Fallback ResNet50 (ImageNet, 2048-d) loaded")
        except Exception as e:
            print(f"Warning: ResNet50 unavailable: {e}")
            _resnet_model = "unavailable"
    return None if _resnet_model == "unavailable" else _resnet_model


def get_sentence_model():
    """Lazy-load Sentence-BERT model."""
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
    visual_embedding: Optional[np.ndarray] = None   # 256-d (fine-tuned) or 2048-d (fallback)
    color_histogram: Optional[np.ndarray] = None    # 96-d (32 bins × 3 channels)
    dominant_colors: Optional[List[List[int]]] = None  # Top-5 dominant RGB colors
    brightness: float = 0.5
    saturation: float = 0.5
    text_embedding: Optional[np.ndarray] = None     # Sentence-BERT 384-d
    embedding_dim: int = 0                          # Track which model produced the embedding


@dataclass
class ClothingItem:
    """Represents a single clothing item in the wardrobe"""
    id: str
    filename: str
    path: str
    category: str  # 'top' or 'bottom'
    features: ClothingFeatures
    tags: List[str] = field(default_factory=list)
    category_label: str = ""  # Human-readable label e.g. "casual top shirt"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "filename": self.filename,
            "path": self.path,
            "category": self.category,
            "category_label": self.category_label,
            "tags": self.tags,
            "has_visual": self.features.visual_embedding is not None,
            "has_color": self.features.color_histogram is not None,
            "brightness": round(self.features.brightness, 3),
            "saturation": round(self.features.saturation, 3),
            "dominant_colors": self.features.dominant_colors or [],
            "embedding_dim": self.features.embedding_dim,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], features: "ClothingFeatures") -> "ClothingItem":
        category = data.get("category", "clothing")
        category_label = data.get("category_label")
        if not category_label:
            tags = data.get("tags", [])
            tag_str = f" {' '.join(tags)}" if tags else ""
            category_label = f"casual {category}{tag_str}".strip()

        return cls(
            id=data["id"],
            filename=data["filename"],
            path=data["path"],
            category=category,
            features=features,
            tags=data.get("tags", []),
            category_label=category_label,
        )


class FeatureExtractor:
    """Extracts multi-modal features from clothing images"""

    def extract_visual_features(self, image_path: str) -> Optional[np.ndarray]:
        """
        Extract visual embedding.

        Tries the fine-tuned fashion embedding model first (256-d, fashion-specific).
        Falls back to stock ResNet50 (2048-d) if unavailable.

        Both models use the same ResNet50 preprocess_input, matching the
        preprocessing used during training of the fine-tuned model.
        """
        from tensorflow.keras.preprocessing import image as keras_image
        from tensorflow.keras.applications.resnet50 import preprocess_input

        try:
            img = keras_image.load_img(image_path, target_size=(224, 224))
            x = keras_image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)   # same preprocessing used during fine-tuning

            # Try fine-tuned model first
            model = get_embed_model()
            if model is not None:
                features = model.predict(x, verbose=0)
                return features.flatten()

            # Fallback to stock ResNet50
            model = get_resnet_model()
            if model is not None:
                features = model.predict(x, verbose=0)
                return features.flatten()

            return None

        except Exception as e:
            print(f"Visual feature error for {image_path}: {e}")
            return None

    def get_embedding_dim(self) -> int:
        """Return the dimensionality of the visual embedding in use."""
        if get_embed_model() is not None:
            return 256
        if get_resnet_model() is not None:
            return 2048
        return 0

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

    def extract_all_features(
        self,
        image_path: str,
        description: str = "",
        category_label: str = "",
    ) -> Optional[ClothingFeatures]:
        """
        Extract all features from an image.

        Args:
            image_path: Path to the clothing image.
            description: Deprecated alias for category_label; kept for backward compatibility.
            category_label: Human-readable label for the garment, e.g. "casual top shirt".
                            Used for the Sentence-BERT text embedding.
                            If empty, falls back to the parent directory name (e.g. "tops"),
                            which is at least meaningful unlike a UUID filename stem.
        """
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return None

        visual = self.extract_visual_features(image_path)
        color_hist, dominant_colors, brightness, saturation = self.extract_color_features(image_path)

        # Determine the text to encode for SBERT.
        # Priority: explicit category_label > description > parent directory name.
        # We never fall back to the filename itself because UUIDs are semantically useless.
        label = category_label or description
        if not label:
            parent_dir = os.path.basename(os.path.dirname(image_path))
            label = parent_dir.replace("_", " ").lower()  # e.g. "tops", "bottoms"

        text_emb = self.extract_text_features(label)

        # Require at least visual or color features
        if visual is None and color_hist is None:
            print(f"Could not extract any features from {image_path}")
            return None

        embedding_dim = len(visual) if visual is not None else 0

        return ClothingFeatures(
            visual_embedding=visual,
            color_histogram=color_hist,
            dominant_colors=dominant_colors,
            brightness=brightness,
            saturation=saturation,
            text_embedding=text_emb,
            embedding_dim=embedding_dim,
        )
