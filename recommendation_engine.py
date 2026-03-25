"""
Recommendation Engine
Multi-signal outfit scoring combining visual similarity, color harmony,
occasion relevance, and user feedback learning.
"""

import os
import json
import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Set

from ml_features import ClothingItem, ClothingFeatures, FeatureExtractor


# ---------------------------------------------------------------------------
# Occasion keyword mapping → text the model embeds for similarity
# ---------------------------------------------------------------------------
OCCASION_DESCRIPTIONS = {
    "casual":   "relaxed casual everyday wear comfortable jeans t-shirt",
    "office":   "professional business formal office smart attire blazer",
    "party":    "party night out trendy stylish fun club social event",
    "date":     "romantic elegant dinner date night dress sophisticated",
    "formal":   "black tie formal gala ceremonial elegant luxury",
    "sport":    "athletic sporty workout gym activewear performance",
    "beach":    "beach summer vacation swimwear tropical light airy",
    "outdoor":  "outdoor hiking nature casual durable layered adventure",
}


def cosine_sim(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    """Safe cosine similarity returning 0.0 if inputs are None or zero."""
    if a is None or b is None:
        return 0.0
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def color_harmony_score(
    top_hist: Optional[np.ndarray],
    bottom_hist: Optional[np.ndarray],
    top_dominant: Optional[List],
    bottom_dominant: Optional[List],
) -> float:
    """
    Combine histogram similarity with a complementary color bonus.
    Complementary colors: hue difference ~180° → bonus.
    Analogous colors: hue difference <30° → bonus.
    """
    hist_score = cosine_sim(top_hist, bottom_hist)

    # Dominant-color bonus
    bonus = 0.0
    if top_dominant and bottom_dominant:
        try:
            # Convert first dominant color to rough hue (0-360)
            def to_hue(rgb):
                r, g, b = [x / 255.0 for x in rgb]
                mx, mn = max(r, g, b), min(r, g, b)
                diff = mx - mn
                if diff == 0:
                    return 0
                if mx == r:
                    h = 60 * ((g - b) / diff % 6)
                elif mx == g:
                    h = 60 * ((b - r) / diff + 2)
                else:
                    h = 60 * ((r - g) / diff + 4)
                return h % 360

            top_hue = to_hue(top_dominant[0])
            bot_hue = to_hue(bottom_dominant[0])
            diff = abs(top_hue - bot_hue)
            diff = min(diff, 360 - diff)

            if 150 <= diff <= 210:   # complementary
                bonus = 0.15
            elif diff <= 30:         # analogous / monochromatic
                bonus = 0.10
            elif 90 <= diff <= 150:  # split-complementary
                bonus = 0.05
        except Exception:
            pass

    return min(1.0, hist_score * 0.85 + bonus)


# ---------------------------------------------------------------------------
# Feedback / learning system
# ---------------------------------------------------------------------------
FEEDBACK_FILE = "feedback.json"


class FeedbackSystem:
    """Stores liked / rejected outfit pairs and adjusts scores accordingly."""

    def __init__(self, feedback_file: str = FEEDBACK_FILE):
        self.feedback_file = feedback_file
        self.liked_pairs: Set[Tuple[str, str]] = set()
        self.rejected_pairs: Set[Tuple[str, str]] = set()
        self.feedback_history: List[Dict] = []
        self._load()

    def _load(self):
        if os.path.exists(self.feedback_file):
            try:
                with open(self.feedback_file) as f:
                    data = json.load(f)
                self.feedback_history = data.get("history", [])
                for entry in self.feedback_history:
                    pair = (entry["top_id"], entry["bottom_id"])
                    if entry["liked"]:
                        self.liked_pairs.add(pair)
                    else:
                        self.rejected_pairs.add(pair)
            except Exception as e:
                print(f"Warning: could not load feedback: {e}")

    def _save(self):
        try:
            data = {
                "history": self.feedback_history,
                "liked_count": len(self.liked_pairs),
                "rejected_count": len(self.rejected_pairs),
            }
            with open(self.feedback_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: could not save feedback: {e}")

    def record_feedback(self, top_id: str, bottom_id: str, liked: bool, occasion: str = ""):
        pair = (top_id, bottom_id)
        entry = {
            "top_id": top_id,
            "bottom_id": bottom_id,
            "liked": liked,
            "occasion": occasion,
            "timestamp": time.time(),
        }
        self.feedback_history.append(entry)
        if liked:
            self.liked_pairs.add(pair)
            self.rejected_pairs.discard(pair)
        else:
            self.rejected_pairs.add(pair)
            self.liked_pairs.discard(pair)
        self._save()

    def get_score_modifier(self, top_id: str, bottom_id: str) -> float:
        pair = (top_id, bottom_id)
        if pair in self.liked_pairs:
            return 0.10   # boost liked pairs
        if pair in self.rejected_pairs:
            return -0.30  # penalise rejected pairs
        return 0.0

    def is_rejected(self, top_id: str, bottom_id: str) -> bool:
        return (top_id, bottom_id) in self.rejected_pairs


# ---------------------------------------------------------------------------
# Recommendation result
# ---------------------------------------------------------------------------
@dataclass
class OutfitRecommendation:
    top: ClothingItem
    bottom: ClothingItem
    score: float            # raw 0-1
    breakdown: Dict[str, float] = field(default_factory=dict)
    explanation: str = ""

    @property
    def percentage(self) -> float:
        return round(self.score * 100, 1)


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------
class RecommendationEngine:
    """
    Scores all (top, bottom) pairs using:
      - Visual similarity (ResNet50 embeddings)
      - Color harmony (histogram + complementary heuristic)
      - Occasion relevance (Sentence-BERT)
      - Feedback adjustment
    """

    # Signal weights (must sum to 1.0)
    W_VISUAL = 0.40
    W_COLOR  = 0.30
    W_OCCASION = 0.20
    W_BRIGHTNESS = 0.10

    def __init__(self, feature_extractor: FeatureExtractor):
        self.fe = feature_extractor
        self.feedback_system = FeedbackSystem()
        self._occasion_cache: Dict[str, Optional[np.ndarray]] = {}

    def _get_occasion_embedding(self, occasion: str) -> Optional[np.ndarray]:
        key = occasion.lower()
        if key not in self._occasion_cache:
            text = OCCASION_DESCRIPTIONS.get(key, key)
            self._occasion_cache[key] = self.fe.extract_text_features(text)
        return self._occasion_cache[key]

    def _score_pair(
        self,
        top: ClothingItem,
        bottom: ClothingItem,
        occasion_emb: Optional[np.ndarray],
    ) -> Tuple[float, Dict[str, float]]:
        tf = top.features
        bf = bottom.features

        # 1. Visual similarity
        visual = cosine_sim(tf.visual_embedding, bf.visual_embedding)

        # 2. Color harmony
        color = color_harmony_score(
            tf.color_histogram, bf.color_histogram,
            tf.dominant_colors, bf.dominant_colors,
        )

        # 3. Occasion relevance (average of top & bottom vs occasion)
        if occasion_emb is not None:
            occ_top = cosine_sim(tf.text_embedding, occasion_emb)
            occ_bot = cosine_sim(bf.text_embedding, occasion_emb)
            occasion_score = (occ_top + occ_bot) / 2.0
        else:
            occasion_score = 0.5  # neutral

        # 4. Brightness compatibility (similar brightness = better match)
        brightness_diff = abs(tf.brightness - bf.brightness)
        brightness_score = max(0.0, 1.0 - brightness_diff * 2.0)

        raw = (
            self.W_VISUAL    * visual +
            self.W_COLOR     * color +
            self.W_OCCASION  * occasion_score +
            self.W_BRIGHTNESS * brightness_score
        )

        breakdown = {
            "visual":     round(visual, 3),
            "color":      round(color, 3),
            "occasion":   round(occasion_score, 3),
            "brightness": round(brightness_score, 3),
        }

        return raw, breakdown

    def _build_explanation(self, breakdown: Dict[str, float], score: float) -> str:
        parts = []
        if breakdown.get("color", 0) >= 0.7:
            parts.append("great color harmony")
        if breakdown.get("visual", 0) >= 0.7:
            parts.append("matching style")
        if breakdown.get("occasion", 0) >= 0.6:
            parts.append("perfect for the occasion")
        if breakdown.get("brightness", 0) >= 0.7:
            parts.append("balanced tones")

        if not parts:
            parts.append("decent overall match")

        pct = round(score * 100)
        return f"{pct}% match — {', '.join(parts)}."

    def recommend_outfits(
        self,
        tops: List[ClothingItem],
        bottoms: List[ClothingItem],
        occasion: str = "casual",
        num_results: int = 5,
        exclude_rejected: bool = True,
    ) -> List[OutfitRecommendation]:
        """Score all (top, bottom) pairs and return top-N recommendations."""
        occasion_emb = self._get_occasion_embedding(occasion)
        results: List[OutfitRecommendation] = []

        for top in tops:
            for bottom in bottoms:
                if exclude_rejected and self.feedback_system.is_rejected(top.id, bottom.id):
                    continue

                raw, breakdown = self._score_pair(top, bottom, occasion_emb)
                modifier = self.feedback_system.get_score_modifier(top.id, bottom.id)
                final = max(0.0, min(1.0, raw + modifier))

                explanation = self._build_explanation(breakdown, final)
                results.append(OutfitRecommendation(
                    top=top,
                    bottom=bottom,
                    score=final,
                    breakdown=breakdown,
                    explanation=explanation,
                ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:num_results]

    def record_like(self, top_id: str, bottom_id: str, occasion: str = ""):
        self.feedback_system.record_feedback(top_id, bottom_id, liked=True, occasion=occasion)

    def get_alternative_outfit(
        self,
        rejected_top_id: str,
        rejected_bottom_id: str,
        tops: List[ClothingItem],
        bottoms: List[ClothingItem],
        occasion: str = "casual",
    ) -> Optional[OutfitRecommendation]:
        """Return next-best outfit excluding the rejected pair."""
        self.feedback_system.record_feedback(
            rejected_top_id, rejected_bottom_id, liked=False, occasion=occasion
        )
        recs = self.recommend_outfits(tops, bottoms, occasion, num_results=20, exclude_rejected=True)
        for rec in recs:
            if rec.top.id != rejected_top_id or rec.bottom.id != rejected_bottom_id:
                return rec
        return None
