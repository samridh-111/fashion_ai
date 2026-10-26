"""
Recommendation Engine
Multi-signal outfit scoring combining:
  - Style consistency (fine-tuned fashion embedding similarity)
  - Color harmony (histogram + complementary hue heuristic)
  - Occasion relevance (Sentence-BERT)
  - Category compatibility (rule-based fashion knowledge)
  - Brightness balance
  - User feedback adjustment

KEY FIXES (vs original):
  1. Uses 75% as a PRESENTATION TIER threshold: outfits scoring > 0.75 are
     labelled as strong matches; lower-scoring outfits are still returned
     as alternatives, ranked by compatibility.
  2. Adds CATEGORY_COMPATIBILITY: rule-based scores penalise meaningless pairings
     (e.g. blazer + shorts) and reward coherent ones (e.g. shirt + jeans).
  3. Visual score is now called "style" — it measures style consistency between
     top and bottom, not compatibility. Labelled accordingly in breakdown.
  4. score is documented as a transparent compatibility heuristic, NOT a
     calibrated probability. Labelled "style_compatibility_score" in the API.
  5. Weights re-balanced: category compatibility signal added (W_CATEGORY = 0.20),
     visual weight reduced to 0.30, occasion weight remains 0.20.
"""

import os
import json
import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Set

from ml_features import ClothingItem, ClothingFeatures, FeatureExtractor


# ---------------------------------------------------------------------------
# Presentation tier threshold
# ---------------------------------------------------------------------------
# Used ONLY for labeling / presentation tiers, NOT for filtering.
# Outfits at or above this value are labelled as strong/good matches;
# outfits below are still returned as alternatives, ranked by compatibility.
# This is a heuristic compatibility score, not a calibrated probability.
PROMOTION_THRESHOLD = 0.75


# ---------------------------------------------------------------------------
# Category compatibility matrix
# ---------------------------------------------------------------------------
# Maps (top_category_keyword, bottom_category_keyword) → bonus score (0-1).
# "tops" / "bottoms" are the default labels when no explicit label is given.
# These reflect basic fashion outfit rules.
#
# Convention: score 1.0 = highly compatible, 0.5 = neutral, 0.0 = poor match.
CATEGORY_COMPATIBILITY: Dict[Tuple[str, str], float] = {
    # Tops × Bottoms — well-matched combinations
    ("shirt",    "jeans"):       1.0,
    ("shirt",    "trouser"):     1.0,
    ("shirt",    "chino"):       1.0,
    ("shirt",    "skirt"):       0.9,
    ("shirt",    "shorts"):      0.8,
    ("kemeja",   "celana_panjang"): 1.0,
    ("kemeja",   "jeans"):       1.0,
    ("kemeja",   "rok"):         0.9,
    ("kaos",     "jeans"):       1.0,
    ("kaos",     "celana_pendek"): 1.0,
    ("kaos",     "shorts"):      1.0,
    ("kaos",     "chino"):       0.9,
    ("tshirt",   "jeans"):       1.0,
    ("tshirt",   "shorts"):      1.0,
    ("hoodie",   "jeans"):       0.95,
    ("hoodie",   "celana_panjang"): 0.9,
    ("hoodie",   "shorts"):      0.85,
    ("sweter",   "jeans"):       0.95,
    ("sweter",   "rok"):         0.90,
    ("sweter",   "celana_panjang"): 0.90,
    ("blazer",   "jeans"):       0.85,
    ("blazer",   "trouser"):     1.0,
    ("blazer",   "celana_panjang"): 1.0,
    ("blazer",   "skirt"):       0.90,
    ("polo",     "jeans"):       0.95,
    ("polo",     "chino"):       1.0,
    ("polo",     "celana_panjang"): 0.95,
    ("jaket",    "jeans"):       0.90,
    ("jaket_denim", "jeans"):    0.75,  # double-denim — acceptable but not ideal
    ("jaket_olahraga", "celana_pendek"): 1.0,
    ("mantel",   "jeans"):       0.90,
    ("mantel",   "trouser"):     1.0,
    ("gaun",     "jeans"):       0.40,  # dress over jeans — generally poor
}

# Category groups for fallback lookup
TOP_KEYWORDS = ["shirt", "kemeja", "kaos", "tshirt", "hoodie", "sweter", "blazer",
                "polo", "jaket", "mantel", "gaun", "top", "blouse"]
BOTTOM_KEYWORDS = ["jeans", "celana_panjang", "celana_pendek", "shorts", "skirt",
                   "rok", "trouser", "chino", "pants"]


def get_category_keyword(category_label: str, fallback_category: str) -> str:
    """
    Extract the most specific keyword from a category label for compatibility lookup.
    """
    label = (category_label or fallback_category).lower()
    # Try exact keyword match first
    all_keywords = TOP_KEYWORDS + BOTTOM_KEYWORDS
    for kw in all_keywords:
        if kw in label:
            return kw
    return label.split()[0] if label.split() else fallback_category


def category_compatibility_score(
    top_label: str,
    bottom_label: str,
    top_category: str,
    bottom_category: str,
) -> float:
    """
    Return a 0-1 compatibility score based on clothing category rules.

    If the pair is not in the explicit compatibility matrix, return 0.7 (neutral)
    for same-domain pairs (top+bottom) and 0.5 for unknown combinations.
    """
    top_kw = get_category_keyword(top_label, top_category)
    bot_kw = get_category_keyword(bottom_label, bottom_category)

    # Direct lookup
    score = CATEGORY_COMPATIBILITY.get((top_kw, bot_kw))
    if score is not None:
        return score

    # Try reversed (in case order matters less than expected)
    score = CATEGORY_COMPATIBILITY.get((bot_kw, top_kw))
    if score is not None:
        return score

    # Default: top + bottom pairing is at least neutral
    if top_category == "top" and bottom_category == "bottom":
        return 0.70   # neutral — not specifically poor, not specifically good
    return 0.50       # unknown category combination


# ---------------------------------------------------------------------------
# Occasion keyword mapping
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
    Combine histogram similarity with a complementary/analogous color bonus.

    Note: cosine_sim on colour histograms rewards similar colour distributions.
    The hue bonus partially corrects for this by rewarding complementary and
    analogous pairings that may have different distributions but still look good.
    """
    hist_score = cosine_sim(top_hist, bottom_hist)

    # Dominant-color bonus
    bonus = 0.0
    if top_dominant and bottom_dominant:
        try:
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
    score: float            # raw 0-1 style compatibility heuristic
    breakdown: Dict[str, float] = field(default_factory=dict)
    explanation: str = ""

    @property
    def percentage(self) -> float:
        """
        Returns the score as a percentage (0-100).
        This is a transparent style-compatibility heuristic, NOT a calibrated
        probability. It should be labelled as 'Style Match' in the UI,
        not as 'AI Confidence' or 'Match Probability'.
        """
        if not np.isfinite(self.score):
            return 0.0
        return round(self.score * 100, 1)

    @property
    def is_promoted(self) -> bool:
        """True if the outfit qualifies as a top-tier / strong match (score >= PROMOTION_THRESHOLD)."""
        return bool(np.isfinite(self.score) and self.score >= PROMOTION_THRESHOLD)

    @property
    def tier(self) -> str:
        """
        Presentation tier based on score.
        Used for UI presentation and labeling, NEVER for filtering out outfits.
        """
        if not np.isfinite(self.score):
            return "invalid"
        if self.score >= 0.85:
            return "strong"
        elif self.score >= PROMOTION_THRESHOLD:
            return "good"
        elif self.score >= 0.60:
            return "moderate"
        elif self.score >= 0.40:
            return "alternative"
        else:
            return "experimental"

    @property
    def tier_label(self) -> str:
        """Human-readable label for the presentation tier."""
        labels = {
            "strong": "Strong Match",
            "good": "Good Match",
            "moderate": "Worth Considering",
            "alternative": "Alternative Option",
            "experimental": "Experimental Combo",
            "invalid": "Unscored Combination",
        }
        return labels.get(self.tier, "Alternative Option")


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------
class RecommendationEngine:
    """
    Scores all valid (top, bottom) pairs using:
      - Style consistency (fashion embedding similarity) — W_STYLE = 0.30
      - Color harmony (histogram + hue complementarity)  — W_COLOR = 0.25
      - Occasion relevance (Sentence-BERT)               — W_OCCASION = 0.20
      - Category compatibility (fashion rules)           — W_CATEGORY = 0.20
      - Brightness balance                               — W_BRIGHTNESS = 0.05

    Returns the best available outfits ranked by compatibility score.
    The 75% threshold is used exclusively for presentation tier labeling,
    never to hide or filter out valid combinations from the user's wardrobe.
    """

    W_STYLE      = 0.30
    W_COLOR      = 0.25
    W_OCCASION   = 0.20
    W_CATEGORY   = 0.20
    W_BRIGHTNESS = 0.05

    def __init__(self, feature_extractor: FeatureExtractor):
        self.fe = feature_extractor
        self.feedback_system = FeedbackSystem()
        self._occasion_cache: Dict[str, Optional[np.ndarray]] = {}

    def _get_occasion_embedding(self, occasion: str) -> Optional[np.ndarray]:
        key = occasion.lower()
        if key not in self._occasion_cache:
            text = OCCASION_DESCRIPTIONS.get(key, occasion)
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

        # 1. Style consistency (embedding similarity)
        #    Measures whether the two items share a visual style signature.
        #    High score = similar aesthetic (both casual, both formal, etc.)
        style = cosine_sim(tf.visual_embedding, bf.visual_embedding)

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
            occasion_score = 0.5  # neutral when SBERT unavailable

        # 4. Category compatibility (fashion outfit rules)
        category_score = category_compatibility_score(
            top.category_label, bottom.category_label,
            top.category, bottom.category,
        )

        # 5. Brightness balance (similar brightness = better match)
        brightness_diff = abs(tf.brightness - bf.brightness)
        brightness_score = max(0.0, 1.0 - brightness_diff * 2.0)

        raw = (
            self.W_STYLE      * style +
            self.W_COLOR      * color +
            self.W_OCCASION   * occasion_score +
            self.W_CATEGORY   * category_score +
            self.W_BRIGHTNESS * brightness_score
        )

        breakdown = {
            "style":      round(style, 3),
            "color":      round(color, 3),
            "occasion":   round(occasion_score, 3),
            "category":   round(category_score, 3),
            "brightness": round(brightness_score, 3),
        }

        return raw, breakdown

    def _build_explanation(self, breakdown: Dict[str, float], score: float, top: ClothingItem, bottom: ClothingItem) -> str:
        if not np.isfinite(score):
            return "Incomplete model data for this combination."

        parts = []
        top_label = (top.category_label or top.category or "").lower()
        bot_label = (bottom.category_label or bottom.category or "").lower()
        top_kw = get_category_keyword(top_label, top.category)
        bot_kw = get_category_keyword(bot_label, bottom.category)

        # Category compatibility signal
        cat_score = breakdown.get("category", 0)
        if cat_score >= 0.95:
            parts.append(f"classic pairing of {top_kw} and {bot_kw}")
        elif cat_score >= 0.80:
            parts.append(f"compatible {top_kw} and {bot_kw} pairing")

        # Color harmony signal
        color_score = breakdown.get("color", 0)
        if color_score >= 0.75:
            parts.append("great color harmony")
        elif color_score >= 0.55:
            parts.append("balanced color tones")

        # Style consistency signal
        style_score = breakdown.get("style", 0)
        if style_score >= 0.75:
            parts.append("consistent visual aesthetic")
        elif style_score >= 0.55:
            parts.append("complementary styling")

        # Occasion relevance
        occ_score = breakdown.get("occasion", 0)
        if occ_score >= 0.65:
            parts.append("well-suited for the occasion")

        # Brightness balance
        bright_score = breakdown.get("brightness", 0)
        if bright_score >= 0.80:
            parts.append("clean tonal balance")

        pct = round(score * 100)

        # Honest contextual framing based on score tier
        if not parts:
            if score >= 0.50:
                parts.append("a wearable everyday combination from your wardrobe")
            else:
                parts.append("an unconventional pairing to experiment with")

        if score >= 0.85:
            prefix = f"{pct}% match"
        elif score >= PROMOTION_THRESHOLD:
            prefix = f"{pct}% match"
        elif score >= 0.60:
            prefix = f"{pct}% match — solid option"
        elif score >= 0.40:
            prefix = f"{pct}% match — usable alternative"
        else:
            prefix = f"{pct}% match — experimental"

        return f"{prefix}: {', '.join(parts)}."

    def recommend_outfits(
        self,
        tops: List[ClothingItem],
        bottoms: List[ClothingItem],
        occasion: str = "casual",
        num_results: int = 5,
        exclude_rejected: bool = True,
        apply_threshold: bool = True,
    ) -> Tuple[List[OutfitRecommendation], bool]:
        """
        Score all valid (top, bottom) pairs and return top-N recommendations,
        ranked by compatibility score.

        The ranking-first approach:
        1. Validates clothing categories and pairs (excluding invalid/corrupt items).
        2. Evaluates each combination using multi-signal compatibility model.
        3. Deduplicates combinations and removes rejected pairs (if exclude_rejected).
        4. Ranks all valid combinations from highest to lowest compatibility.
        5. Applies diversity filtering across tops and bottoms to avoid repetitive looks.
        6. Returns the best available recommendations, regardless of whether their
           scores are above or below 75%.
        7. The 75% threshold is NEVER used to hide or filter valid outfits; it is
           used solely for presentation tier labeling ('is_promoted' / 'tier').

        Args:
            tops: List of top ClothingItems.
            bottoms: List of bottom ClothingItems.
            occasion: Occasion description string.
            num_results: Maximum number of recommendations to return.
            exclude_rejected: If True, skip previously rejected pairs.
            apply_threshold: Kept for API compatibility. The 75% threshold no
                             longer filters out valid combinations.

        Returns:
            Tuple of (recommendations_list, has_promoted_tier).
        """
        if not tops or not bottoms:
            return [], False

        occasion_emb = self._get_occasion_embedding(occasion)
        results: List[OutfitRecommendation] = []
        seen_pairs: Set[Tuple[str, str]] = set()

        for top in tops:
            # Validate top item
            if top is None or not getattr(top, 'id', None):
                continue
            if getattr(top, 'category', '') not in ('top', 'tops'):
                continue

            for bottom in bottoms:
                # Validate bottom item
                if bottom is None or not getattr(bottom, 'id', None):
                    continue
                if getattr(bottom, 'category', '') not in ('bottom', 'bottoms'):
                    continue

                # An item cannot be paired with itself
                if top.id == bottom.id:
                    continue

                # Deduplicate identical combinations
                pair_key = (top.id, bottom.id)
                if pair_key in seen_pairs:
                    continue

                if exclude_rejected and self.feedback_system.is_rejected(top.id, bottom.id):
                    continue

                raw, breakdown = self._score_pair(top, bottom, occasion_emb)

                # Guard against non-finite or invalid model outputs
                if not np.isfinite(raw):
                    continue

                modifier = self.feedback_system.get_score_modifier(top.id, bottom.id)
                final = max(0.0, min(1.0, raw + modifier))

                if not np.isfinite(final):
                    continue

                seen_pairs.add(pair_key)
                explanation = self._build_explanation(breakdown, final, top, bottom)
                results.append(OutfitRecommendation(
                    top=top,
                    bottom=bottom,
                    score=final,
                    breakdown=breakdown,
                    explanation=explanation,
                ))

        # Rank all valid combinations from highest to lowest score
        results.sort(key=lambda r: r.score, reverse=True)

        # Apply diversity filtering
        diverse_results = self._apply_diversity(results, num_results)

        has_promoted = any(r.is_promoted for r in diverse_results)
        return diverse_results, has_promoted

    def _apply_diversity(
        self,
        ranked: List[OutfitRecommendation],
        num_results: int,
    ) -> List[OutfitRecommendation]:
        """
        Select up to num_results outfits from the ranked list while
        promoting diversity across items and categories.

        Rules:
        - Outfits are processed in order of compatibility ranking.
        - Prevent the same top or bottom from dominating all recommendations
          when diverse valid alternatives exist.
        - Cap repetition to at most 2 occurrences of any single item in the
          recommendations, unless backfill is needed to satisfy the requested count.
        - Diversity NEVER excludes valid outfits if fewer alternatives exist;
          it degrades gracefully to show whatever the user owns (Scenario D).
        """
        if len(ranked) <= num_results:
            return ranked

        selected: List[OutfitRecommendation] = []
        top_counts: Dict[str, int] = {}
        bottom_counts: Dict[str, int] = {}
        seen_pairs: Set[Tuple[str, str]] = set()

        # Pass 1: select outfits with diversity constraints (max 2 per item)
        max_repeats = 2 if num_results > 2 else 1
        for rec in ranked:
            if len(selected) >= num_results:
                break
            pair = (rec.top.id, rec.bottom.id)
            if pair in seen_pairs:
                continue

            top_uses = top_counts.get(rec.top.id, 0)
            bot_uses = bottom_counts.get(rec.bottom.id, 0)

            if top_uses < max_repeats and bot_uses < max_repeats:
                selected.append(rec)
                seen_pairs.add(pair)
                top_counts[rec.top.id] = top_uses + 1
                bottom_counts[rec.bottom.id] = bot_uses + 1

        # Pass 2: backfill if diversity was too strict (e.g. user only has 1 top)
        if len(selected) < num_results:
            for rec in ranked:
                if len(selected) >= num_results:
                    break
                pair = (rec.top.id, rec.bottom.id)
                if pair not in seen_pairs:
                    selected.append(rec)
                    seen_pairs.add(pair)

        return selected

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
        recs, _ = self.recommend_outfits(
            tops, bottoms, occasion, num_results=20, exclude_rejected=True
        )
        for rec in recs:
            if rec.top.id != rejected_top_id or rec.bottom.id != rejected_bottom_id:
                return rec

        return None




