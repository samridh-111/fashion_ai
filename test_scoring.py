"""
Automated tests for Fashion AI recommendation scoring pipeline.

Tests cover all 11 requirements from STEP 8:
  1. Outfits scoring 90% are displayed.
  2. Outfits scoring 75% are displayed.
  3. Outfits scoring 74% are displayed.
  4. Outfits scoring below 75% can still be recommended.
  5. Recommendations are ranked correctly.
  6. Duplicate outfits are handled appropriately.
  7. Invalid combinations are excluded.
  8. A wardrobe with no high-scoring outfits still returns its best valid alternatives.
  9. A genuinely empty or invalid wardrobe is handled correctly.
  10. Missing, non-finite, or invalid model scores do not produce fabricated confidence percentages.
  11. Backend filtering and frontend rendering behave consistently.
Plus existing tests for feature helpers, category compatibility, feedback, and score clamping.
"""

import sys
import os
import numpy as np
import unittest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass, field
from typing import Optional, List, Dict

# Ensure we import from the project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from recommendation_engine import (
    RecommendationEngine,
    FeedbackSystem,
    OutfitRecommendation,
    PROMOTION_THRESHOLD,
    cosine_sim,
    color_harmony_score,
    category_compatibility_score,
)
from ml_features import ClothingItem, ClothingFeatures


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def make_features(
    visual: Optional[np.ndarray] = None,
    color: Optional[np.ndarray] = None,
    brightness: float = 0.5,
    text: Optional[np.ndarray] = None,
) -> ClothingFeatures:
    """Create a ClothingFeatures object for testing."""
    return ClothingFeatures(
        visual_embedding=visual,
        color_histogram=color,
        brightness=brightness,
        saturation=0.5,
        text_embedding=text,
        dominant_colors=[[200, 150, 100]],
        embedding_dim=len(visual) if visual is not None else 0,
    )


def make_item(
    item_id: str = "test-id",
    category: str = "top",
    category_label: str = "",
    visual: Optional[np.ndarray] = None,
    brightness: float = 0.5,
) -> ClothingItem:
    """Create a ClothingItem for testing."""
    vis = visual if visual is not None else np.random.rand(256)
    vis = vis / (np.linalg.norm(vis) + 1e-9)
    return ClothingItem(
        id=item_id,
        filename=f"{item_id}.jpg",
        path=f"user_clothes/{category}s/{item_id}.jpg",
        category=category,
        category_label=category_label or f"{category} item",
        features=make_features(
            visual=vis,
            color=np.random.rand(96),
            brightness=brightness,
            text=np.random.rand(384),
        ),
    )


def make_engine() -> RecommendationEngine:
    """Create a RecommendationEngine with mocked feature extractor."""
    mock_fe = MagicMock()
    mock_fe.extract_text_features.return_value = np.random.rand(384)
    engine = RecommendationEngine(mock_fe)
    # Avoid disk I/O for feedback in tests
    engine.feedback_system = FeedbackSystem.__new__(FeedbackSystem)
    engine.feedback_system.liked_pairs = set()
    engine.feedback_system.rejected_pairs = set()
    engine.feedback_system.feedback_history = []
    engine.feedback_system.feedback_file = "/tmp/test_feedback.json"
    return engine


# ---------------------------------------------------------------------------
# STEP 8 Required Tests: 1 through 11
# ---------------------------------------------------------------------------

class TestStep8RecommendationPipeline(unittest.TestCase):
    """
    Direct verification of the 11 requirements from STEP 8.
    """

    def test_01_outfits_scoring_90_displayed(self):
        """1. Outfits scoring 90% are displayed."""
        engine = make_engine()
        top = make_item("t1", "top")
        bottom = make_item("b1", "bottom")

        with patch.object(engine, '_score_pair', return_value=(0.90, {"style": 0.9, "color": 0.9})):
            recs, has_promoted = engine.recommend_outfits([top], [bottom])

        self.assertEqual(len(recs), 1)
        self.assertAlmostEqual(recs[0].percentage, 90.0, places=1)
        self.assertTrue(recs[0].is_promoted)
        self.assertEqual(recs[0].tier, "strong")
        self.assertEqual(recs[0].tier_label, "Strong Match")
        self.assertTrue(has_promoted)

    def test_02_outfits_scoring_75_displayed(self):
        """2. Outfits scoring 75% are displayed."""
        engine = make_engine()
        top = make_item("t1", "top")
        bottom = make_item("b1", "bottom")

        with patch.object(engine, '_score_pair', return_value=(0.75, {"style": 0.75, "color": 0.75})):
            recs, has_promoted = engine.recommend_outfits([top], [bottom])

        self.assertEqual(len(recs), 1)
        self.assertAlmostEqual(recs[0].percentage, 75.0, places=1)
        self.assertTrue(recs[0].is_promoted)
        self.assertEqual(recs[0].tier, "good")
        self.assertEqual(recs[0].tier_label, "Good Match")
        self.assertTrue(has_promoted)

    def test_03_outfits_scoring_74_displayed(self):
        """3. Outfits scoring 74% are displayed (NOT filtered out by 75% cutoff)."""
        engine = make_engine()
        top = make_item("t1", "top")
        bottom = make_item("b1", "bottom")

        with patch.object(engine, '_score_pair', return_value=(0.74, {"style": 0.74, "color": 0.74})):
            recs, has_promoted = engine.recommend_outfits([top], [bottom])

        self.assertEqual(len(recs), 1)
        self.assertAlmostEqual(recs[0].percentage, 74.0, places=1)
        # 74% is below promotion threshold for presentation labeling, but must be displayed!
        self.assertFalse(recs[0].is_promoted)
        self.assertEqual(recs[0].tier, "moderate")
        self.assertEqual(recs[0].tier_label, "Worth Considering")

    def test_04_outfits_scoring_below_75_can_still_be_recommended(self):
        """4. Outfits scoring below 75% can still be recommended."""
        engine = make_engine()
        top = make_item("t1", "top")
        bottom = make_item("b1", "bottom")

        for low_score, expected_tier in [(0.64, "moderate"), (0.48, "alternative"), (0.35, "experimental")]:
            with patch.object(engine, '_score_pair', return_value=(low_score, {"style": low_score})):
                recs, _ = engine.recommend_outfits([top], [bottom])

            self.assertEqual(len(recs), 1, f"Score {low_score} was improperly filtered out!")
            self.assertAlmostEqual(recs[0].percentage, round(low_score * 100, 1), places=1)
            self.assertEqual(recs[0].tier, expected_tier)

    def test_05_recommendations_ranked_correctly(self):
        """5. Recommendations are ranked correctly from highest to lowest compatibility."""
        engine = make_engine()
        tops = [make_item(f"t{i}", "top") for i in range(5)]
        bottom = make_item("b1", "bottom")

        scores = {"t0": 0.48, "t1": 0.91, "t2": 0.64, "t3": 0.83, "t4": 0.72}

        def mock_score(t, b, occ):
            return scores[t.id], {"style": scores[t.id]}

        with patch.object(engine, '_score_pair', side_effect=mock_score):
            recs, _ = engine.recommend_outfits(tops, [bottom], num_results=5)

        self.assertEqual(len(recs), 5)
        # Should be ordered: 0.91, 0.83, 0.72, 0.64, 0.48
        result_scores = [r.score for r in recs]
        expected_scores = [0.91, 0.83, 0.72, 0.64, 0.48]
        self.assertEqual(result_scores, expected_scores)

    def test_06_duplicate_outfits_handled(self):
        """6. Duplicate outfits are handled appropriately."""
        engine = make_engine()
        # Same top and bottom passed twice in list
        top = make_item("t1", "top")
        bottom = make_item("b1", "bottom")

        with patch.object(engine, '_score_pair', return_value=(0.80, {})):
            recs, _ = engine.recommend_outfits([top, top], [bottom, bottom], num_results=5)

        # Must deduplicate to only 1 recommendation for the unique pair (t1, b1)
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0].top.id, "t1")
        self.assertEqual(recs[0].bottom.id, "b1")

    def test_07_invalid_combinations_excluded(self):
        """7. Invalid combinations are excluded."""
        engine = make_engine()

        # Item paired with itself
        item_both = make_item("item-1", "top")
        item_both_as_bottom = make_item("item-1", "bottom")
        recs, _ = engine.recommend_outfits([item_both], [item_both_as_bottom])
        self.assertEqual(len(recs), 0, "Item paired with itself should be excluded")

        # Top item that is mistakenly categorized as bottom
        invalid_top = make_item("bad_top", "bottom")
        valid_bottom = make_item("b1", "bottom")
        recs, _ = engine.recommend_outfits([invalid_top], [valid_bottom])
        self.assertEqual(len(recs), 0, "Invalid top category must be excluded")

        # Bottom item that is mistakenly categorized as top
        valid_top = make_item("t1", "top")
        invalid_bottom = make_item("bad_bottom", "top")
        recs, _ = engine.recommend_outfits([valid_top], [invalid_bottom])
        self.assertEqual(len(recs), 0, "Invalid bottom category must be excluded")

    def test_08_wardrobe_with_no_high_scoring_outfits_returns_best_alternatives(self):
        """8. A wardrobe with no high-scoring outfits still returns its best valid alternatives (Scenario C)."""
        engine = make_engine()
        tops = [make_item("t1", "top"), make_item("t2", "top")]
        bottoms = [make_item("b1", "bottom")]

        # All scores are strictly below 75%
        scores = {("t1", "b1"): 0.68, ("t2", "b1"): 0.52}

        def mock_score(t, b, occ):
            return scores[(t.id, b.id)], {"style": scores[(t.id, b.id)]}

        with patch.object(engine, '_score_pair', side_effect=mock_score):
            recs, has_promoted = engine.recommend_outfits(tops, bottoms, num_results=5)

        self.assertEqual(len(recs), 2, "Must return valid alternatives even if all are < 75%!")
        self.assertFalse(has_promoted)
        self.assertEqual(recs[0].top.id, "t1")
        self.assertAlmostEqual(recs[0].percentage, 68.0, places=1)
        self.assertEqual(recs[1].top.id, "t2")
        self.assertAlmostEqual(recs[1].percentage, 52.0, places=1)

    def test_09_genuinely_empty_or_invalid_wardrobe(self):
        """9. A genuinely empty or invalid wardrobe is handled correctly (Scenario E)."""
        engine = make_engine()

        # Completely empty
        recs, has_promoted = engine.recommend_outfits([], [])
        self.assertEqual(recs, [])
        self.assertFalse(has_promoted)

        # No bottoms
        recs, has_promoted = engine.recommend_outfits([make_item("t1", "top")], [])
        self.assertEqual(recs, [])
        self.assertFalse(has_promoted)

        # No tops
        recs, has_promoted = engine.recommend_outfits([], [make_item("b1", "bottom")])
        self.assertEqual(recs, [])
        self.assertFalse(has_promoted)

        # All combinations rejected
        engine.feedback_system.rejected_pairs = {("t1", "b1")}
        recs, has_promoted = engine.recommend_outfits(
            [make_item("t1", "top")],
            [make_item("b1", "bottom")],
            exclude_rejected=True,
        )
        self.assertEqual(recs, [])
        self.assertFalse(has_promoted)

    def test_10_missing_or_non_finite_scores_no_fabricated_percentages(self):
        """10. Missing, non-finite, or invalid model scores do not produce fabricated confidence percentages."""
        engine = make_engine()
        top = make_item("t1", "top")
        bottom = make_item("b1", "bottom")

        # Test OutfitRecommendation dataclass directly with NaN
        rec_nan = OutfitRecommendation(top=top, bottom=bottom, score=float('nan'))
        self.assertEqual(rec_nan.percentage, 0.0)
        self.assertFalse(rec_nan.is_promoted)
        self.assertEqual(rec_nan.tier, "invalid")
        self.assertEqual(rec_nan.tier_label, "Unscored Combination")

        # Test engine skipping non-finite raw outputs rather than inventing scores
        with patch.object(engine, '_score_pair', return_value=(float('nan'), {})):
            recs, _ = engine.recommend_outfits([top], [bottom])
            self.assertEqual(recs, [], "Non-finite score must not produce fabricated recommendations")

        with patch.object(engine, '_score_pair', return_value=(float('inf'), {})):
            recs, _ = engine.recommend_outfits([top], [bottom])
            self.assertEqual(recs, [], "Inf score must not produce fabricated recommendations")

    def test_11_backend_filtering_and_frontend_consistency(self):
        """11. Backend filtering and frontend rendering behave consistently."""
        engine = make_engine()
        tops = [make_item("t1", "top"), make_item("t2", "top"), make_item("t3", "top")]
        bottoms = [make_item("b1", "bottom")]

        scores = {"t1": 0.88, "t2": 0.72, "t3": 0.55}

        def mock_score(t, b, occ):
            return scores[t.id], {"style": scores[t.id], "color": 0.8, "category": 0.9}

        with patch.object(engine, '_score_pair', side_effect=mock_score):
            recs, _ = engine.recommend_outfits(tops, bottoms, num_results=5)

        # Simulate backend API response transformation
        api_results = []
        for rec in recs:
            api_results.append({
                "match_score": rec.percentage,
                "is_promoted": rec.is_promoted,
                "tier": rec.tier,
                "tier_label": rec.tier_label,
                "explanation": rec.explanation,
            })

        # All 3 must be returned
        self.assertEqual(len(api_results), 3)

        # Top match (0.88): promoted=True, tier=strong
        self.assertEqual(api_results[0]["match_score"], 88.0)
        self.assertTrue(api_results[0]["is_promoted"])
        self.assertEqual(api_results[0]["tier"], "strong")
        self.assertEqual(api_results[0]["tier_label"], "Strong Match")

        # Alternative 1 (0.72): promoted=False, tier=moderate (NOT filtered out!)
        self.assertEqual(api_results[1]["match_score"], 72.0)
        self.assertFalse(api_results[1]["is_promoted"])
        self.assertEqual(api_results[1]["tier"], "moderate")
        self.assertEqual(api_results[1]["tier_label"], "Worth Considering")

        # Alternative 2 (0.55): promoted=False, tier=alternative
        self.assertEqual(api_results[2]["match_score"], 55.0)
        self.assertFalse(api_results[2]["is_promoted"])
        self.assertEqual(api_results[2]["tier"], "alternative")

        # Frontend tier grouping simulation:
        promoted = [r for r in api_results if r["is_promoted"]]
        alternatives = [r for r in api_results if not r["is_promoted"]]
        self.assertEqual(len(promoted), 1)
        self.assertEqual(len(alternatives), 2)


# ---------------------------------------------------------------------------
# Diversity & Graceful Degradation Tests
# ---------------------------------------------------------------------------

class TestDiversityAndDegradation(unittest.TestCase):
    """Test diversity handling and small wardrobe degradation."""

    def test_diversity_limits_overrepresentation(self):
        """Diversity prevents a single item from dominating all recommendations when alternatives exist."""
        engine = make_engine()
        # 1 versatile top, 1 other top, 4 bottoms
        top_versatile = make_item("top_versatile", "top")
        top_alt = make_item("top_alt", "top")
        bottoms = [make_item(f"b{i}", "bottom") for i in range(4)]

        # versatile top scores slightly higher with all bottoms
        def mock_score(t, b, occ):
            base = 0.85 if t.id == "top_versatile" else 0.82
            return base, {}

        with patch.object(engine, '_score_pair', side_effect=mock_score):
            recs, _ = engine.recommend_outfits([top_versatile, top_alt], bottoms, num_results=4)

        self.assertEqual(len(recs), 4)
        top_ids = [r.top.id for r in recs]
        # Should include both top_versatile and top_alt, not exclusively top_versatile
        self.assertIn("top_versatile", top_ids)
        self.assertIn("top_alt", top_ids)
        # top_versatile should not be repeated more than 2 times
        self.assertLessEqual(top_ids.count("top_versatile"), 2)

    def test_scenario_d_small_wardrobe_graceful_degradation(self):
        """Scenario D: Very few wardrobe items (e.g. 1 top and 2 bottoms) generates all possible combinations."""
        engine = make_engine()
        tops = [make_item("only_top", "top")]
        bottoms = [make_item("b1", "bottom"), make_item("b2", "bottom")]

        with patch.object(engine, '_score_pair', return_value=(0.60, {})):
            recs, _ = engine.recommend_outfits(tops, bottoms, num_results=5)

        self.assertEqual(len(recs), 2, "Must return all available combinations instead of demanding more items")


# ---------------------------------------------------------------------------
# Existing Component Tests
# ---------------------------------------------------------------------------

class TestCosineSim(unittest.TestCase):
    """Test cosine similarity helper."""

    def test_identical_vectors(self):
        v = np.array([1.0, 2.0, 3.0])
        self.assertAlmostEqual(cosine_sim(v, v), 1.0)

    def test_orthogonal_vectors(self):
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        self.assertAlmostEqual(cosine_sim(v1, v2), 0.0)

    def test_none_inputs_return_zero(self):
        self.assertEqual(cosine_sim(None, np.array([1.0])), 0.0)
        self.assertEqual(cosine_sim(np.array([1.0]), None), 0.0)
        self.assertEqual(cosine_sim(None, None), 0.0)

    def test_zero_vector_returns_zero(self):
        self.assertEqual(cosine_sim(np.zeros(3), np.array([1.0, 2.0, 3.0])), 0.0)


class TestCategoryCompatibility(unittest.TestCase):
    """Test category compatibility scoring rules."""

    def test_known_good_pair(self):
        score = category_compatibility_score("shirt item", "jeans item", "top", "bottom")
        self.assertGreaterEqual(score, 0.9)

    def test_known_ok_pair(self):
        score = category_compatibility_score("blazer clothing", "jeans clothing", "top", "bottom")
        self.assertGreaterEqual(score, 0.8)

    def test_generic_top_bottom_is_neutral(self):
        score = category_compatibility_score("top item", "bottom item", "top", "bottom")
        self.assertAlmostEqual(score, 0.70, places=1)

    def test_unknown_combo_returns_low_score(self):
        score = category_compatibility_score("something weird", "more weirdness", "unknown", "unknown")
        self.assertLessEqual(score, 0.55)


class TestFeedbackSystem(unittest.TestCase):
    """Test feedback scoring modifiers."""

    def test_liked_pair_gets_positive_modifier(self):
        fb = FeedbackSystem.__new__(FeedbackSystem)
        fb.liked_pairs = {("t1", "b1")}
        fb.rejected_pairs = set()
        fb.feedback_history = []
        self.assertEqual(fb.get_score_modifier("t1", "b1"), 0.10)

    def test_rejected_pair_gets_negative_modifier(self):
        fb = FeedbackSystem.__new__(FeedbackSystem)
        fb.liked_pairs = set()
        fb.rejected_pairs = {("t1", "b1")}
        fb.feedback_history = []
        self.assertEqual(fb.get_score_modifier("t1", "b1"), -0.30)

    def test_unseen_pair_gets_zero_modifier(self):
        fb = FeedbackSystem.__new__(FeedbackSystem)
        fb.liked_pairs = set()
        fb.rejected_pairs = set()
        fb.feedback_history = []
        self.assertEqual(fb.get_score_modifier("t1", "b1"), 0.0)

    def test_is_rejected(self):
        fb = FeedbackSystem.__new__(FeedbackSystem)
        fb.liked_pairs = set()
        fb.rejected_pairs = {("t1", "b1")}
        fb.feedback_history = []
        self.assertTrue(fb.is_rejected("t1", "b1"))
        self.assertFalse(fb.is_rejected("t2", "b2"))


class TestScoreClamp(unittest.TestCase):
    """Test that scores are clamped to [0, 1]."""

    def test_liked_boost_does_not_exceed_one(self):
        """A liked pair with high raw score should be clamped to 1.0."""
        engine = make_engine()
        engine.feedback_system.liked_pairs = {("t1", "b1")}

        tops = [make_item("t1", "top")]
        bottoms = [make_item("b1", "bottom")]

        with patch.object(engine, '_score_pair', return_value=(0.98, {})):
            recs, _ = engine.recommend_outfits(tops=tops, bottoms=bottoms)

        self.assertLessEqual(recs[0].score, 1.0)

    def test_rejected_penalty_does_not_go_below_zero(self):
        """A rejected pair with low raw score should be clamped to 0.0."""
        engine = make_engine()
        engine.feedback_system.rejected_pairs = {("t1", "b1")}

        tops = [make_item("t1", "top")]
        bottoms = [make_item("b1", "bottom")]

        with patch.object(engine, '_score_pair', return_value=(0.10, {})):
            recs, _ = engine.recommend_outfits(
                tops=tops, bottoms=bottoms,
                exclude_rejected=False,
            )

        self.assertGreaterEqual(recs[0].score, 0.0)


class TestScoreBreakdownConsistency(unittest.TestCase):
    """Test that breakdown keys are present and values are in [0, 1]."""

    def test_breakdown_values_in_valid_range(self):
        engine = make_engine()
        tops = [make_item("t1", "top", "kaos")]
        bottoms = [make_item("b1", "bottom", "jeans")]

        occasion_emb = np.random.rand(384)
        _, breakdown = engine._score_pair(tops[0], bottoms[0], occasion_emb)

        for key, val in breakdown.items():
            self.assertGreaterEqual(val, 0.0, f"{key} below 0")
            self.assertLessEqual(val, 1.0, f"{key} above 1")

    def test_breakdown_contains_expected_keys(self):
        engine = make_engine()
        tops = [make_item("t1", "top")]
        bottoms = [make_item("b1", "bottom")]
        occasion_emb = np.random.rand(384)
        _, breakdown = engine._score_pair(tops[0], bottoms[0], occasion_emb)

        expected_keys = {"style", "color", "occasion", "category", "brightness"}
        self.assertEqual(set(breakdown.keys()), expected_keys)


class TestAlternativeOutfit(unittest.TestCase):
    """Test alternative outfit generation."""

    def test_alternative_returns_next_best(self):
        engine = make_engine()
        tops = [make_item("t1", "top"), make_item("t2", "top")]
        bottoms = [make_item("b1", "bottom")]

        scores = {("t1", "b1"): 0.85, ("t2", "b1"): 0.70}

        def mock_score(t, b, occ):
            return scores[(t.id, b.id)], {}

        with patch.object(engine, '_score_pair', side_effect=mock_score):
            alt = engine.get_alternative_outfit("t1", "b1", tops, bottoms)

        self.assertIsNotNone(alt)
        self.assertEqual(alt.top.id, "t2")
        self.assertEqual(alt.bottom.id, "b1")


if __name__ == "__main__":
    print("Running Fashion AI scoring tests...")
    result = unittest.main(verbosity=2, exit=False)
    if result.result.wasSuccessful():
        print("\nAll tests passed successfully.")
        sys.exit(0)
    else:
        failures = len(result.result.failures) + len(result.result.errors)
        print(f"\n{failures} test(s) failed.")
        sys.exit(1)
