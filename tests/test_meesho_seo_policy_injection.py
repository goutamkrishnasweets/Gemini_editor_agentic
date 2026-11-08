"""
tests/test_meesho_seo_policy_injection.py
=========================================
Verification suite for Runtime JSON Policy Injection (Meesho Creator Club).
Tests:
  1. External configuration loading (_load_meesho_policy_config).
  2. Keyword detection in user_title, user_hint, video_context, brand_info, metadata.
  3. Non-injection when Meesho keywords are absent.
  4. Runtime policy injection into gen_evidence under external_policy.
  5. Complete, untruncated JSON serialization in _build_generation_prompt (EVIDENCE section).
  6. Propagation of external_policy to critique calls (_build_critic_prompt).
  7. Verification that templates (_GENERATE_TEMPLATE, _RULES, _CRITIC_TEMPLATE) are unmodified.
  8. Full pipeline execution with generate_platform_seo().
"""

import json
import unittest
from unittest.mock import patch, MagicMock

from Gemini_Modules.platform_seo_generator import (
    _load_meesho_policy_config,
    _detect_meesho_keywords,
    build_evidence,
    generate_platform_seo,
    _build_generation_prompt,
    _build_critic_prompt,
    _GENERATE_TEMPLATE,
    _CRITIC_TEMPLATE,
    _RULES,
    PLATFORM_SPECS,
)


class TestMeeshoSEOPolicyInjection(unittest.TestCase):

    def setUp(self):
        self.config = _load_meesho_policy_config()

    def test_01_external_config_loaded_completely(self):
        """Verify meesho_seo_gemini_config.json is loaded externally without truncation."""
        self.assertIsNotNone(self.config, "Meesho configuration failed to load externally.")
        self.assertIsInstance(self.config, dict)
        self.assertIn("module", self.config)
        self.assertIn("routing", self.config)
        self.assertIn("meesho_creator_policy", self.config)
        # Check specific nested keys to ensure complete loading
        self.assertEqual(
            self.config["module"].get("name"),
            "Meesho Creator SEO and Monetisation Compliance Profile"
        )
        self.assertIn(
            "mandatory_caption_and_tag_rules",
            self.config["meesho_creator_policy"]
        )

    def test_02_keyword_detection_in_user_inputs(self):
        """Verify Meesho-related keywords are detected in user_title, user_hint, video_context."""
        # A) In user_title
        self.assertTrue(_detect_meesho_keywords(user_title="Top Meesho Kurti Haul under 500"))
        self.assertTrue(_detect_meesho_keywords(user_title="Affordable styling with #meeshoapp"))

        # B) In user_hint
        self.assertTrue(_detect_meesho_keywords(user_hint="Use meesho creator club guidelines"))
        self.assertTrue(_detect_meesho_keywords(user_hint="tag @meeshoapp in the caption"))

        # C) In video_context
        self.assertTrue(_detect_meesho_keywords(video_context="The creator reviews a floral dress ordered from meesho"))

        # D) In brand_info or metadata
        self.assertTrue(_detect_meesho_keywords(brand_info="Meesho Creator"))
        self.assertTrue(_detect_meesho_keywords(metadata={"hashtags": ["fashion", "meeshohaul"]}))

        # E) Case insensitivity
        self.assertTrue(_detect_meesho_keywords(user_title="HUGE MEESHO HAUL"))
        self.assertTrue(_detect_meesho_keywords(user_hint="@MEESHOAPP"))

    def test_03_no_injection_when_meesho_keywords_absent(self):
        """Verify that when Meesho keywords are absent, policy is NOT injected."""
        # Generic title, hint, and context with no Meesho mention
        is_detected = _detect_meesho_keywords(
            user_title="Zara Summer Dress Collection",
            user_hint="Focus on elegant evening wear",
            video_context="High-fashion strut on runway with cocktail dress",
            brand_info="Zara"
        )
        self.assertFalse(is_detected, "False positive Meesho keyword detection.")

        ev = build_evidence(
            video_context="High-fashion strut on runway with cocktail dress",
            user_title="Zara Summer Dress Collection",
            brand_info="Zara",
            user_hint="Focus on elegant evening wear"
        )
        self.assertNotIn("external_policy", ev, "external_policy was injected when keywords were absent!")

        # Verify generation prompt contains no external_policy
        prompt = _build_generation_prompt(ev, ["instagram", "youtube"], {"instagram": "dm_keyword", "youtube": "inline"})
        self.assertNotIn("external_policy", prompt)
        self.assertNotIn("Meesho Creator SEO and Monetisation Compliance Profile", prompt)

    def test_04_injection_into_evidence_and_generation_prompt(self):
        """Verify runtime injection into gen_evidence under external_policy and serialization."""
        ev = build_evidence(
            video_context="Trying on kurti haul from meesho creator store",
            user_title="Budget Ethnic Wear Haul",
            user_hint="Include #meesho in tags"
        )
        self.assertIn("external_policy", ev, "external_policy missing from evidence when Meesho keywords present.")
        self.assertIsInstance(ev["external_policy"], dict)
        self.assertEqual(
            ev["external_policy"]["module"]["name"],
            "Meesho Creator SEO and Monetisation Compliance Profile"
        )

        # Build generation prompt and verify complete untruncated serialization into EVIDENCE
        modes = {"instagram": "dm_keyword", "youtube": "inline"}
        gen_prompt = _build_generation_prompt(ev, ["instagram", "youtube"], modes)

        self.assertIn("EVIDENCE", gen_prompt)
        self.assertIn('"external_policy":', gen_prompt)
        self.assertIn("Meesho Creator SEO and Monetisation Compliance Profile", gen_prompt)
        self.assertIn("mandatory_caption_and_tag_rules", gen_prompt)
        self.assertIn("@meeshoapp", gen_prompt)

    def test_05_propagation_to_critique_calls(self):
        """Verify external_policy is available in subsequent Gemini critique calls."""
        ev = build_evidence(
            video_context="Unboxing meesho jewellery pieces",
            user_title="Meesho Jewellery under 200"
        )
        self.assertIn("external_policy", ev)

        analysis = {"primary_keyword": "meesho jewellery", "niche": "fashion"}
        draft = {
            "platforms": {
                "instagram": {
                    "title": "Jewellery Haul",
                    "description": "Amazing finds from @meeshoapp #meesho",
                    "keywords": ["meesho jewellery"],
                    "hashtags": ["#meesho"]
                }
            }
        }
        lint = {"instagram": []}
        modes = {"instagram": "dm_keyword"}

        critic_prompt = _build_critic_prompt(ev, analysis, draft, lint, ["instagram"], modes)
        self.assertIn("EVIDENCE", critic_prompt)
        self.assertIn('"external_policy":', critic_prompt)
        self.assertIn("Meesho Creator Club", critic_prompt)

    def test_06_templates_unmodified(self):
        """Verify _GENERATE_TEMPLATE, _RULES, and _CRITIC_TEMPLATE were not modified."""
        # Ensure standard placeholders remain unmodified
        self.assertIn("__RULES__", _GENERATE_TEMPLATE)
        self.assertIn("__BRIEFS__", _GENERATE_TEMPLATE)
        self.assertIn("__EVIDENCE__", _GENERATE_TEMPLATE)
        self.assertIn("__PLATFORMS__", _GENERATE_TEMPLATE)

        self.assertIn("__RULES__", _CRITIC_TEMPLATE)
        self.assertIn("__BRIEFS__", _CRITIC_TEMPLATE)
        self.assertIn("__EVIDENCE__", _CRITIC_TEMPLATE)
        self.assertIn("__ANALYSIS__", _CRITIC_TEMPLATE)
        self.assertIn("__DRAFT__", _CRITIC_TEMPLATE)
        self.assertIn("__LINT__", _CRITIC_TEMPLATE)

        # No hardcoded Meesho text inside the template strings themselves
        self.assertNotIn("meesho", _GENERATE_TEMPLATE.lower())
        self.assertNotIn("meesho", _CRITIC_TEMPLATE.lower())
        self.assertNotIn("meesho", _RULES.lower())

    @patch("Gemini_Modules.platform_seo_generator._llm_json")
    def test_07_full_pipeline_generate_platform_seo(self, mock_llm):
        """Verify generate_platform_seo runs with Meesho keyword detection and injects external_policy into gen_evidence."""
        captured_evidence = {}

        def mock_llm_json(prompt, temp, tag):
            if tag == "generate":
                # Verify prompt contains external_policy
                self.assertIn('"external_policy":', prompt)
                self.assertIn("Meesho Creator SEO and Monetisation Compliance Profile", prompt)
                return {
                    "analysis": {
                        "primary_keyword": "meesho kurti haul",
                        "secondary_keywords": ["budget ethnic wear"],
                        "niche": "ethnic fashion",
                        "commercial_intent": "deal",
                        "funnel_stage": "consideration",
                        "buyer_queries": ["best meesho kurti"],
                        "value_props": ["pure cotton", "under 499"],
                        "objections": ["quality"],
                        "transformation_angle": "honest review",
                        "audience": "budget shoppers",
                        "hook_angle": "price drop",
                        "on_screen_text": ["Under Rs 499!"]
                    },
                    "platforms": {
                        "instagram": {
                            "title": "Unreal Meesho Kurti Find",
                            "title_alternates": ["Best Budget Kurti", "Meesho Ethnic Wear"],
                            "description": "Found this pure cotton kurti on Meesho for under 499. Fabric is surprisingly breathable.\n\nShop details below. #ad",
                            "keywords": ["meesho kurti", "ethnic haul"],
                            "hashtags": ["#meesho", "#meeshohaul", "#kurtis", "#ethnicwear", "#fashion"]
                        }
                    },
                    "content_category": "fashion",
                    "target_audience": "women shoppers"
                }
            return None

        mock_llm.side_effect = mock_llm_json

        res = generate_platform_seo(
            video_context="Reviewing a pastel yellow kurti bought from Meesho",
            user_title="Honest Meesho Kurti Review",
            user_hint="Check stitching and fabric quality",
            platforms=["instagram"],
            optimize_rounds=0,
        )

        self.assertIsNotNone(res)
        self.assertIn("platforms", res)
        self.assertIn("instagram", res["platforms"])
        # Verify result retained external_policy
        self.assertIn("external_policy", res)
        self.assertEqual(
            res["external_policy"]["module"]["name"],
            "Meesho Creator SEO and Monetisation Compliance Profile"
        )


if __name__ == "__main__":
    unittest.main()
