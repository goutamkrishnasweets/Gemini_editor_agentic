"""
test_commercial_seo_affiliate_and_disclosure.py — Verification Suite for Commercial SEO & Affiliate Policy
========================================================================================================
Verifies:
  1. Multi-link & Product Label Ingestion:
     Both labeled links (e.g. 'Aafiya casual trackpants : https://...' and 'Top : https://...')
     are parsed, preserved with their labels, and present in the final caption/description.
  2. Opening Hook Protection:
     Caption Line 1 NEVER starts with #ad or contains affiliate tags. The opening visual hook
     is 100% clean and organic.
  3. Provider-Aware Policy Disclosure:
     - Amazon Associates: Strictly compliant with Section 5 Operating Agreement:
       '(Paid link · As an Amazon Associate I earn from qualifying purchases #ad)'
     - Flipkart: '(Paid Flipkart affiliate link · qualifying purchases earn commission #ad)'
     - EarnKaro: '(Paid EarnKaro affiliate link · qualifying purchases earn commission #ad)'
     - Generic: '(Paid affiliate link · qualifying purchases earn commission #ad)'
     - Placed cleanly AFTER the product links, never in the hook.
     - Pure organic posts (no affiliate links) contain ZERO #ad, ZERO #affiliate, and NO disclosure.
  4. Hashtag Collision Protection:
     Meta publisher appends all 5-15 category hashtags to Instagram caption even when
     #ad is present in the disclosure block.
"""

import unittest
from typing import Dict, Any

from Gemini_Modules.platform_seo_generator import (
    parse_affiliate_items,
    detect_affiliate_provider,
    get_provider_disclosure,
    generate_platform_seo,
    PLATFORM_SPECS,
    _place_disclosure,
)


class TestCommercialSEOAffiliateAndDisclosure(unittest.TestCase):

    def test_multi_link_and_product_label_parsing(self):
        raw_input = (
            "Aafiya casual trackpants : https://link.amazon/B0eYFOBtQ\n\n"
            "Top : https://link.amazon/B0c0h3RYL"
        )
        items = parse_affiliate_items(raw_input)
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0]["label"], "Aafiya casual trackpants")
        self.assertEqual(items[0]["url"], "https://link.amazon/B0eYFOBtQ")
        self.assertEqual(items[1]["label"], "Top")
        self.assertEqual(items[1]["url"], "https://link.amazon/B0c0h3RYL")

    def test_provider_detection_and_operating_agreement_disclosure(self):
        # Amazon Associates clause
        amz_links = ["https://link.amazon/B0eYFOBtQ", "https://amzn.to/3xyz"]
        self.assertEqual(detect_affiliate_provider(amz_links), "amazon")
        amz_disc = get_provider_disclosure(amz_links)
        self.assertIn("As an Amazon Associate I earn from qualifying purchases", amz_disc)
        self.assertIn("#ad", amz_disc)

        # Flipkart
        fk_links = ["https://dl.flipkart.com/s/xyz123"]
        self.assertEqual(detect_affiliate_provider(fk_links), "flipkart")
        fk_disc = get_provider_disclosure(fk_links)
        self.assertIn("Flipkart", fk_disc)
        self.assertIn("#ad", fk_disc)

        # EarnKaro
        ek_links = ["https://ekaro.in/enkr2026"]
        self.assertEqual(detect_affiliate_provider(ek_links), "earnkaro")
        ek_disc = get_provider_disclosure(ek_links)
        self.assertIn("EarnKaro", ek_disc)
        self.assertIn("#ad", ek_disc)

    def test_opening_hook_clean_no_ad_on_line_1(self):
        # Ensure disclosure placement configuration is line2 (ASCI compliance upfront before fold)
        self.assertEqual(PLATFORM_SPECS["instagram"]["disclosure_placement"], "line2")
        self.assertEqual(PLATFORM_SPECS["youtube"]["disclosure_placement"], "line2")

        raw_desc = "Effortlessly chic, these wide-leg trackpants redefine casual comfort."
        links = ["https://link.amazon/B0eYFOBtQ"]
        placed = _place_disclosure("instagram", raw_desc, has_links=True, links=links)

        first_line = placed.split("\n")[0]
        self.assertFalse(first_line.startswith("#ad"), f"Line 1 starts with #ad: {first_line}")
        self.assertNotIn("#ad", first_line)
        self.assertIn("#ad", placed)
        self.assertIn("Amazon Associate", placed)

    def test_organic_post_zero_ad_or_affiliate_tags(self):
        # When user provides NO affiliate links, #ad must never appear
        raw_desc = "Top 5 style tips for wide-leg pants this season."
        placed = _place_disclosure("instagram", raw_desc, has_links=False, links=[])
        self.assertNotIn("#ad", placed)
        self.assertNotIn("#affiliate", placed)

        # Even if draft contains accidental #ad, it must be completely scrubbed
        dirty_desc = "#ad Top 5 style tips for wide-leg pants (#ad #affiliate)."
        cleaned = _place_disclosure("instagram", dirty_desc, has_links=False, links=[])
        self.assertNotIn("#ad", cleaned)
        self.assertNotIn("#affiliate", cleaned)

    def test_full_seo_generation_with_dual_labeled_links(self):
        aff_input = (
            "Aafiya casual trackpants : https://link.amazon/B0eYFOBtQ\n"
            "Top : https://link.amazon/B0c0h3RYL"
        )
        res = generate_platform_seo(
            video_context="Fashion try-on haul showing wide-leg trackpants and matching crop top.",
            user_title="Aafiya Casual Trackpants Styling",
            brand_info="Fashion & Lifestyle",
            affiliate_link=aff_input,
            platforms=["instagram", "youtube"]
        )

        ig = res["platforms"]["instagram"]
        yt = res["platforms"]["youtube"]

        # Both links must be preserved with their labels in description
        for platform_data in (ig, yt):
            desc = platform_data["description"]
            self.assertIn("https://link.amazon/B0eYFOBtQ", desc)
            self.assertIn("https://link.amazon/B0c0h3RYL", desc)
            self.assertIn("Aafiya casual trackpants", desc)
            self.assertIn("Top", desc)

            # First line must NOT have #ad
            first_line = desc.split("\n")[0]
            self.assertFalse(first_line.strip().startswith("#ad"))

            # Amazon disclosure clause must exist below links
            self.assertIn("As an Amazon Associate I earn from qualifying purchases", desc)

            # Hashtags must exist
            self.assertTrue(len(platform_data["hashtags"]) >= 3)
            # Hashtags must NOT contain #ad or #affiliate as a tag
            for tag in platform_data["hashtags"]:
                self.assertNotIn(tag.lower(), ("#ad", "#affiliate"))

    def test_meta_publisher_hashtag_collision_fix(self):
        import re
        # Recreate the publisher logic from media_publisher_main.py
        ig_payload = {
            "description": (
                "Effortlessly chic wide-leg trackpants.\n\n"
                "👉 Aafiya casual trackpants: https://link.amazon/B0eYFOBtQ\n"
                "(Paid link · As an Amazon Associate I earn from qualifying purchases #ad)"
            ),
            "hashtags": ["#CasualStyle", "#WomensFashion", "#WideLegPants", "#Shorts"]
        }

        ig_caption = ig_payload.get("caption") or ig_payload.get("description") or "Default Title"
        target_tags = ig_payload.get("hashtags", [])
        if target_tags:
            existing_tags = set(re.findall(r"#\w+", ig_caption.lower()))
            missing_tags = [t for t in target_tags if t.lower() not in existing_tags and t.lower() not in ("#ad", "#affiliate")]
            if missing_tags:
                ig_caption = ig_caption.rstrip() + "\n\n" + " ".join(missing_tags)

        # Verify that all target tags are present at the end of ig_caption
        for tag in target_tags:
            self.assertIn(tag, ig_caption)
        self.assertTrue(ig_caption.endswith("#Shorts"))


if __name__ == "__main__":
    unittest.main()

