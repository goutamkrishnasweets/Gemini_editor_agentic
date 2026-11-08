"""
tests/test_seo_p0_compliance_fixes.py
=====================================
Regression & Verification Suite for SEO P0 Remediation:
1. Disclosure Regex & Idempotency: finalize(finalize(x)) == finalize(x).
2. Upfront Disclosure Placement (line2): ASCI compliance before the fold.
3. Policy-Aware Mention Exemption: Preserves @meeshoapp without hardcoding merchants.
4. Year vs Price Loophole: 2025 is allowed as year, but ₹1,999 requires evidence.
5. Router Failure Latch: Only latches _ROUTER_TRIED upon success.
6. Instagram Hashtag Cap: Defaults to 5.
"""

import re
import unittest
from Gemini_Modules.platform_seo_generator import (
    _place_disclosure,
    _disclosure_ok,
    _DISC_ANY,
    ProvenanceGuard,
    lint_platform,
    PLATFORM_SPECS,
    _get_router,
    get_provider_disclosure,
)


class TestSEOP0ComplianceFixes(unittest.TestCase):

    def test_01_disclosure_regex_and_idempotency(self):
        """Verify _place_disclosure strips full disclosure parentheticals cleanly and is idempotent."""
        desc = (
            "Effortlessly chic, these wide-leg trackpants redefine casual comfort.\n\n"
            "Style them with your favorite sneakers for an everyday aesthetic."
        )
        links = ["https://link.amazon/B0eYFOBtQ"]

        # First pass
        placed_once = _place_disclosure("instagram", desc, has_links=True, links=links)
        self.assertIn("#ad", placed_once)
        self.assertIn("As an Amazon Associate", placed_once)

        # Second pass (simulating finalize(finalize(x)) or critic rewrite)
        placed_twice = _place_disclosure("instagram", placed_once, has_links=True, links=links)

        # Must be byte-for-byte identical (Idempotent!)
        self.assertEqual(placed_once, placed_twice, "Disclosure placement is NOT idempotent!")

        # Verify no dangling broken parentheticals like '(Paid link ... )'
        dangling = re.findall(r"\([^)]*?(?:Paid|qualifying)[^)]*?\)", placed_twice, re.I)
        for p in dangling:
            self.assertIn("#ad", p, f"Dangling parenthetical without #ad found: {p}")

        # Count occurrences: exactly one disclosure
        self.assertEqual(placed_twice.count("As an Amazon Associate"), 1)
        self.assertEqual(placed_twice.count("#ad"), 1)

    def test_02_disclosure_placement_line2_and_validation(self):
        """Verify disclosure placement defaults to line2 and _disclosure_ok verifies it sits in lines 1-2."""
        self.assertEqual(PLATFORM_SPECS["instagram"]["disclosure_placement"], "line2")
        self.assertEqual(PLATFORM_SPECS["youtube"]["disclosure_placement"], "line2")

        desc = (
            "Effortlessly chic, these wide-leg trackpants redefine casual comfort.\n"
            "(Paid link · As an Amazon Associate I earn from qualifying purchases #ad)\n\n"
            "Style them with sneakers."
        )
        # Sits on line 2 -> Valid
        self.assertTrue(_disclosure_ok("instagram", desc))

        # Buried on line 5 -> Must FAIL line2 verification
        buried_desc = (
            "Line 1 hook\n"
            "Line 2 feature\n"
            "Line 3 benefits\n"
            "Line 4 cta\n"
            "(Paid link · As an Amazon Associate I earn from qualifying purchases #ad)"
        )
        self.assertFalse(_disclosure_ok("instagram", buried_desc))

    def test_03_policy_aware_mention_exemption(self):
        """Verify ProvenanceGuard preserves @meeshoapp when policy-allowed, while stripping source leaks."""
        # 1. Guard with @meeshoapp specifically allowed by policy
        guard = ProvenanceGuard(
            identifiers=["source_creator_123"],
            allow_text=["Kurti Haul"],
            allowed_mentions=["@meeshoapp"]
        )

        caption = "Honest review of this floral kurti from @meeshoapp! Credits to @source_creator_123"
        scrubbed = guard.scrub_text(caption)

        # @meeshoapp must be preserved!
        self.assertIn("@meeshoapp", scrubbed)
        # Source channel @source_creator_123 must be stripped!
        self.assertNotIn("@source_creator_123", scrubbed)

        # Leak check: @meeshoapp must NOT be flagged as a leak
        leaks = guard.find_leaks(scrubbed)
        self.assertNotIn("@meeshoapp", leaks)

    def test_04_no_arbitrary_merchant_whitelist(self):
        """Verify arbitrary merchants are NOT globally whitelisted unless policy-mandated."""
        # Without explicit policy mention, random handles (even mentioning brands) remain protected
        guard = ProvenanceGuard(identifiers=["competitor_channel"])
        text = "Check out deals on @random_store and @competitor_channel"
        scrubbed = guard.scrub_text(text)
        self.assertNotIn("@random_store", scrubbed)
        self.assertNotIn("@competitor_channel", scrubbed)

    def test_05_year_vs_price_grounding(self):
        """Verify calendar year 2025 is allowed, but ₹1,999 price is flagged if not in evidence."""
        content = {
            "title": "Summer Style Trends 2025 #Shorts",
            "description": "Top fashion trends for 2025.\n(Paid affiliate link · #ad)\nBuy now for ₹1999."
        }
        # Evidence only has year 2025, not 1999
        ctx = {"evidence_text": "Summer fashion collection 2025 trends."}
        _, issues = lint_platform("youtube", content, links=["https://example.com"], primary_keyword="summer", ctx=ctx)

        issue_codes = [i["code"] for i in issues]
        # 1999 must be flagged as ungrounded number!
        self.assertIn("ungrounded_number", issue_codes)

        # But 2025 should NOT be flagged as an ungrounded number
        un_issues = [i for i in issues if i["code"] == "ungrounded_number"]
        self.assertTrue(any("1999" in i["message"] for i in un_issues))
        self.assertFalse(any("2025" in i["message"] for i in un_issues))

    def test_06_instagram_hashtag_limit(self):
        """Verify Instagram hashtags max is 5 in PLATFORM_SPECS."""
        self.assertEqual(PLATFORM_SPECS["instagram"]["hashtags_max"], 5)
        self.assertEqual(PLATFORM_SPECS["instagram"]["hashtags_total_max"], 5)

    def test_07_meesho_policy_mention_validation(self):
        """Verify Meesho Creator Club mandatory @meeshoapp mention is strictly enforced in lint_platform."""
        meesho_policy = {
            "meesho_creator_policy": {
                "mandatory_caption_and_tag_rules": {
                    "required_account_mention": "@meeshoapp"
                }
            }
        }
        # Caption missing @meeshoapp -> hard failure
        content_bad = {
            "title": "Meesho Saree Haul",
            "description": "Gorgeous floral saree review.\n(Paid link · #ad)\nBuy now!"
        }
        ctx_bad = {"external_policy": meesho_policy}
        _, issues_bad = lint_platform("instagram", content_bad, links=["https://meesho.com/p/123"], ctx=ctx_bad)
        codes_bad = [i["code"] for i in issues_bad]
        self.assertIn("missing_policy_mention", codes_bad)

        # Caption with @meeshoapp -> passes mention check
        content_good = {
            "title": "Meesho Saree Haul",
            "description": "Gorgeous floral saree from @meeshoapp.\n(Paid link · #ad)\nBuy now!"
        }
        ctx_good = {"external_policy": meesho_policy}
        _, issues_good = lint_platform("instagram", content_good, links=["https://meesho.com/p/123"], ctx=ctx_good)
        codes_good = [i["code"] for i in issues_good]
        self.assertNotIn("missing_policy_mention", codes_good)

    def test_08_meta_feature_command_token_stripping(self):
        """Verify meta feature tokens (/trail_reels, /collab, /tag, /no_grid) are cleanly stripped."""
        from Gemini_Modules.platform_seo_generator import _strip_meta_tokens_local

        raw_title = "Trendy Kurtis /trail_reels /collab @partner1 @partner2 /no_grid"
        cleaned = _strip_meta_tokens_local(raw_title)
        self.assertEqual(cleaned, "Trendy Kurtis")

        raw_hint = "/tag @shop /audio_name 'Original Sound' Check out these styles"
        cleaned_hint = _strip_meta_tokens_local(raw_hint)
        self.assertEqual(cleaned_hint, "Check out these styles")

    def test_09_ownership_claim_on_reused_content(self):
        """Verify that first-person ownership claims on reused footage are strictly blocked."""
        content = {
            "title": "My Viral Review Video",
            "description": "This is my exclusive video and I tested it myself.\n(Paid link · #ad)"
        }
        ctx = {"content_origin": "reused", "evidence_text": "Product review footage."}
        _, issues = lint_platform("youtube", content, links=["https://example.com"], ctx=ctx)
        codes = [i["code"] for i in issues]
        self.assertIn("ownership_claim", codes)


if __name__ == "__main__":
    unittest.main()
