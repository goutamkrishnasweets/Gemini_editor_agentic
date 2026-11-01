import unittest
from Gemini_Modules.lyric_rhythm_aligner import (
    compare_semantic_entities,
    _build_shazam_context,
)


class TestSongIntelligenceAndEntitySynergy(unittest.TestCase):
    def setUp(self):
        self.song_ents = {
            "title": "Bom Diggy Diggy",
            "movie_or_album": "Sonu Ke Titu Ki Sweety",
            "cast_members": ["Kartik Aaryan", "Sunny Singh", "Sakshi Malik"],
            "singers": ["Zack Knight", "Jasmin Walia"],
            "search_keywords": ["Bom Diggy Diggy", "Sakshi Malik", "Kartik Aaryan", "Sonu Ke Titu Ki Sweety"]
        }

    def test_cast_member_hashtag_match(self):
        clip_text = "Check out this dance #SakshiMalik #trending"
        bonus, matched = compare_semantic_entities(clip_text, self.song_ents)
        self.assertIn("Cast: Sakshi Malik", matched)
        self.assertGreaterEqual(bonus, 0.40)

    def test_movie_name_match(self):
        clip_text = "Iconic club scene from Sonu Ke Titu Ki Sweety"
        bonus, matched = compare_semantic_entities(clip_text, self.song_ents)
        self.assertTrue(any("Sonu Ke Titu Ki Sweety" in m for m in matched))
        self.assertGreaterEqual(bonus, 0.30)

    def test_singer_match(self):
        clip_text = "Vocals by Jasmin Walia live concert"
        bonus, matched = compare_semantic_entities(clip_text, self.song_ents)
        self.assertTrue(any("Jasmin Walia" in m for m in matched))
        self.assertGreaterEqual(bonus, 0.20)

    def test_shazam_context_formatting(self):
        ctx = _build_shazam_context(
            {"title": "Bom Diggy Diggy", "artist": "Zack Knight", "genre": "Bollywood"},
            song_entities=self.song_ents
        )
        self.assertIn("Bom Diggy Diggy", ctx)
        self.assertIn("Sonu Ke Titu Ki Sweety", ctx)
        self.assertIn("Sakshi Malik", ctx)


if __name__ == "__main__":
    unittest.main()

