import importlib
import json
import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from Intelligence_Modules.news_trend_fetcher import (
    NewsTrendFetcher,
    scout_candidate_videos_metadata,
)
from Gemini_Modules.creative_planner import (
    GeminiCreativePlanner,
    load_creative_brief,
    save_creative_brief,
)
from Gemini_Modules.forensic_analyzer import ForensicVideoAnalyzer
from Telegram_Storage_Modules import telegram_user_manager as tum

step06 = importlib.import_module("Phase_2.06_ffmpeg_synthesis")


def test_news_fetcher_cascade_and_dedup():
    """Tests NewsData -> GNews -> Guardian -> Google News RSS cascade & deduplication."""
    newsdata_resp = MagicMock()
    newsdata_resp.raise_for_status.return_value = None
    newsdata_resp.json.return_value = {
        "results": [
            {
                "title": "Anushka Sharma spotted at Mumbai event",
                "description": "Fans cheered as Anushka arrived.",
                "source_name": "NDTV",
                "pubDate": "2026-11-18",
                "link": "https://example.com/1",
            }
        ]
    }

    gnews_resp = MagicMock()
    gnews_resp.raise_for_status.return_value = None
    gnews_resp.json.return_value = {
        "articles": [
            # Duplicate title -> should be deduplicated
            {
                "title": "Anushka Sharma spotted at Mumbai event!",
                "description": "Duplicate story.",
                "source": {"name": "Times"},
                "publishedAt": "2026-11-18",
                "url": "https://example.com/dup",
            },
            {
                "title": "Anushka Sharma announces new production slate",
                "description": "Clean Slate Filmz unveils upcoming projects.",
                "source": {"name": "Variety"},
                "publishedAt": "2026-11-18",
                "url": "https://example.com/2",
            },
        ]
    }

    rss_resp = MagicMock()
    rss_resp.raise_for_status.return_value = None
    rss_resp.text = """<?xml version="1.0" encoding="UTF-8"?>
    <rss><channel>
      <item>
        <title>Anushka Sharma viral red carpet look</title>
        <link>https://example.com/3</link>
        <description>Trending across social media.</description>
        <pubDate>Wed, 18 Nov 2026 10:00:00 GMT</pubDate>
        <source>Hindustan Times</source>
      </item>
    </channel></rss>"""

    def fake_get(url, **kwargs):
        if "newsdata.io" in url:
            return newsdata_resp
        if "gnews.io" in url:
            return gnews_resp
        if "guardianapis.com" in url:
            raise RuntimeError("Guardian quota exceeded")
        if "news.google.com" in url:
            return rss_resp
        raise AssertionError(f"Unexpected URL: {url}")

    fetcher = NewsTrendFetcher(
        newsdata_api_key="nd-key",
        gnews_api_key="gn-key",
        guardian_api_key="gd-key",
    )
    with patch("Intelligence_Modules.news_trend_fetcher.requests.get", side_effect=fake_get):
        items = fetcher.fetch("Anushka Sharma", max_results=5)

    assert len(items) == 3
    providers = [it["provider"] for it in items]
    assert providers == ["newsdata", "gnews", "google_news_rss"]


def test_zero_download_metadata_scout_respects_limit_and_ranks_keywords():
    """Scout runs yt-dlp --flat-playlist --skip-download with user limit & ranks matches."""
    fake_stdout = "\n".join([
        json.dumps({
            "id": "vid1",
            "title": "Random vlog in London",
            "description": "Coffee and rain",
            "url": "https://youtube.com/watch?v=vid1",
            "duration": 18,
            "view_count": 5000,
        }),
        json.dumps({
            "id": "vid2",
            "title": "Anushka Sharma grand entry at awards night",
            "description": "Viral clip of Anushka Sharma smiling",
            "tags": ["anushka sharma", "bollywood"],
            "url": "https://youtube.com/watch?v=vid2",
            "duration": 14,
            "view_count": 12000,
        }),
    ])
    fake_proc = SimpleNamespace(returncode=0, stdout=fake_stdout, stderr="")

    with patch("Intelligence_Modules.news_trend_fetcher.subprocess.run", return_value=fake_proc) as mock_run:
        cands = scout_candidate_videos_metadata(
            keywords=["Anushka Sharma"],
            sources=["@BollywoodClips"],
            limit=12,
        )

    cmd_called = mock_run.call_args[0][0]
    assert "--flat-playlist" in cmd_called
    assert "--skip-download" in cmd_called
    assert "--playlist-end" in cmd_called
    assert cmd_called[cmd_called.index("--playlist-end") + 1] == "12"

    assert len(cands) == 2
    assert cands[0]["id"] == "vid2"
    assert cands[0]["keyword_score"] > cands[1]["keyword_score"]
    assert "anushka sharma" in cands[0]["matched_keywords"]


def test_chat_controlled_news_keys_keywords_and_scout_limit(tmp_path):
    """Per-user News API keys, scout limit, and keywords persist cleanly via telegram_user_manager."""
    users_file = tmp_path / "telegram_users.json"
    real_save = tum.save_all_users
    with patch.object(tum, "USERS_JSON_PATH", str(users_file)), \
         patch.object(tum, "_is_admin_user", return_value=False), \
         patch.object(tum, "sync_user_secret_to_github", return_value=False), \
         patch.object(tum, "save_all_users", side_effect=lambda u, **kw: real_save(u, sync_to_vault=False)):
        assert tum.set_user_news_api_key("12345", "newsdata", "nd_secret_123")
        assert tum.set_user_news_api_key("12345", "gnews", "gn_secret_456")
        assert tum.set_user_news_api_key("12345", "guardian", "gd_secret_789")
        assert tum.set_user_metadata_scout_limit("12345", 30)
        assert tum.set_user_auto_target_keywords("12345", "Anushka Sharma, Cricket")

        keys = tum.get_user_news_api_keys("12345")
        assert keys["newsdata_api_key"] == "nd_secret_123"
        assert keys["gnews_api_key"] == "gn_secret_456"
        assert keys["guardian_api_key"] == "gd_secret_789"
        assert tum.get_user_metadata_scout_limit("12345") == 30
        assert tum.get_user_auto_target_keywords("12345") == ["Anushka Sharma", "Cricket"]

        # Clamping upper bound to 50
        tum.set_user_metadata_scout_limit("12345", 999)
        assert tum.get_user_metadata_scout_limit("12345") == 50


def test_creative_planner_and_auto_only_isolation(tmp_path):
    """
    Verifies:
    - CreativePlanner outputs validated brief & filters selected URLs to real candidates
    - Auto Input clip dir with creative_brief.json injects Shot-Hunter into Forensic Analyzer
      and promotes matched hook shot to index 0 (<=3s) in Step 06
    - Manual Input clip dir (no creative_brief.json) stays completely untouched
    """
    candidates = [
        {"url": "https://youtube.com/watch?v=good1", "title": "Anushka Sharma latest interview", "keyword_score": 5.0},
        {"url": "https://youtube.com/watch?v=other2", "title": "Other video", "keyword_score": 0.0},
    ]
    news = [{"title": "Anushka Sharma attends charity gala", "source": "NDTV", "provider": "newsdata"}]

    fake_gemini_json = json.dumps({
        "story_angle": "Anushka Sharma's surprise charity gala appearance",
        "hook_line": "Wait until you see who just stole the spotlight in Mumbai!",
        "narration_script": "Wait until you see who just stole the spotlight in Mumbai! Anushka Sharma turned heads at the charity gala tonight.",
        "first_3s_visual_directive": "Close-up of Anushka Sharma smiling at camera",
        "videos_needed": 1,
        "selected_video_urls": ["https://youtube.com/watch?v=good1", "https://hallucinated.invalid/x"],
        "mandatory_shots": [
            {"role": "hook", "description": "Close-up smile", "min_s": 1.5, "max_s": 2.5},
            {"role": "context", "description": "Full walk on red carpet", "min_s": 2.0, "max_s": 3.5},
        ],
        "enforced_feature_flags": {
            "enable_voiceover": True,
            "enable_fast_pacing": True,
            "enable_cinematic_zoom": True,
            "enable_speed_ramps": False,
        },
        "tone": "energetic_commentary",
        "bgm_mood": "upbeat_subtle",
    })

    fake_router = MagicMock()
    fake_router.generate.return_value = SimpleNamespace(text=fake_gemini_json)

    with patch("Gemini_Modules.creative_planner._HAS_ROUTER", True), \
         patch("Gemini_Modules.creative_planner.gemini_router", fake_router):
        brief = GeminiCreativePlanner(api_key="dummy").plan("Anushka Sharma", news, candidates)

    # Hallucinated URL must be stripped; only real candidate URL kept
    assert brief["selected_video_urls"] == ["https://youtube.com/watch?v=good1"]
    assert brief["enforced_feature_flags"]["enable_voiceover"] is True

    # Auto Input folder saves/loads creative_brief.json; Manual Input folder returns None
    auto_dir = tmp_path / "auto_clip"
    manual_dir = tmp_path / "manual_clip"
    auto_dir.mkdir()
    manual_dir.mkdir()
    save_creative_brief(brief, str(auto_dir))

    assert load_creative_brief(str(auto_dir)) is not None
    assert load_creative_brief(str(manual_dir)) is None

    # Forensic prompt includes Shot-Hunter ONLY when creative_brief is supplied
    analyzer = ForensicVideoAnalyzer.__new__(ForensicVideoAnalyzer)
    proxy = {"width": 480, "height": 854, "fps": 2.0, "duration_sec": 12.0}
    prompt_manual = analyzer._build_prompt(proxy, None, "creator", None, creative_brief=None)
    prompt_auto = analyzer._build_prompt(proxy, None, "creator", None, creative_brief=brief)
    assert "CREATIVE PLANNER SHOT-HUNTER TASK" not in prompt_manual
    assert "CREATIVE PLANNER SHOT-HUNTER TASK" in prompt_auto
    assert "Close-up smile" in prompt_auto

    # Step 06 promotes matched hook shot first (clamped <=3s) ONLY on Auto Input
    orig_shots = [{"start": 0.0, "end": 2.0, "_section": "verse", "score": 0.7}]
    manual_out = step06._apply_creative_brief_to_shots(orig_shots, {"matched_creative_shots": [{"role": "hook", "start": 5.0, "end": 9.0}]})
    assert manual_out == orig_shots  # No creative_brief -> completely untouched!

    auto_ctx = {
        "creative_brief": brief,
        "matched_creative_shots": [
            {"role": "context", "start": 1.0, "end": 3.5, "score": 0.85},
            {"role": "hook", "start": 6.0, "end": 10.5, "score": 0.98, "reason": "Best close-up smile"},
        ],
    }
    auto_out = step06._apply_creative_brief_to_shots(orig_shots, auto_ctx)
    assert auto_out[0]["_section"] == "hook"
    assert auto_out[0]["start"] == 6.0
    assert (auto_out[0]["end"] - auto_out[0]["start"]) <= 3.0
