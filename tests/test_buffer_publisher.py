import os
import unittest
from unittest.mock import patch, MagicMock

from Publishing_Modules.buffer_publisher import (
    is_buffer_configured,
    _format_x_caption,
    _format_tiktok_caption,
    _format_pinterest_caption,
    _resolve_profile_ids,
    publish_to_buffer,
    set_user_buffer_token,
    get_user_buffer_token,
)


class TestBufferPublisher(unittest.TestCase):
    def setUp(self):
        self._orig_token = os.environ.get("BUFFER_ACCESS_TOKEN")

    def tearDown(self):
        if self._orig_token is not None:
            os.environ["BUFFER_ACCESS_TOKEN"] = self._orig_token
        else:
            os.environ.pop("BUFFER_ACCESS_TOKEN", None)

    def test_is_buffer_configured(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(is_buffer_configured())

        with patch.dict(os.environ, {"BUFFER_ACCESS_TOKEN": "test_buf_tok_123"}):
            self.assertTrue(is_buffer_configured())

    def test_x_caption_length_enforcement(self):
        long_title = "A" * 300
        caption = _format_x_caption(
            title=long_title,
            caption="Some description",
            tags="#fashion #style",
            affiliate_link="https://amzn.to/example"
        )
        self.assertLessEqual(len(caption), 280)
        self.assertTrue(caption.endswith("https://amzn.to/example"))

    def test_tiktok_caption_formatting(self):
        title = "Cool Drone Shots"
        tags = "#drone #fpv #aerial"
        caption = _format_tiktok_caption(
            title=title,
            caption="Check out these epic drone shots!",
            tags=tags,
            affiliate_link="https://amzn.to/drone"
        )
        self.assertLessEqual(len(caption), 2200)
        self.assertIn("#drone", caption)
        self.assertIn("https://amzn.to/drone", caption)

    def test_pinterest_caption_formatting(self):
        title = "P" * 150
        desc = "D" * 600
        res = _format_pinterest_caption(title=title, caption=desc, tags="#pin")
        self.assertLessEqual(len(res["title"]), 100)
        self.assertLessEqual(len(res["description"]), 500)

    def test_resolve_profile_ids_env_override(self):
        with patch.dict(os.environ, {
            "BUFFER_TIKTOK_PROFILE_ID": "tt_111",
            "BUFFER_PINTEREST_PROFILE_ID": "pin_222",
            "BUFFER_X_PROFILE_ID": "x_333",
        }):
            resolved = _resolve_profile_ids("test_token")
            self.assertEqual(resolved["tiktok"], ["tt_111"])
            self.assertEqual(resolved["pinterest"], ["pin_222"])
            self.assertEqual(resolved["x"], ["x_333"])

    @patch("Publishing_Modules.buffer_publisher._resolve_media_url", return_value="https://pub.url/reel.mp4")
    @patch("Publishing_Modules.buffer_publisher._resolve_profile_ids", return_value={
        "tiktok": ["tt_1"], "pinterest": ["pin_1"], "x": ["x_1"]
    })
    @patch("Publishing_Modules.buffer_publisher._post_to_buffer")
    def test_publish_to_buffer_success(self, mock_post, mock_profiles, mock_url):
        mock_post.return_value = {
            "status": "success",
            "update_id": "up_abc123",
            "updates": [{"id": "up_abc123"}]
        }

        with patch.dict(os.environ, {"BUFFER_ACCESS_TOKEN": "valid_token"}):
            res = publish_to_buffer(
                video_path="dummy.mp4",
                title="Test Reel",
                description="Test description",
                tags="#test #viral",
            )
            self.assertEqual(res["status"], "success")
            self.assertIn("tiktok", res["published_to"])
            self.assertIn("pinterest", res["published_to"])
            self.assertIn("x", res["published_to"])
            self.assertEqual(mock_post.call_count, 3)

    def test_publish_to_buffer_skipped_when_no_token(self):
        with patch.dict(os.environ, {}, clear=True):
            res = publish_to_buffer(
                video_path="dummy.mp4",
                title="Test Reel",
            )
            self.assertEqual(res["status"], "skipped")


if __name__ == "__main__":
    unittest.main()
