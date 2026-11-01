import os
import unittest
from unittest.mock import patch, MagicMock
from Publishing_Modules.make_publisher import (
    build_make_payload,
    resolve_public_media_url,
    is_make_configured,
)


class TestMakePublisher(unittest.TestCase):
    def test_payload_builder(self):
        payload = build_make_payload(
            "https://example.com/video.mp4",
            title="Daphne cosplay outfit",
            caption="Check out this outfit! #Shorts",
            tags="cosplay, daphne",
            post_type="reel"
        )
        self.assertEqual(payload["video_url"], "https://example.com/video.mp4")
        self.assertEqual(payload["title"], "Daphne cosplay outfit")
        self.assertEqual(payload["post_type"], "reel")
        self.assertIn("first_comment", payload)

    def test_resolve_public_media_url_with_existing_url(self):
        url = "https://api.telegram.org/file/bot123/video.mp4"
        self.assertEqual(resolve_public_media_url(url), url)

    def test_resolve_public_media_url_local_file(self):
        dummy_file = "test_dummy_video.mp4"
        with open(dummy_file, "wb") as f:
            f.write(b"RIFF" + b"\x00" * 1024)

        try:
            with patch("Telegram_Storage_Modules.telegram_http.send_document") as mock_send, \
                 patch("urllib.request.urlopen") as mock_url:

                mock_send.return_value = {"message_id": 99, "document": {"file_id": "test_fid_abc"}}

                mock_resp = MagicMock()
                mock_resp.read.return_value = b'{"ok": true, "result": {"file_path": "videos/reel.mp4"}}'
                mock_url.return_value.__enter__.return_value = mock_resp

                with patch.dict(os.environ, {
                    "TELEGRAM_BOT_TOKEN": "123456:TEST-TOKEN",
                    "TELEGRAM_STORAGE_GROUP_ID": "-1001999999"
                }):
                    resolved = resolve_public_media_url(dummy_file)
                    self.assertEqual(resolved, "https://api.telegram.org/file/bot123456:TEST-TOKEN/videos/reel.mp4")

                    # Verify local_path was sent as local_path, NOT as chat_id
                    mock_send.assert_called_once()
                    called_local_path = mock_send.call_args.kwargs.get("local_path") or mock_send.call_args.args[0]
                    self.assertEqual(called_local_path, dummy_file)
        finally:
            if os.path.exists(dummy_file):
                os.remove(dummy_file)


if __name__ == "__main__":
    unittest.main()
