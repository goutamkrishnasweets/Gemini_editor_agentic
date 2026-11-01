# Make.com Multi-Platform Webhook Publisher

## Overview
The `make_publisher.py` module integrates Gemini Editor with **Make.com** via a single, zero-maintenance Webhook.

It eliminates all complex Meta Developer App registrations, API keys, and reverse-engineered scrapers by leveraging Make.com's pre-verified enterprise apps for 1-click publishing across social networks.

---

## Supported Media & Post Types

| Post Type (`post_type`) | Media Type (`media_type`) | Supported Formats | Target Platforms |
| :--- | :--- | :--- | :--- |
| `reel` | `video` | `.mp4`, `.mov` (Vertical 9:16) | Instagram Reels, YouTube Shorts, Facebook Reels |
| `post` | `image` or `video` | `.jpg`, `.png`, `.mp4` (1:1, 4:5, 16:9) | Instagram Feed, Facebook Page, LinkedIn |
| `carousel` | `carousel` | Multi-image / Multi-video | Instagram Carousels |
| `story` | `image` or `video` | 9:16 vertical | Instagram Stories, Facebook Stories |

---

## Affiliate Link Features
- **Automatic Compliance Disclosure**: Appends `(#ad #affiliate)` if affiliate links are present and no disclosure exists.
- **YouTube Description Placement**: Automatically formats clickable affiliate links into the `youtube.description` payload field.
- **Instagram First Comment**: Generates a dedicated `first_comment` payload with clickable affiliate item links (e.g., `🛒 Featured item: https://...`).
- **Story Link Attachment**: Generates `story_link` / `primary_link` for story link stickers or destination URLs.

---

## Payload Structure Sent to Make.com

```json
{
  "event": "gemini_editor_publish",
  "timestamp": 1793551200.0,
  "user_id": "admin",
  "post_type": "reel",
  "media_type": "video",
  "video_url": "https://api.telegram.org/file/bot.../reel.mp4",
  "image_url": "",
  "media_urls": ["https://api.telegram.org/file/bot.../reel.mp4"],
  "title": "Viral Reel Title",
  "caption": "Check out this amazing edit! #viral #trending\n\n(#ad #affiliate)",
  "hashtags": "#viral #trending",
  "story_link": "https://amazon.com/item/123",
  "primary_link": "https://amazon.com/item/123",
  "affiliate_links": ["https://amazon.com/item/123"],
  "first_comment": "🛒 Featured item link: https://amazon.com/item/123 (#ad)",
  "youtube": {
    "title": "Viral Reel Title",
    "description": "Full description with links...",
    "tags": ["viral", "trending"],
    "privacy_status": "public"
  },
  "instagram": {
    "caption": "Instagram formatted caption...",
    "media_type": "REELS",
    "first_comment": "🛒 Featured item link..."
  },
  "facebook": {
    "caption": "Facebook formatted caption...",
    "media_type": "REELS"
  }
}
```

---

## Configuration

Set this variable in your `.env` or GitHub Repository Secrets:
```env
MAKE_WEBHOOK_URL="https://hook.eu2.make.com/your-private-webhook-token"
```

