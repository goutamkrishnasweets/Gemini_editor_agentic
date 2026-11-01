# Postiz Publisher Engine (`postiz_publisher.py`)

`postiz_publisher.py` is the **Unified Multi-Channel Publishing Engine** for AMTCE.

It connects the Python video rendering pipeline directly to a self-hosted or cloud [Postiz](https://github.com/gitroomhq/postiz-app) server via the Postiz Public REST API.

---

## Architecture Overview

```
AMTCE Video Rendered & SEO Generated
               │
               ▼
   postiz_publisher.publish_to_postiz(...)
               │
               ├──> 1. Upload Video (POST /public/v1/upload)
               ├──> 2. Query Active Channels (GET /public/v1/integrations)
               └──> 3. Create Multi-Channel Post (POST /public/v1/posts)
                            │
               ┌────────────┼────────────┐
               ▼            ▼            ▼
         YouTube Shorts  Instagram    TikTok
```

---

## Configuration

Set these environment variables in `.env` or GitHub Secrets:

| Variable | Description | Example |
| :--- | :--- | :--- |
| `POSTIZ_URL` | Base URL of your Postiz server | `https://postiz.yourdomain.com` or `http://localhost:5000` |
| `POSTIZ_API_KEY` | Postiz API Key (from Postiz Settings ➔ API Keys) | `postiz_live_abc123...` |
| `POSTIZ_CHANNELS` | Target channels filter | `all` (default) or `youtube,instagram,tiktok` |

---

## Core Functions

### 1. `is_postiz_configured() -> bool`
Checks if `POSTIZ_URL` and `POSTIZ_API_KEY` are configured.

### 2. `get_active_integrations() -> list`
Fetches all active social accounts connected to Postiz via `GET /public/v1/integrations`.

### 3. `upload_video_to_postiz(video_path: str) -> Optional[str]`
Uploads the local MP4 video reel to Postiz storage via `POST /public/v1/upload`. Returns the uploaded media ID.

### 4. `publish_to_postiz(video_path, title, description, tags, ...) -> dict`
Master function that uploads the reel, builds the caption and hashtag payload, and posts across all selected channels.

### 5. `PublishQueue`
Modern drop-in adapter that replaces the legacy in-memory queue with direct Postiz integration.
