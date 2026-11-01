# Phase 4 Media Publisher (`media_publisher_main.py`)

`media_publisher_main.py` is the **Phase 4 Multi-Platform Broadcasting Orchestrator**.

It is invoked automatically when a user clicks **[Approve]** or replies with a custom title in Telegram.

---

## Workflow

1. Receives rendered master reel and SEO data from `platform_seo_generator.py`.
2. Dispatches broadcasting directly to **Postiz** (`postiz_publisher.publish_to_postiz`).
   - Postiz handles YouTube Shorts, Instagram Reels, TikTok, and Facebook.
3. Broadcasts video to configured Telegram Public Channel (`publish_to_telegram`).
4. Sends real-time status and live post links back to the user's Telegram chat window.
