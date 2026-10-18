# 🎬 Gemini Editor — Autonomous Multi-Platform AI Video Factory & RawGrab Pipeline

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-Scheduled_Runs-green.svg)](#-github-actions-automation)
[![Gemini Vision AI](https://img.shields.io/badge/Gemini_AI-Vision_%26_SEO-orange.svg)](https://deepmind.google/technologies/gemini/)
[![Telegram Bot](https://img.shields.io/badge/Telegram_Bot-Cloud_Vault_Sync-blue.svg)](#-telegram-bot--cloud-vault-hydration)

> **Gemini Editor** is an enterprise-grade, autonomous short-form video creation, editing, auditing, and multi-platform broadcasting system. It automatically ingests content from targeted creators, removes watermarks, overlays custom brand signatures, audits clip engagement with Gemini Vision AI, generates feed-injecting SEO titles/descriptions, and posts seamlessly across social networks with organic anti-bot humanization.

---

## 🌟 Key Features

### 🤖 Telegram Bot & Cloud Vault Hydration
* **Interactive Control**: Manage target accounts, trigger manual ingestion runs, select platforms, and inspect session outputs via Telegram commands (`/start`, `/addaccount`).
* **Zero Data Loss Vault**: Automatically back up and hydrate `master_vault_index.json`, user sessions, and `source_accounts.json` from Telegram Storage Group.

### ⏰ Anti-Duplicate Scheduled Harvester
* **Automated Clock Schedules**: Runs in two fixed 6-hour shifts (**04:20 AM to 10:20 AM IST** and **04:20 PM to 10:20 PM IST**).
* **Anti-Duplicate Rotation**: Selects up to 2 accounts per session prioritized by `account_last_scraped` timestamp so morning and evening runs never duplicate scraping on the same accounts.
* **30-Day Rolling Expiration**: Automatically purges source accounts older than 30 days.

### 🔍 Gemini Vision Audit & Subject Discovery
* **Watermark Overlap Audit**: Verifies that custom brand overlays cover inpainted watermark regions with $\ge 85\%$ mathematical IoU coverage.
* **Dopamine Hook & Retention Score**: Rates the first 3 seconds hook retention and pacing.
* **Real Subject Discovery**: Discovers the true focal subject/star in the video from visual frames and post metadata without using aggregator handles.
* **Strict Handle ID Suppression**: Prevents raw account handle IDs (e.g. `@username`, `id123`) from appearing in titles, descriptions, or hashtags.

### ⏳ Humanized Organic Publishing Stagger
* **Anti-Automation Delay**: Introduces a randomized **3 to 6 minute organic delay** (`PUBLISH_STAGGER_MIN_SECONDS` to `PUBLISH_STAGGER_MAX_SECONDS`) between consecutive clip uploads to protect accounts from bot detection rate-limits.

### 📤 Multi-Platform Simultaneous Broadcasting
* 🔴 **YouTube Shorts** (YouTube Data API v3)
* 📸 **Instagram Reels** (Meta Graph API)
* 📘 **Facebook Reels** (Meta Graph API)
* ✈️ **Telegram Channels & Storage Groups** (Primary Admin Chat, Public Channel, and Vault Storage Group)

---

## 📁 Repository Structure

```text
Gemini_editor/
├── .github/workflows/
│   └── gemini-editor-runner.yml       # GitHub Actions 6-hour scheduled daemon workflow
├── Audio_Modules/               # Audio extraction, pool management, & rhythm alignment
├── Content_Scraper_Modules/
│   └── source_accounts.json     # Clean platform target accounts index & last_scraped timestamps
├── Core_Modules/                # Session manager, credential manager, & approval flows
├── Downloader_Modules/
│   ├── apify_downloader.py      # Apify scraper engine & quota management
│   └── scheduled_scraper_manager.py # Max 2-account anti-duplicate rotation manager
├── Gemini_Modules/
│   ├── gemini_clip_auditor.py   # Proxy encoder reuse, watermark IoU audit, & retention scoring
│   ├── platform_seo_generator.py # AI SEO title/description generator with handle suppression
│   └── gemini_router_module/    # Model routing & Gemini Governor
├── Main_Modules/                # Master AI editor, proxy encoder, & frame samplers
├── Phase_1/                     # Ingestion & deduplication pipeline
├── Phase_2/                     # Scene perception, BGM selection, & FFmpeg synthesis
├── Phase_3/                     # Monetization gate & metadata generation
├── Publishing_Modules/
│   ├── media_publisher_main.py  # 4-platform broadcasting orchestrator
│   ├── meta_uploader.py         # Multi-niche Instagram & Facebook Reels publisher
│   └── telegram_vault_indexer.py# Cloud vault backup & hydration engine
├── main.py                      # Master orchestrator & Telegram Bot entry point
├── requirements.txt             # Production dependency manifest
└── README.md
```

---

## ⚙️ Configuration Setup

All credentials and parameters live in `Credentials/.env` (strictly ignored by `.gitignore`):

```env
# ── TELEGRAM BOT & VAULT STORAGE ───────────────────────────────────────────
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_ADMIN_ID=your_admin_id
TELEGRAM_STORAGE_GROUP_ID=your_storage_group_id
TELEGRAM_PUBLIC_GROUP_ID=your_public_group_id

# ── GEMINI AI ENGINE ──────────────────────────────────────────────────────
GEMINI_API_KEY=your_gemini_api_key

# ── APIFY SCRAPE ENGINE ───────────────────────────────────────────────────
APIFY_API_TOKEN=your_apify_api_token
AUTO_INPUT_SCHEDULE_TIMES=06:00,19:00

# ── ORGANIC PUBLISHING STAGGER (SECONDS) ──────────────────────────────────
PUBLISH_STAGGER_MIN_SECONDS=180
PUBLISH_STAGGER_MAX_SECONDS=360

# ── META (INSTAGRAM & FACEBOOK REELS) ─────────────────────────────────────
META_PAGE_ID=your_facebook_page_id
META_PAGE_TOKEN=your_facebook_page_token
IG_BUSINESS_ID=your_instagram_business_id
IG_BUSINESS_TOKEN=your_instagram_business_token
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run Telegram Bot & Background Scheduler
```bash
python main.py
```

### 3. Add Source Accounts via Telegram
In Telegram, message your bot:
```text
/addaccount @username
```

---

## ⚡ GitHub Actions Automation

The workflow in `.github/workflows/gemini-editor-runner.yml` automatically triggers on GitHub Actions runners every 6 hours (**04:20 AM, 10:20 AM, 04:20 PM, 10:20 PM IST**):

```yaml
  schedule:
    # 2 clean 6-hour windows (Indian Standard Time — IST):
    - cron: '50 22 * * *'  # Morning window: 04:20 AM to 10:20 AM IST
    - cron: '50 10 * * *'  # Evening window: 04:20 PM to 10:20 PM IST
```

---

## 🛡️ Security Note

This repository contains **zero hardcoded tokens, secrets, or API keys**. All credentials, `.env` files, audio caches, and rendered videos are protected by `.gitignore`.

---

## 📝 License

Distributed under the MIT License. See `LICENSE` for details.
