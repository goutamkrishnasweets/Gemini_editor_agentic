"""
downloader_main.py — Standalone Phase 1 Ingestion & Downloader Orchestrator
=============================================================================
Orchestrates Phase 1 Video Ingestion & Downloader pipeline.

Supports 2 Standalone Ingestion Workers:
  1. Worker 1 (Automated Harvester):
     Scrapes target accounts listed in `Content_Scraper_Modules/source_accounts.json`
     via `apify_downloader.py` & `harvester.py`.
  2. Worker 2 (Manual Input Downloader):
     Ingests raw video clip directly from a specific URL or file path via `downloader.py`.

Usage:
    from Downloader_Modules.downloader_main import run_phase1_ingestion

    # Run Worker 1 (Automated Scraping)
    result = run_phase1_ingestion(mode="auto")

    # Run Worker 2 (Manual URL Download)
    result = run_phase1_ingestion(mode="manual", url="https://instagram.com/p/...")

CLI:
    python Downloader_Modules/downloader_main.py --mode auto
    python Downloader_Modules/downloader_main.py --url https://instagram.com/p/...
"""

import os
import sys
import json
import logging
import argparse
import time
import threading
import re
from typing import Dict, List, Optional, Any

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("downloader_main")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    from dotenv import load_dotenv
    for env_path in [
        os.path.join(_REPO_ROOT, "Credentials", ".env"),
        os.path.join(_REPO_ROOT, ".env"),
    ]:
        if os.path.exists(env_path):
            load_dotenv(env_path, override=True)
except ImportError:
    pass


def _extract_shortcode(url: str) -> Optional[str]:
    """Extracts shortcode/ID from Instagram Reel, YouTube Short, TikTok, or generic URL."""
    if not url:
        return None
    m_ig = re.search(r"/(?:p|reel|reels)/([A-Za-z0-9_-]+)", url)
    if m_ig:
        return m_ig.group(1)
    m_yt = re.search(r"(?:shorts/|v=|youtu\.be/)([A-Za-z0-9_-]{11})", url)
    if m_yt:
        return m_yt.group(1)
    m_tt = re.search(r"/video/(\d+)", url)
    if m_tt:
        return m_tt.group(1)
    return None


def send_raw_download_to_telegram_vault_async(
    out_file: str,
    url: str,
    shortcode: str,
    platform: str,
    clip_dir: str
) -> None:
    """
    Asynchronously uploads raw downloaded clip to Telegram Storage Vault without blocking main thread.
    Updates local metadata.json with file_id and registers entry in TelegramVaultIndexer.
    """
    def _upload_task():
        try:
            logger.info(f"📤 [VAULT ASYNC] Spawning background vault upload for: {out_file}")
            from Telegram_Storage_Modules.telegram_http import send_document
            from Telegram_Storage_Modules.telegram_vault_indexer import TelegramVaultIndexer

            is_auto = "auto_" in clip_dir.lower() or "auto_" in os.path.basename(clip_dir).lower()
            prefix = "auto_raw_" if is_auto else "manual_raw_"
            raw_display_name = f"{prefix}{shortcode}.mp4" if shortcode else f"{prefix}{os.path.basename(out_file)}"
            caption = f"📥 **[VAULT RAW SOURCE]** `{raw_display_name}`\n🔗 `{url}`\n🆔 `{os.path.basename(clip_dir)}`"
            resp = send_document(out_file, caption=caption, custom_filename=raw_display_name)

            if resp and isinstance(resp, dict):
                result = resp.get("result", resp)
                file_id = None
                if isinstance(result, dict):
                    if "video" in result and isinstance(result["video"], dict):
                        file_id = result["video"].get("file_id")
                    elif "document" in result and isinstance(result["document"], dict):
                        file_id = result["document"].get("file_id")

                if file_id:
                    logger.info(f"   ✓ [VAULT ASYNC SUCCESS] Raw clip uploaded to Telegram Storage Vault (file_id: {file_id})")
                    meta_path = os.path.join(clip_dir, "metadata.json")
                    if os.path.exists(meta_path):
                        try:
                            with open(meta_path, "r", encoding="utf-8") as mf:
                                meta = json.load(mf)
                            meta["raw_vault_file_id"] = file_id
                            with open(meta_path, "w", encoding="utf-8") as mf:
                                json.dump(meta, mf, indent=2, ensure_ascii=False)
                        except Exception as me:
                            logger.warning(f"   ⚠ Could not update metadata.json with file_id: {me}")

                    try:
                        indexer = TelegramVaultIndexer()
                        indexer.record_ingested_clip_source(
                            social_url=url,
                            raw_video_path=out_file,
                            upload_fn=None,
                            existing_raw_file_id=file_id,
                        )
                    except Exception as ie:
                        logger.warning(f"   ⚠ Vault indexer record warning: {ie}")
            else:
                logger.warning(f"   ⚠ [VAULT ASYNC WARNING] Vault upload response not ok: {resp}")
        except Exception as ve:
            logger.warning(f"   ⚠ [VAULT ASYNC FAILED] Non-fatal vault upload error: {ve}")

    thread = threading.Thread(target=_upload_task, name=f"VaultUploader-{shortcode}", daemon=True)
    thread.start()
    logger.info(f"⚡ [VAULT ASYNC] Background uploader thread started for {shortcode} (non-blocking).")


def run_phase1_ingestion(
    mode: str = "auto",
    url: Optional[str] = None,
    limit_per_account: Optional[int] = None,
    target_accounts: Optional[List[str]] = None,
    platform: str = "instagram"
) -> Dict[str, Any]:
    """
    Execute Phase 1 Ingestion Pipeline across multiple platforms.

    Args:
        mode:              "auto" (Worker 1: Automated Harvester) or "manual" (Worker 2: Manual URL)
        url:               Target video URL (required if mode="manual")
        limit_per_account: Max reels to download per account
        target_accounts:   Explicit list of creator handles / channel IDs to scrape
        platform:          "instagram", "youtube", "tiktok", or "direct"

    Returns:
        Dict containing downloaded video paths, count, and execution status.
    """
    if limit_per_account is None:
        try:
            limit_per_account = int(os.getenv("AUTO_SCRAPE_LIMIT", os.getenv("APIFY_REELS_PER_ACCOUNT", "8")))
        except (ValueError, TypeError):
            limit_per_account = 8

    logger.info(f"📥 [PHASE 1 INGESTION] Starting Phase 1 Orchestrator (mode='{mode}', platform='{platform}')...")
    downloads_dir = os.path.join(_REPO_ROOT, "downloads")
    os.makedirs(downloads_dir, exist_ok=True)

    downloaded_files = []

    # ── WORKER 2: Manual Input Downloader (Direct URL / Any Platform) ─────────
    if mode == "manual" or url:
        if not url:
            raise ValueError("Manual mode requires a valid video URL (--url).")

        logger.info(f"👤 [WORKER 2 - MANUAL] Ingesting single video from ({platform}): {url}")
        try:
            from Downloader_Modules.downloader import download_video
            shortcode = _extract_shortcode(url) or f"manual_{int(time.time())}"
            clip_folder_name = f"manual_{shortcode}"
            clip_dir = os.path.join(downloads_dir, clip_folder_name)
            os.makedirs(clip_dir, exist_ok=True)

            # ── Pre-check Tier 1 (Local Disk) & Tier 2 (Telegram Vault) ──
            local_video_path = os.path.join(clip_dir, "video.mp4")
            out_file = None
            if os.path.exists(local_video_path) and os.path.getsize(local_video_path) > 10000:
                out_file = local_video_path
                logger.info(f"   ⚡ [LOCAL SOURCE CACHE HIT] Raw video already exists on disk: {clip_dir}/video.mp4 — skipping download")
            else:
                try:
                    from Telegram_Storage_Modules.telegram_vault_indexer import TelegramVaultIndexer
                    _indexer = TelegramVaultIndexer()
                    _clean_sc = _extract_shortcode(url) or shortcode
                    _v_hit = (
                        _indexer.find_entry_by_shortcode(_clean_sc) or
                        _indexer.find_entry_by_shortcode(url) or
                        _indexer.lookup_downloaded_source(url)
                    )
                    _has_vault_candidate = False
                    if _v_hit:
                        _clean_fid = (
                            _v_hit.get("media_file_ids", {}).get("wm_clean_file_id") or
                            _v_hit.get("wm_clean_file_id")
                        )
                        _raw_fid = (
                            _v_hit.get("media_file_ids", {}).get("raw_video_file_id") or
                            _v_hit.get("raw_video_file_id") or
                            _v_hit.get("raw_file_id")
                        )
                        _has_vault_candidate = bool(_clean_fid or _raw_fid)

                    # Check for watermark-cleaned clip FIRST, falling back to raw source video
                    if _has_vault_candidate or _clean_sc or url:
                        _hydrated = _indexer.hydrate_raw_video_from_vault(_clean_sc, clip_dir, check_clean_first=True)
                        if not _hydrated:
                            _hydrated = _indexer.hydrate_raw_video_from_vault(url, clip_dir, check_clean_first=True)
                        if _hydrated and os.path.exists(_hydrated) and os.path.getsize(_hydrated) > 1024:
                            out_file = _hydrated
                            _std_vid = os.path.join(clip_dir, "video.mp4")
                            if not os.path.exists(_std_vid) and os.path.exists(_hydrated):
                                try:
                                    import shutil
                                    shutil.copy2(_hydrated, _std_vid)
                                except Exception:
                                    pass
                            _kind = "watermark-cleaned" if "clean" in os.path.basename(_hydrated).lower() else "raw source"
                            logger.info(f"   ✅ [VAULT SOURCE CACHE HIT] Hydrated {_kind} from Telegram Vault — skipping yt-dlp download")
                            try:
                                _indexer.hydrate_extracted_audio_from_vault(_clean_sc, dest_dir=clip_dir)
                            except Exception:
                                pass
                except Exception as _v_err:
                    logger.debug(f"[WORKER 2] Vault hydration check notice: {_v_err}")

            if not out_file:
                res = download_video(url, custom_title="video", force_filename="video.mp4", destination_dir=clip_dir)
                out_file = res[0] if isinstance(res, (tuple, list)) else res

            if out_file and isinstance(out_file, str) and os.path.exists(out_file):
                meta_dict = {
                    "shortcode": shortcode,
                    "url": url,
                    "ownerUsername": "manual",
                    "platform": platform,
                    "caption": "",
                    "hashtags": [],
                    "taggedUsers": [],
                    "likesCount": 0,
                    "videoViewCount": 0,
                    "timestamp": None
                }

                # Try reading existing metadata.json if present
                meta_path = os.path.join(clip_dir, "metadata.json")
                if os.path.exists(meta_path):
                    try:
                        with open(meta_path, "r", encoding="utf-8") as mf:
                            existing_m = json.load(mf)
                        if isinstance(existing_m, dict):
                            meta_dict.update({k: v for k, v in existing_m.items() if v})
                    except Exception:
                        pass

                # Try reading yt-dlp video.json sidecar if present
                v_json_path = os.path.join(clip_dir, "video.json")
                if os.path.exists(v_json_path):
                    try:
                        with open(v_json_path, "r", encoding="utf-8") as vf:
                            v_data = json.load(vf)
                        if v_data.get("title") and not meta_dict.get("caption"):
                            meta_dict["caption"] = v_data["title"]
                            meta_dict["hashtags"] = re.findall(r"#(\w+)", v_data["title"])
                        if v_data.get("uploader") and v_data["uploader"] != "unknown" and meta_dict.get("ownerUsername") in ("manual", "unknown"):
                            meta_dict["ownerUsername"] = v_data["uploader"]
                    except Exception:
                        pass

                # Only call Apify as fallback if yt-dlp failed to extract basic metadata (caption/owner)
                has_ytdlp_meta = bool(
                    meta_dict.get("caption") and 
                    meta_dict.get("ownerUsername") and 
                    meta_dict["ownerUsername"] not in ("manual", "unknown")
                )
                if not has_ytdlp_meta and platform == "instagram" and "instagram.com" in url:
                    try:
                        from Downloader_Modules.apify_downloader import _get_client, _check_quota, _consume_quota
                        if _check_quota(1):
                            logger.info(f"🔍 [WORKER 2 METADATA] Fetching rich Instagram post metadata via Apify fallback for: {url}")
                            client = _get_client()
                            run = client.actor("apify/instagram-scraper").call(run_input={
                                "directUrls": [url],
                                "resultsType": "posts",
                                "resultsLimit": 1
                            })
                            _consume_quota(1)
                            items = list(client.dataset(run["defaultDatasetId"]).iterate_items())
                            if items:
                                item = items[0]
                                cap = item.get("caption") or meta_dict["caption"]
                                tags = item.get("hashtags") or (re.findall(r"#(\w+)", cap) if cap else [])
                                owner = item.get("ownerUsername") or item.get("owner", {}).get("username") or meta_dict["ownerUsername"]
                                meta_dict.update({
                                    "caption": cap,
                                    "hashtags": tags,
                                    "taggedUsers": item.get("taggedUsers") or [],
                                    "ownerUsername": owner,
                                    "likesCount": item.get("likesCount", 0),
                                    "videoViewCount": item.get("videoViewCount", 0),
                                    "timestamp": item.get("timestamp"),
                                })
                                logger.info(f"   ✓ [METADATA FETCH SUCCESS] Owner: @{owner} | Caption: {cap[:40]}... | Hashtags: {tags}")
                    except Exception as _m_err:
                        logger.warning(f"   ⚠ Rich metadata fetch warning: {_m_err}")
                elif has_ytdlp_meta:
                    logger.info(f"   ✓ [METADATA REUSED] Extracted from yt-dlp — Owner: @{meta_dict['ownerUsername']} | Caption: {meta_dict['caption'][:40]}... (Apify skipped)")

                meta_path = os.path.join(clip_dir, "metadata.json")
                with open(meta_path, "w", encoding="utf-8") as mf:
                    json.dump(meta_dict, mf, indent=2, ensure_ascii=False)
                downloaded_files.append(os.path.abspath(out_file))
                logger.info(f"   ✓ [WORKER 2 SUCCESS] Saved video -> {clip_dir}/video.mp4 & metadata.json")

                # ⚡ Background Upload to Telegram Storage Vault (Non-blocking)
                send_raw_download_to_telegram_vault_async(out_file, url, shortcode, platform, clip_dir)

                try:
                    from Main_Modules.proxy_encoder import encode_proxy
                    encode_proxy(out_file)
                except Exception as pe:
                    logger.warning(f"   ⚠ Proxy encode warning: {pe}")

                try:
                    from Audio_Modules.audio_extractor import run_phase1_audio_analysis
                    run_phase1_audio_analysis(out_file, clip_dir)
                except Exception as ae:
                    logger.warning(f"   ⚠ Audio analysis warning: {ae}")

            return {
                "success": len(downloaded_files) > 0,
                "mode": "manual",
                "count": len(downloaded_files),
                "downloaded_files": downloaded_files,
                "downloads_dir": downloads_dir,
                "error": None if downloaded_files else f"All download strategies exhausted for {url} (content may be restricted, age-gated, or private)"
            }
        except Exception as err:
            logger.error(f"❌ [WORKER 2 FAILED] Manual download error: {err}")
            return {"success": False, "mode": "manual", "downloaded_files": [], "error": str(err)}

    # ── WORKER 1: Automated Account Harvester ─────────────────────────────────
    else:
        logger.info("🤖 [WORKER 1 - AUTOMATED] Resolving target accounts...")
        accounts_file = os.path.join(_REPO_ROOT, "Content_Scraper_Modules", "source_accounts.json")
        sources = []

        if target_accounts and isinstance(target_accounts, list):
            sources = [a.strip().lstrip("@") for a in target_accounts if a.strip()]
        elif os.path.exists(accounts_file):
            try:
                with open(accounts_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    sources = data.get("source_accounts", [])
            except Exception as e:
                logger.warning(f"   ⚠ Failed to load source_accounts.json: {e}")

        if not sources:
            sources = []

        # Enforce Max 2 accounts limit per run
        sources = sources[:2]

        # Update source_accounts.json active target list
        try:
            if os.path.exists(accounts_file) and sources:
                with open(accounts_file, "r", encoding="utf-8") as f:
                    acc_data = json.load(f)
                acc_data["source_accounts"] = sources
                tmp_accounts_file = accounts_file + ".tmp"
                with open(tmp_accounts_file, "w", encoding="utf-8") as f:
                    json.dump(acc_data, f, indent=2, ensure_ascii=False)
                os.replace(tmp_accounts_file, accounts_file)
        except Exception as _se:
            logger.warning(f"   ⚠ Could not update source_accounts.json: {_se}")

        logger.info(f"   ✓ Targeting MAX 2 accounts for scrape: {sources}")
        try:
            import importlib
            from Downloader_Modules.apify_downloader import apify_scrape_creator_accounts
            from Downloader_Modules.downloader import download_video
            try:
                check_deduplication = importlib.import_module("Phase_1.02_dedup_ledger").check_deduplication
            except Exception:
                def check_deduplication(sc, owner=None, downloads_dir=None):
                    target_dir = os.path.join(downloads_dir or "", f"{owner}_{sc}")
                    is_dup = os.path.exists(target_dir)
                    v_path = os.path.join(target_dir, "video.mp4") if is_dup else None
                    return {"is_duplicate": is_dup, "video_path": v_path}

            approved_reels = apify_scrape_creator_accounts(
                "General",
                sources,
                limit_per_account=limit_per_account,
                bypass_cooldown=bool(target_accounts),
            )
            downloaded_files = []
            for item in (approved_reels or []):
                if isinstance(item, str) and os.path.exists(item):
                    downloaded_files.append(os.path.abspath(item))
                elif isinstance(item, dict):
                    shortcode = item.get("shortcode") or f"clip_{int(time.time())}"
                    owner = item.get("ownerUsername") or item.get("uploader") or "creator"
                    clip_folder_name = f"{owner}_{shortcode}"
                    clip_dir = os.path.join(downloads_dir, clip_folder_name)

                    # Deduplication check across Vault, Disk, and Content Ledger
                    dedup_info = check_deduplication(shortcode, owner=owner, downloads_dir=downloads_dir)
                    if dedup_info.get("is_duplicate"):
                        logger.info(f"♻️ [DEDUP SKIP] Reel {clip_folder_name} already exists in Vault/Disk/Ledger. Skipping from download queue.")
                        continue

                    os.makedirs(clip_dir, exist_ok=True)
                    video_url = item.get("videoUrl") or item.get("url") or (f"https://www.instagram.com/reel/{shortcode}/" if shortcode else None)
                    if video_url:
                        try:
                            logger.info(f"⬇️ [WORKER 1] Downloading approved reel ({clip_folder_name}): {video_url}")
                            res = download_video(
                                video_url,
                                custom_title="video",
                                force_filename="video.mp4",
                                destination_dir=clip_dir
                            )
                            out_file = res[0] if isinstance(res, (tuple, list)) else res
                            if out_file and isinstance(out_file, str) and os.path.exists(out_file):
                                # Save full JSON metadata alongside video.mp4
                                meta_path = os.path.join(clip_dir, "metadata.json")
                                with open(meta_path, "w", encoding="utf-8") as mf:
                                    json.dump(item, mf, indent=2, ensure_ascii=False)

                                downloaded_files.append(os.path.abspath(out_file))
                                logger.info(f"   ✓ [SAVED] {clip_folder_name}/video.mp4 & metadata.json")

                                # ⚡ Background Upload to Telegram Storage Vault (Non-blocking)
                                send_raw_download_to_telegram_vault_async(
                                    out_file,
                                    video_url,
                                    shortcode,
                                    item.get("platform", "instagram"),
                                    clip_dir
                                )

                                # ── Phase 1 Pre-Processing (Proxy 480p & Audio Ingestion) ──────
                                try:
                                    from Main_Modules.proxy_encoder import encode_proxy
                                    encode_proxy(out_file)
                                except Exception as proxy_err:
                                    logger.warning(f"   ⚠ Phase 1 proxy 480p encode failed: {proxy_err}")

                                try:
                                    from Audio_Modules.audio_extractor import run_phase1_audio_analysis
                                    logger.info(f"   🎵 [AUDIO] Running Phase 1 audio extraction + beat analysis...")
                                    run_phase1_audio_analysis(out_file, clip_dir)
                                except Exception as audio_err:
                                    logger.warning(f"   ⚠ Phase 1 audio analysis failed (non-fatal): {audio_err}")
                        except Exception as dl_err:
                            logger.warning(f"   ⚠ Failed to download reel {video_url}: {dl_err}")

            logger.info(f"   ✓ [WORKER 1 SUCCESS] Harvested {len(downloaded_files)} new clip(s) into downloads/")
        except Exception as harvest_err:
            logger.error(f"❌ [WORKER 1 FAILED] Automated harvest error: {harvest_err}")
            return {"success": False, "mode": "auto", "downloaded_files": [], "error": str(harvest_err)}

    return {
        "success": True,
        "mode": mode,
        "count": len(downloaded_files),
        "downloaded_files": downloaded_files,
        "downloads_dir": downloads_dir,
    }


# ── CLI Entrypoint ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1 Ingestion Orchestrator (Worker 1 & Worker 2)")
    parser.add_argument("--mode", type=str, choices=["auto", "manual"], default="auto", help="Ingestion mode ('auto' or 'manual')")
    parser.add_argument("--url", "-i", type=str, default=None, help="Target video URL for manual input mode")
    parser.add_argument("--limit", type=int, default=None, help="Max reels per account for automated mode (defaults to APIFY_REELS_PER_ACCOUNT in .env)")

    args = parser.parse_args()
    mode_to_use = "manual" if args.url else args.mode

    res = run_phase1_ingestion(mode=mode_to_use, url=args.url, limit_per_account=args.limit)
    if res.get("success"):
        print(f"\n🎉 PHASE 1 INGESTION COMPLETE: {res['count']} clip(s) ready in {res['downloads_dir']}")
    else:
        print(f"\n💥 INGESTION FAILED: {res.get('error')}")
        sys.exit(1)
