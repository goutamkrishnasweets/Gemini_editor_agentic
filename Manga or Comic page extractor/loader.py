# loader.py - Mature Manga/Comic Input & Processor (Routed through Universal Downloader)
import os
import logging
import glob
import time
from datetime import datetime
import re
import json
import hashlib
import shutil
import subprocess
from typing import Dict, Optional, List
import sys

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if _CURRENT_DIR not in sys.path:
    sys.path.insert(0, _CURRENT_DIR)

_REPO_ROOT = os.path.dirname(_CURRENT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    from extractors.manga_page_extractor import MangaPageExtractor
except ImportError:
    from .extractors.manga_page_extractor import MangaPageExtractor

try:
    from config import TEMP_DIR, DOWNLOADS_DIR
except ImportError:
    TEMP_DIR = os.path.join(_REPO_ROOT, "temp")
    DOWNLOADS_DIR = os.path.join(_REPO_ROOT, "downloads")

logger = logging.getLogger("loader")
os.makedirs(DOWNLOADS_DIR, exist_ok=True)


def _extract_pdf_pages(pdf_path: str) -> List[str]:
    """Extracts all pages from a manga PDF to PNG images using MangaPageExtractor directly."""
    extractor = MangaPageExtractor(target_dpi=150)
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    clean_name = "".join(c if c.isalnum() else "_" for c in base_name).strip("_")
    out_dir = os.path.join(TEMP_DIR, f"pages_{clean_name}")
    return extractor.extract_pages_from_pdf(
        pdf_path=pdf_path,
        chapter_id=clean_name,
        target_dir=out_dir,
        zero_padded=True
    )


def process_manga_input(input_path: str) -> List[str]:
    """
    Main entry point for pipeline input.
    Directly processes local files (.pdf, images, folders), and routes URLs
    through universal_downloader if available.
    """
    try:
        # 1. Direct local file handling
        if os.path.exists(input_path):
            if input_path.lower().endswith(".pdf"):
                return _extract_pdf_pages(input_path)
            if os.path.isdir(input_path):
                valid_exts = (".png", ".jpg", ".jpeg", ".webp")
                files = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.lower().endswith(valid_exts)]
                files.sort()
                return files
            return [input_path]

        # 2. Remote URL handling via Downloader_Modules if available
        try:
            from Tools.Downloader_Modules.downloader import ingest
            logger.info("🛡️ [INGESTION] Routing media through Universal Downloader: %s", input_path)
            scan_result = ingest(input_path)
            if scan_result and getattr(scan_result, 'safe', False) and getattr(scan_result, 'safe_path', None):
                safe_path = scan_result.safe_path
                if safe_path.lower().endswith(".pdf"):
                    return _extract_pdf_pages(safe_path)
                return [safe_path]
        except Exception:
            pass

        logger.error("❌ Input file or resource not found: %s", input_path)
        return []
    except Exception as exc:
        logger.error("❌ Exception during process_manga_input for '%s': %s", input_path, exc)
        return []
