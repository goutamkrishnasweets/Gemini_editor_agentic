"""
Configuration module for Manga or Comic page extractor.
Provides default directory paths and pipeline hyperparameters.
All storage, temp, downloads, and output directories reside at the project root.
"""

import os

# Root directory of the workspace (Affilate marketing assitant)
_EXTRACTOR_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_EXTRACTOR_DIR)

# Workspace Root Directories (No Phase 1 subfolders)
TEMP_DIR = os.getenv("TEMP_DIR", os.path.join(_ROOT_DIR, "temp"))
OUTPUT_DIR = os.getenv("OUTPUT_DIR", os.path.join(_ROOT_DIR, "output"))
DOWNLOADS_DIR = os.getenv("DOWNLOADS_DIR", os.path.join(_ROOT_DIR, "downloads"))
STORAGE_PAGES_DIR = os.getenv("STORAGE_PAGES_DIR", os.path.join(_ROOT_DIR, "storage", "manga_pages"))

# Extraction output JSON points directly to root output directory
PHASE1_JSON_DIR = os.getenv("PHASE1_JSON_DIR", OUTPUT_DIR)
EXTRACTION_JSON_DIR = PHASE1_JSON_DIR

# Pipeline Settings
PANEL_CONFIDENCE_THRESHOLD = float(os.getenv("PANEL_CONFIDENCE_THRESHOLD", "0.5"))
OCR_LANGUAGES = ["en"]
USE_GPU = os.getenv("USE_GPU", "false").lower() in ("true", "1", "yes")

# Ensure required directories exist at workspace root
for _d in [TEMP_DIR, OUTPUT_DIR, DOWNLOADS_DIR, STORAGE_PAGES_DIR]:
    os.makedirs(_d, exist_ok=True)
