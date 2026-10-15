"""
Phase_1 — Manga/Comic Asset Extraction & Conversion Package
============================================================
Contains core Phase 1 modules:
  - loader.py (Media ingestion entry point & PDF page handler)
  - pdf_extractor.py (PyMuPDF high-DPI page rasterization)
  - scanner.py (OpenCV PanelDetector contour cropping)
  - reader.py (EasyOCR speech bubble text extraction)
  - analyzer.py (SceneGrouper panel clustering)
  - extraction_main.py (Phase 1 pipeline orchestrator)
"""

try:
    from .loader import process_manga_input
except Exception:
    process_manga_input = None

try:
    from .extractors.manga_page_extractor import MangaPageExtractor
except Exception:
    MangaPageExtractor = None

try:
    from .scanner import PanelDetector
except Exception:
    PanelDetector = None

try:
    from .reader import OCREngine
except Exception:
    OCREngine = None

try:
    from .analyzer import SceneGrouper
except Exception:
    SceneGrouper = None

try:
    from .director import Phase1Director
except Exception:
    Phase1Director = None

__all__ = [
    "process_manga_input",
    "process_pdf",
    "PanelDetector",
    "OCREngine",
    "SceneGrouper",
    "Phase1Director",
]
