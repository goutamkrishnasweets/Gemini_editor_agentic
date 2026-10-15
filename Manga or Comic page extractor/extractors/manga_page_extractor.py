"""
Phase_1/extractors/manga_page_extractor.py — High-Res Manga PDF Page Extraction Module
======================================================================================
Extracts manga pages from input PDF files as high-resolution PNG image files.
Uses PyMuPDF (fitz) with fallback options.
"""

import os
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger("manga_page_extractor")

import sys
_PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PACKAGE_DIR not in sys.path:
    sys.path.insert(0, _PACKAGE_DIR)

_REPO_ROOT = os.path.dirname(_PACKAGE_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    from config import STORAGE_PAGES_DIR
except ImportError:
    STORAGE_PAGES_DIR = os.path.join(_REPO_ROOT, "storage", "manga_pages")



class MangaPageExtractor:
    def __init__(self, target_dpi: int = 150):
        self.target_dpi = target_dpi

    def extract_pages_from_pdf(
        self,
        pdf_path: str,
        chapter_id: str = "temp",
        target_dir: Optional[str] = None,
        zero_padded: bool = True,
        use_zoom_matrix: bool = False,
        zoom_scale: float = 2.0
    ) -> List[str]:
        """
        Extracts each page of a manga PDF as a PNG image.
        Default output: storage/manga_pages/{chapter_id}/.
        Can override destination with target_dir.
        
        Returns list of extracted absolute file paths.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Manga PDF file not found: {pdf_path}")

        chapter_pages_dir = target_dir if target_dir else os.path.join(STORAGE_PAGES_DIR, chapter_id)
        os.makedirs(chapter_pages_dir, exist_ok=True)

        extracted_paths = []

        try:
            import pymupdf as fitz
        except ImportError:
            import fitz

        logger.info(f"📖 [PAGE EXTRACTOR] Extracting pages from '{pdf_path}'...")
        doc = fitz.open(pdf_path)
        total_pages = len(doc)

        for page_idx in range(total_pages):
            page = doc[page_idx]
            if use_zoom_matrix:
                pix = page.get_pixmap(matrix=fitz.Matrix(zoom_scale, zoom_scale))
            else:
                pix = page.get_pixmap(dpi=self.target_dpi)

            if zero_padded:
                out_filename = f"page_{page_idx + 1:02d}.png"
            else:
                out_filename = f"page_{page_idx}.png"

            out_path = os.path.join(chapter_pages_dir, out_filename)
            pix.save(out_path)
            extracted_paths.append(out_path)

        doc.close()
        logger.info(f"✅ Extracted {len(extracted_paths)} page image(s) to '{chapter_pages_dir}'.")
        return extracted_paths
