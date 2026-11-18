import os
import logging
import requests
import time

logger = logging.getLogger("font_manager")

FONT_URLS = [
    "https://raw.githubusercontent.com/JulietaUla/Montserrat/master/fonts/ttf/Montserrat-Bold.ttf",
    "https://github.com/google/fonts/raw/main/ofl/montserrat/Montserrat-Bold.ttf",
    "https://github.com/JulietaUla/Montserrat/raw/master/fonts/ttf/Montserrat-Bold.ttf"
]
# Anchor font dir to the project root (this file lives at Text_Modules/font_manager.py,
# so the root is two levels up: Text_Modules -> project root)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOCAL_FONT_DIR = os.path.join(_PROJECT_ROOT, "assets", "fonts")
LOCAL_FONT_PATH = os.path.join(LOCAL_FONT_DIR, "Montserrat-Bold.ttf")
MIN_FONT_SIZE_BYTES = 80 * 1024  # Reject if under 80KB

def ensure_montserrat_font():
    """
    Downloads Montserrat-Bold.ttf if it does not exist or is corrupted.
    Returns the absolute path to the font file.
    """
    os.makedirs(LOCAL_FONT_DIR, exist_ok=True)
    
    if _validate_font():
        return os.path.abspath(LOCAL_FONT_PATH)
        
    logger.info("⬇️ Downloading Montserrat-Bold.ttf...")
    for url in FONT_URLS:
        for attempt in range(2):
            try:
                response = requests.get(url, timeout=15, allow_redirects=True)
                response.raise_for_status()
                
                with open(LOCAL_FONT_PATH, "wb") as f:
                    f.write(response.content)
                    
                if _validate_font():
                    logger.info(f"✅ Montserrat font downloaded and verified from: {url}")
                    return os.path.abspath(LOCAL_FONT_PATH)
                else:
                    logger.error(f"❌ Downloaded font validation failed (file too small) from: {url}")
                    
            except Exception as e:
                logger.warning(f"⚠️ Font download attempt from {url} failed: {e}")
                time.sleep(1)

    for sys_font in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ):
        if os.path.exists(sys_font):
            logger.info(f"🔤 Using Linux system fallback font: {sys_font}")
            return sys_font

    logger.error("❌ Failed to download Montserrat font. ASS subtitles may fallback or fail.")
    return None

def _validate_font():
    if not os.path.exists(LOCAL_FONT_PATH):
        return False
    if os.path.getsize(LOCAL_FONT_PATH) < MIN_FONT_SIZE_BYTES:
        return False
    return True
