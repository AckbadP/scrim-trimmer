"""
ocr_processor.py - Run OCR on the chat window region of a video frame.

The chat region is defined as an arbitrary box (x1, y1, x2, y2) expressed as
fractions of the frame dimensions.  Default: left 15% x bottom 65% of height,
which matches the EVE Online local chat window position.

GPU acceleration: if `easyocr` is installed and a CUDA GPU is available,
EasyOCR is used instead of Tesseract. Falls back to Tesseract automatically.
"""

import cv2
import numpy as np
import pytesseract
from PIL import Image

try:
    import easyocr
    import torch
    _GPU_AVAILABLE = torch.cuda.is_available()
except ImportError:
    easyocr = None
    _GPU_AVAILABLE = False

# GPU OCR disabled: EasyOCR does not produce reliable results for EVE chat UI.
# Re-enable when a suitable GPU engine is found.
_easyocr_reader = None
# if easyocr is not None and _GPU_AVAILABLE:
#     _easyocr_reader = easyocr.Reader(['en'], gpu=True)


def crop_chat_region(
    frame: np.ndarray,
    x1_pct: float = 0.0,
    y1_pct: float = 0.35,
    x2_pct: float = 0.15,
    y2_pct: float = 1.0,
) -> np.ndarray:
    """
    Crop an arbitrary chat window region from a full game frame.

    Args:
        frame:   BGR numpy array (full video frame).
        x1_pct: Left edge as fraction of frame width (0.0–1.0).
        y1_pct: Top edge as fraction of frame height (0.0–1.0).
        x2_pct: Right edge as fraction of frame width (0.0–1.0).
        y2_pct: Bottom edge as fraction of frame height (0.0–1.0).

    Returns:
        Cropped BGR numpy array containing the chat region.
    """
    h, w = frame.shape[:2]
    x1 = int(w * x1_pct)
    y1 = int(h * y1_pct)
    x2 = int(w * x2_pct)
    y2 = int(h * y2_pct)
    return frame[y1:y2, x1:x2]


def preprocess_for_ocr(region: np.ndarray) -> Image.Image:
    """
    Preprocess the chat region for better OCR accuracy.

    EVE Online chat uses light text on a dark background.
    Steps:
    1. Convert to grayscale
    2. Invert (dark background → white, light text → dark)
    3. Scale up 2x for better tesseract accuracy on small fonts
    """
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    inverted = cv2.bitwise_not(gray)
    scaled = cv2.resize(inverted, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return Image.fromarray(scaled)


def _run_ocr_tesseract(region: np.ndarray) -> str:
    pil_img = preprocess_for_ocr(region)
    # PSM 6: Assume a single uniform block of text (good for chat columns)
    config = "--psm 6 --oem 3"
    return pytesseract.image_to_string(pil_img, config=config)


def _run_ocr_gpu(region: np.ndarray) -> str:
    results = _easyocr_reader.readtext(region)
    results.sort(key=lambda r: r[0][0][1])  # sort by top-left y-coordinate
    return "\n".join(text for _, text, _ in results)


def run_ocr_on_region(region: np.ndarray) -> str:
    """
    Extract text from a pre-cropped chat region.

    Uses GPU-accelerated EasyOCR if available, otherwise falls back to Tesseract.

    Args:
        region: Pre-cropped BGR numpy array of the chat window.

    Returns:
        OCR text string from the chat window.
    """
    if _easyocr_reader is not None:
        return _run_ocr_gpu(region)
    return _run_ocr_tesseract(region)


def run_ocr(
    frame: np.ndarray,
    x1_pct: float = 0.0,
    y1_pct: float = 0.35,
    x2_pct: float = 0.15,
    y2_pct: float = 1.0,
) -> str:
    """
    Extract text from the chat window region of a video frame.

    Uses GPU-accelerated EasyOCR if available, otherwise falls back to Tesseract.

    Args:
        frame:   Full BGR video frame.
        x1_pct: Left edge as fraction of frame width.
        y1_pct: Top edge as fraction of frame height.
        x2_pct: Right edge as fraction of frame width.
        y2_pct: Bottom edge as fraction of frame height.

    Returns:
        OCR text string from the chat window.
    """
    region = crop_chat_region(frame, x1_pct=x1_pct, y1_pct=y1_pct,
                               x2_pct=x2_pct, y2_pct=y2_pct)
    return run_ocr_on_region(region)
