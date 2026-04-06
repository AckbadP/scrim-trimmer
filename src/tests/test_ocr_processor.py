"""
Tests for ocr_processor.py — crop_chat_region, preprocess_for_ocr, and run_ocr.
"""

import os
import sys
import unittest.mock as mock

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import ocr_processor
from ocr_processor import crop_chat_region, preprocess_for_ocr


def _frame(h: int = 400, w: int = 640) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# crop_chat_region
# ---------------------------------------------------------------------------

class TestCropChatRegion:
    def test_default_region_dimensions(self):
        # Default: x1=0, y1=35%, x2=15%, y2=100%
        # On 640×400: width=96px (15% of 640), height=260px (65% of 400)
        frame = _frame(400, 640)
        region = crop_chat_region(frame)
        assert region.shape[0] == 260
        assert region.shape[1] == 96

    def test_full_frame_region(self):
        frame = _frame(400, 640)
        region = crop_chat_region(frame, 0.0, 0.0, 1.0, 1.0)
        assert region.shape == frame.shape

    def test_custom_region_dimensions(self):
        # On 200×400: x1=25%=100, y1=50%=100, x2=75%=300, y2=100%=200
        # → width=200, height=100
        frame = _frame(200, 400)
        region = crop_chat_region(frame, 0.25, 0.5, 0.75, 1.0)
        assert region.shape[0] == 100
        assert region.shape[1] == 200

    def test_pixel_values_preserved(self):
        # Fill bottom-left quadrant with a distinct colour
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[50:100, 0:50] = [0, 128, 255]
        region = crop_chat_region(frame, 0.0, 0.5, 0.5, 1.0)
        assert np.all(region == [0, 128, 255])

    def test_zero_size_region(self):
        # x1 == x2 → empty crop (0-width)
        frame = _frame(100, 100)
        region = crop_chat_region(frame, 0.5, 0.0, 0.5, 1.0)
        assert region.shape[1] == 0

    def test_returns_bgr_array(self):
        frame = _frame()
        region = crop_chat_region(frame)
        assert region.ndim == 3
        assert region.shape[2] == 3


# ---------------------------------------------------------------------------
# preprocess_for_ocr
# ---------------------------------------------------------------------------

class TestPreprocessForOcr:
    def test_returns_pil_image(self):
        result = preprocess_for_ocr(_frame(50, 80))
        assert isinstance(result, Image.Image)

    def test_output_is_2x_input_size(self):
        # Input 50h×80w → output PIL size (width, height) = (160, 100)
        result = preprocess_for_ocr(_frame(50, 80))
        assert result.size == (160, 100)

    def test_dark_background_inverted_to_white(self):
        # All-black input (EVE dark background) → inverted to white
        dark = np.zeros((20, 20, 3), dtype=np.uint8)
        result = preprocess_for_ocr(dark)
        assert np.all(np.array(result) == 255)

    def test_white_background_inverted_to_black(self):
        white = np.full((20, 20, 3), 255, dtype=np.uint8)
        result = preprocess_for_ocr(white)
        assert np.all(np.array(result) == 0)

    def test_output_is_grayscale(self):
        # Result is a single-channel (L-mode) PIL image
        result = preprocess_for_ocr(_frame(30, 30))
        assert result.mode == "L"


# ---------------------------------------------------------------------------
# run_ocr — Tesseract path
# ---------------------------------------------------------------------------

class TestRunOcr:
    """Tests for run_ocr — uses mocks, no real Tesseract binary needed."""

    def _make_frame(self):
        return np.zeros((100, 100, 3), dtype=np.uint8)

    def test_calls_tesseract(self):
        frame = self._make_frame()
        with mock.patch('ocr_processor.pytesseract.image_to_string', return_value='hello') as mock_tess:
            result = ocr_processor.run_ocr(frame)
        assert result == 'hello'
        mock_tess.assert_called_once()

    def test_run_ocr_on_region_calls_tesseract(self):
        region = np.zeros((50, 50, 3), dtype=np.uint8)
        with mock.patch('ocr_processor.pytesseract.image_to_string', return_value='world') as mock_tess:
            result = ocr_processor.run_ocr_on_region(region)
        assert result == 'world'
        mock_tess.assert_called_once()
