"""
Tests for ocr_processor.py — crop_chat_region and preprocess_for_ocr.

run_ocr is not tested here because it requires the Tesseract binary or a GPU.
Dispatch logic is tested via mocking.
"""

import importlib
import os
import sys
import unittest.mock as mock

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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
# run_ocr dispatch logic
# ---------------------------------------------------------------------------

class TestRunOcrDispatching:
    """Tests for run_ocr's GPU vs Tesseract dispatch — uses mocks, no real OCR."""

    def _make_frame(self):
        return np.zeros((100, 100, 3), dtype=np.uint8)

    def test_uses_tesseract_when_reader_is_none(self):
        import ocr_processor
        frame = self._make_frame()
        with mock.patch.object(ocr_processor, '_easyocr_reader', None), \
             mock.patch('ocr_processor.pytesseract.image_to_string', return_value='tesseract text') as mock_tess:
            result = ocr_processor.run_ocr(frame)
        assert result == 'tesseract text'
        mock_tess.assert_called_once()

    def test_uses_gpu_when_reader_is_set(self):
        import ocr_processor
        frame = self._make_frame()
        bbox = [[0, 0], [10, 0], [10, 10], [0, 10]]
        mock_reader = mock.Mock()
        mock_reader.readtext.return_value = [(bbox, 'gpu text', 0.99)]
        with mock.patch.object(ocr_processor, '_easyocr_reader', mock_reader), \
             mock.patch('ocr_processor.pytesseract.image_to_string') as mock_tess:
            result = ocr_processor.run_ocr(frame)
        assert result == 'gpu text'
        mock_tess.assert_not_called()

    def test_gpu_joins_multiline_results(self):
        import ocr_processor
        frame = self._make_frame()
        bbox_top = [[0, 5], [10, 5], [10, 15], [0, 15]]
        bbox_bottom = [[0, 20], [10, 20], [10, 30], [0, 30]]
        mock_reader = mock.Mock()
        mock_reader.readtext.return_value = [
            (bbox_top, 'line one', 0.99),
            (bbox_bottom, 'line two', 0.98),
        ]
        with mock.patch.object(ocr_processor, '_easyocr_reader', mock_reader):
            result = ocr_processor.run_ocr(frame)
        assert result == 'line one\nline two'

    def test_gpu_sorts_by_y_coordinate(self):
        import ocr_processor
        frame = self._make_frame()
        # Return results in reverse y-order (bottom first)
        bbox_higher_y = [[0, 50], [10, 50], [10, 60], [0, 60]]
        bbox_lower_y = [[0, 5], [10, 5], [10, 15], [0, 15]]
        mock_reader = mock.Mock()
        mock_reader.readtext.return_value = [
            (bbox_higher_y, 'second line', 0.98),
            (bbox_lower_y, 'first line', 0.99),
        ]
        with mock.patch.object(ocr_processor, '_easyocr_reader', mock_reader):
            result = ocr_processor.run_ocr(frame)
        assert result == 'first line\nsecond line'


# ---------------------------------------------------------------------------
# Module-level init (GPU detection)
# ---------------------------------------------------------------------------

class TestModuleInit:
    """Tests for the _easyocr_reader initialization at import time."""

    def _reload_ocr_processor(self):
        """Reload ocr_processor with current sys.modules state."""
        sys.modules.pop('ocr_processor', None)
        import ocr_processor as m
        return m

    def test_reader_is_none_when_easyocr_not_installed(self):
        fake_modules = {'easyocr': None, 'torch': None}
        with mock.patch.dict(sys.modules, fake_modules):
            m = self._reload_ocr_processor()
        assert m._easyocr_reader is None

    def test_reader_is_none_when_no_gpu(self):
        mock_torch = mock.Mock()
        mock_torch.cuda.is_available.return_value = False
        mock_easyocr = mock.Mock()
        fake_modules = {'easyocr': mock_easyocr, 'torch': mock_torch}
        with mock.patch.dict(sys.modules, fake_modules):
            m = self._reload_ocr_processor()
        assert m._easyocr_reader is None
        mock_easyocr.Reader.assert_not_called()
