"""
Tests for frame_extractor.py — extract_frames and get_video_duration.

Error paths are tested with real cv2 behaviour (non-existent files).
Internal-state paths (zero FPS, duration calculation) use mocks.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import cv2
import pytest

from frame_extractor import extract_frames, get_video_duration


def _mock_cap(fps: float, total_frames: int, opened: bool = True):
    cap = MagicMock()
    cap.isOpened.return_value = opened

    def _get(prop):
        if prop == cv2.CAP_PROP_FPS:
            return fps
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return float(total_frames)
        return 0.0

    cap.get.side_effect = _get
    return cap


# ---------------------------------------------------------------------------
# get_video_duration
# ---------------------------------------------------------------------------

class TestGetVideoDuration:
    def test_raises_on_nonexistent_file(self):
        with pytest.raises(ValueError, match="Cannot open video"):
            get_video_duration("/no/such/file.mp4")

    def test_returns_zero_when_fps_is_zero(self):
        cap = _mock_cap(fps=0.0, total_frames=100)
        with patch("cv2.VideoCapture", return_value=cap):
            result = get_video_duration("fake.mp4")
        assert result == 0.0

    def test_duration_calculation(self):
        # 300 frames at 30 fps → 10.0 seconds
        cap = _mock_cap(fps=30.0, total_frames=300)
        with patch("cv2.VideoCapture", return_value=cap):
            result = get_video_duration("fake.mp4")
        assert result == pytest.approx(10.0)

    def test_non_integer_duration(self):
        # 100 frames at 24 fps → ~4.166…s
        cap = _mock_cap(fps=24.0, total_frames=100)
        with patch("cv2.VideoCapture", return_value=cap):
            result = get_video_duration("fake.mp4")
        assert result == pytest.approx(100 / 24.0)

    def test_cap_is_released(self):
        cap = _mock_cap(fps=30.0, total_frames=300)
        with patch("cv2.VideoCapture", return_value=cap):
            get_video_duration("fake.mp4")
        cap.release.assert_called_once()


# ---------------------------------------------------------------------------
# extract_frames
# ---------------------------------------------------------------------------

class TestExtractFrames:
    def test_raises_on_nonexistent_file(self):
        with pytest.raises(ValueError, match="Cannot open video"):
            list(extract_frames("/no/such/file.mp4"))

    def test_raises_on_zero_fps(self):
        cap = _mock_cap(fps=0.0, total_frames=100)
        with patch("cv2.VideoCapture", return_value=cap):
            with pytest.raises(ValueError, match="Invalid FPS"):
                list(extract_frames("fake.mp4"))

    def test_cap_released_on_fps_error(self):
        cap = _mock_cap(fps=0.0, total_frames=100)
        with patch("cv2.VideoCapture", return_value=cap):
            with pytest.raises(ValueError):
                list(extract_frames("fake.mp4"))
        cap.release.assert_called_once()

    def test_yields_second_and_frame_tuples(self):
        import numpy as np
        fake_frame = np.zeros((10, 10, 3), dtype=np.uint8)
        cap = _mock_cap(fps=1.0, total_frames=3)
        cap.read.return_value = (True, fake_frame)
        with patch("cv2.VideoCapture", return_value=cap):
            results = list(extract_frames("fake.mp4"))
        seconds = [s for s, _ in results]
        assert seconds == [0, 1, 2]   # 3 frames at 1 fps → seconds 0, 1, 2

    def test_stops_when_read_fails(self):
        import numpy as np
        fake_frame = np.zeros((10, 10, 3), dtype=np.uint8)
        # Sequential decoder: a failed read terminates iteration (stream ended/error)
        cap = _mock_cap(fps=1.0, total_frames=3)
        cap.read.side_effect = [
            (True, fake_frame),
            (False, None),   # fails at second=1 → stop
        ]
        with patch("cv2.VideoCapture", return_value=cap):
            results = list(extract_frames("fake.mp4"))
        seconds = [s for s, _ in results]
        assert seconds == [0]

    def test_target_frame_exceeds_total_stops_iteration(self):
        # fps=2, total_frames=2: after yielding second=0, next target_frame=2 >= 2
        # → break on line 44, only one frame yielded.
        import numpy as np
        fake_frame = np.zeros((10, 10, 3), dtype=np.uint8)
        cap = _mock_cap(fps=2.0, total_frames=2)
        cap.read.return_value = (True, fake_frame)
        with patch("cv2.VideoCapture", return_value=cap):
            results = list(extract_frames("fake.mp4"))
        seconds = [s for s, _ in results]
        assert seconds == [0]

    def test_grab_failure_stops_iteration(self):
        # fps=2, total_frames=5: yields second=0, grabs frame 1 successfully
        # (covers current_frame += 1 at line 52), reads frame 2, yields second=1,
        # then tries to grab frame 3 but grab() fails → early return.
        import numpy as np
        fake_frame = np.zeros((10, 10, 3), dtype=np.uint8)
        cap = _mock_cap(fps=2.0, total_frames=5)
        cap.read.return_value = (True, fake_frame)
        cap.grab.side_effect = [True, False]  # first grab succeeds, second fails
        with patch("cv2.VideoCapture", return_value=cap):
            results = list(extract_frames("fake.mp4"))
        seconds = [s for s, _ in results]
        assert seconds == [0, 1]
        cap.release.assert_called()
