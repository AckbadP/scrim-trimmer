"""
Tests for frame_extractor.py — extract_frames, get_video_duration,
probe_duration, and check_source_video.

Error paths are tested with real cv2 behaviour (non-existent files).
Internal-state paths (zero FPS, duration calculation) use mocks.
"""

import os
import subprocess
import sys
from unittest.mock import MagicMock, patch

import cv2
import pytest

import frame_extractor
from frame_extractor import (
    check_source_video,
    extract_frames,
    get_video_duration,
    probe_duration,
)


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

# get_video_duration prefers probe_duration(); force it to "unavailable" so
# these tests exercise (and stay scoped to) the OpenCV fallback path, exactly
# as before probe_duration existed.
@pytest.fixture(autouse=True)
def _no_probe(monkeypatch):
    monkeypatch.setattr(frame_extractor, "probe_duration", lambda path, timeout=30.0: None)


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

    def test_prefers_probed_duration_over_opencv(self, monkeypatch):
        # When probe_duration succeeds, get_video_duration must use it and
        # never touch cv2 at all — this is what makes duration reliable
        # against an unfinalized/still-growing recording, where OpenCV's
        # frame-count estimate can be garbage.
        monkeypatch.setattr(frame_extractor, "probe_duration", lambda path, timeout=30.0: 123.4)
        with patch("cv2.VideoCapture") as mock_cap_cls:
            result = get_video_duration("fake.mp4")
        assert result == pytest.approx(123.4)
        mock_cap_cls.assert_not_called()


# ---------------------------------------------------------------------------
# probe_duration
# ---------------------------------------------------------------------------

class TestProbeDuration:
    def test_returns_none_when_ffprobe_missing(self, monkeypatch):
        monkeypatch.setattr("frame_extractor.shutil.which", lambda name: None)
        assert probe_duration("whatever.mp4") is None

    def test_returns_duration_on_success(self, monkeypatch):
        monkeypatch.setattr("frame_extractor.shutil.which", lambda name: "/usr/bin/ffprobe")
        result = MagicMock(returncode=0, stdout="42.500000\n")
        with patch("subprocess.run", return_value=result) as mock_run:
            duration = probe_duration("video.mp4")
        assert duration == pytest.approx(42.5)
        # stdin must be redirected — an ffprobe inheriting an interactive
        # stdin should never be able to block waiting on it.
        assert mock_run.call_args.kwargs.get("stdin") == subprocess.DEVNULL

    def test_returns_none_on_nonzero_returncode(self, monkeypatch):
        monkeypatch.setattr("frame_extractor.shutil.which", lambda name: "/usr/bin/ffprobe")
        result = MagicMock(returncode=1, stdout="")
        with patch("subprocess.run", return_value=result):
            assert probe_duration("missing.mp4") is None

    def test_returns_none_on_unparseable_output(self, monkeypatch):
        monkeypatch.setattr("frame_extractor.shutil.which", lambda name: "/usr/bin/ffprobe")
        result = MagicMock(returncode=0, stdout="N/A\n")
        with patch("subprocess.run", return_value=result):
            assert probe_duration("weird.mp4") is None

    def test_returns_none_on_zero_duration(self, monkeypatch):
        monkeypatch.setattr("frame_extractor.shutil.which", lambda name: "/usr/bin/ffprobe")
        result = MagicMock(returncode=0, stdout="0.000000\n")
        with patch("subprocess.run", return_value=result):
            assert probe_duration("empty.mp4") is None

    def test_returns_none_on_timeout(self, monkeypatch):
        monkeypatch.setattr("frame_extractor.shutil.which", lambda name: "/usr/bin/ffprobe")
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="ffprobe", timeout=30)):
            assert probe_duration("slow.mp4") is None

    def test_real_ffprobe_on_missing_file(self):
        # End-to-end sanity check against the real ffprobe binary, if present.
        if not __import__("shutil").which("ffprobe"):
            pytest.skip("ffprobe not installed")
        assert probe_duration("/no/such/file.mp4") is None


# ---------------------------------------------------------------------------
# check_source_video
# ---------------------------------------------------------------------------

class TestCheckSourceVideo:
    def test_none_when_stable_and_probeable(self, monkeypatch, tmp_path):
        video = tmp_path / "finished.mp4"
        video.write_bytes(b"x" * 100)
        monkeypatch.setattr("frame_extractor.time.sleep", lambda s: None)
        monkeypatch.setattr(frame_extractor, "probe_duration", lambda path, timeout=30.0: 60.0)
        assert check_source_video(str(video), grow_check_wait=0.01) is None

    def test_warns_when_file_is_growing(self, monkeypatch, tmp_path):
        video = tmp_path / "recording.mp4"
        video.write_bytes(b"x" * 100)
        sizes = iter([100, 250])
        monkeypatch.setattr("frame_extractor.os.path.getsize", lambda p: next(sizes))
        monkeypatch.setattr("frame_extractor.time.sleep", lambda s: None)
        warning = check_source_video(str(video), grow_check_wait=0.01)
        assert warning is not None
        assert "still" in warning.lower() or "growing" in warning.lower()

    def test_warns_when_duration_unprobeable(self, monkeypatch, tmp_path):
        video = tmp_path / "unfinalized.mp4"
        video.write_bytes(b"x" * 100)
        monkeypatch.setattr("frame_extractor.time.sleep", lambda s: None)
        monkeypatch.setattr(frame_extractor, "probe_duration", lambda path, timeout=30.0: None)
        warning = check_source_video(str(video), grow_check_wait=0.01)
        assert warning is not None
        assert "duration" in warning.lower() or "index" in warning.lower()

    def test_none_on_missing_file(self, monkeypatch):
        # getsize raising OSError (e.g. file vanished) must not be treated
        # as a hang/crash trigger — just skip the check silently.
        monkeypatch.setattr(
            "frame_extractor.os.path.getsize",
            MagicMock(side_effect=OSError("no such file")),
        )
        assert check_source_video("/no/such/file.mp4") is None


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

    def test_cap_released_when_generator_abandoned_mid_iteration(self):
        # A consumer that stops early (e.g. Cancel, or an exception raised
        # between yields) throws GeneratorExit into the generator at its
        # current `yield`. Without a try/finally around the loop, cap never
        # gets released and the capture handle leaks.
        import numpy as np
        fake_frame = np.zeros((10, 10, 3), dtype=np.uint8)
        cap = _mock_cap(fps=1.0, total_frames=100)
        cap.read.return_value = (True, fake_frame)
        with patch("cv2.VideoCapture", return_value=cap):
            gen = extract_frames("fake.mp4")
            next(gen)          # pull one frame, leaving the generator suspended at yield
            gen.close()        # simulate the consumer abandoning iteration
        cap.release.assert_called_once()
