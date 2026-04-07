"""
Tests for video_clipper.py — extract_clip, stitch_clips, create_clips.

subprocess.run is mocked so no real ffmpeg or video files are needed.
"""

import os
import sys
import tempfile
from unittest.mock import MagicMock, call, patch

import pytest

from video_clipper import create_clips, extract_clip, stitch_clips


def _ok():
    m = MagicMock()
    m.returncode = 0
    return m


def _fail(stderr="ffmpeg error"):
    m = MagicMock()
    m.returncode = 1
    m.stderr = stderr
    return m


# Force the ffmpeg code path for all tests in this module.
@pytest.fixture(autouse=True)
def force_ffmpeg(monkeypatch):
    import video_clipper
    monkeypatch.setattr(video_clipper, "_force_python", False)
    monkeypatch.setattr(video_clipper, "_use_python_clipper", lambda: False)


# ---------------------------------------------------------------------------
# stitch_clips
# ---------------------------------------------------------------------------

class TestStitchClips:
    def test_empty_list_raises_value_error(self):
        with pytest.raises(ValueError, match="No clips"):
            stitch_clips([], "out.mp4")

    def test_ffmpeg_failure_raises_runtime_error(self):
        with patch("subprocess.run", return_value=_fail("concat error")):
            with pytest.raises(RuntimeError, match="ffmpeg concat"):
                stitch_clips(["a.mp4"], "out.mp4")

    def test_success_calls_subprocess(self):
        with patch("subprocess.run", return_value=_ok()) as mock_run:
            stitch_clips(["a.mp4", "b.mp4"], "out.mp4")
        mock_run.assert_called_once()

    def test_command_uses_concat_demuxer(self):
        with patch("subprocess.run", return_value=_ok()) as mock_run:
            stitch_clips(["a.mp4"], "out.mp4")
        cmd = mock_run.call_args[0][0]
        assert "-f" in cmd
        assert "concat" in cmd

    def test_command_uses_stream_copy(self):
        with patch("subprocess.run", return_value=_ok()) as mock_run:
            stitch_clips(["a.mp4"], "out.mp4")
        cmd = mock_run.call_args[0][0]
        assert "-c" in cmd
        assert "copy" in cmd

    def test_output_path_in_command(self):
        with patch("subprocess.run", return_value=_ok()) as mock_run:
            stitch_clips(["a.mp4"], "final.mp4")
        cmd = mock_run.call_args[0][0]
        assert "final.mp4" in cmd

    def test_concat_list_file_deleted_on_success(self):
        # The temp concat list file must be cleaned up after a successful run.
        created_files = []
        original_unlink = os.unlink

        def tracking_unlink(path):
            created_files.append(path)
            original_unlink(path)

        with patch("subprocess.run", return_value=_ok()):
            with patch("os.unlink", side_effect=tracking_unlink):
                stitch_clips(["a.mp4"], "out.mp4")

        assert len(created_files) == 1
        assert not os.path.exists(created_files[0])

    def test_concat_list_file_deleted_on_failure(self):
        # Temp file must be cleaned up even when ffmpeg fails.
        created_files = []
        original_unlink = os.unlink

        def tracking_unlink(path):
            created_files.append(path)
            original_unlink(path)

        with patch("subprocess.run", return_value=_fail()):
            with patch("os.unlink", side_effect=tracking_unlink):
                with pytest.raises(RuntimeError):
                    stitch_clips(["a.mp4"], "out.mp4")

        assert len(created_files) == 1


# ---------------------------------------------------------------------------
# extract_clip
# ---------------------------------------------------------------------------

class TestExtractClip:
    def test_ffmpeg_failure_raises_runtime_error(self):
        with patch("subprocess.run", return_value=_fail("bad input")):
            with pytest.raises(RuntimeError, match="ffmpeg failed"):
                extract_clip("in.mp4", 0, 10, "out.mp4")

    def test_success_calls_subprocess(self):
        with patch("subprocess.run", return_value=_ok()) as mock_run:
            extract_clip("in.mp4", 5, 30, "out.mp4")
        mock_run.assert_called_once()

    def test_start_and_end_times_in_command(self):
        with patch("subprocess.run", return_value=_ok()) as mock_run:
            extract_clip("in.mp4", 15, 45, "out.mp4")
        cmd = mock_run.call_args[0][0]
        assert "15" in cmd and "45" in cmd

    def test_input_and_output_paths_in_command(self):
        with patch("subprocess.run", return_value=_ok()) as mock_run:
            extract_clip("source.mkv", 0, 10, "clip.mp4")
        cmd = mock_run.call_args[0][0]
        assert "source.mkv" in cmd
        assert "clip.mp4" in cmd

    def test_uses_stream_copy(self):
        with patch("subprocess.run", return_value=_ok()) as mock_run:
            extract_clip("in.mp4", 0, 10, "out.mp4")
        cmd = mock_run.call_args[0][0]
        assert "-c" in cmd
        assert "copy" in cmd

    def test_overwrite_flag_present(self):
        with patch("subprocess.run", return_value=_ok()) as mock_run:
            extract_clip("in.mp4", 0, 10, "out.mp4")
        cmd = mock_run.call_args[0][0]
        assert "-y" in cmd


# ---------------------------------------------------------------------------
# create_clips
# ---------------------------------------------------------------------------

class TestCreateClips:
    def test_empty_pairs_returns_empty_list(self):
        with patch("subprocess.run", return_value=_ok()):
            with tempfile.TemporaryDirectory() as tmpdir:
                result = create_clips("in.mp4", [], tmpdir)
        assert result == []

    def test_returns_correct_number_of_paths(self):
        with patch("subprocess.run", return_value=_ok()):
            with tempfile.TemporaryDirectory() as tmpdir:
                result = create_clips("in.mp4", [(0, 10), (20, 30), (40, 50)], tmpdir)
        assert len(result) == 3

    def test_clip_paths_are_sequentially_numbered(self):
        with patch("subprocess.run", return_value=_ok()):
            with tempfile.TemporaryDirectory() as tmpdir:
                result = create_clips("in.mp4", [(0, 10), (20, 30)], tmpdir)
        assert os.path.basename(result[0]) == "clip_001.mp4"
        assert os.path.basename(result[1]) == "clip_002.mp4"

    def test_creates_output_directory(self):
        with patch("subprocess.run", return_value=_ok()):
            with tempfile.TemporaryDirectory() as tmpdir:
                new_dir = os.path.join(tmpdir, "clips")
                create_clips("in.mp4", [(0, 10)], new_dir)
                assert os.path.isdir(new_dir)

    def test_clips_in_correct_output_directory(self):
        with patch("subprocess.run", return_value=_ok()):
            with tempfile.TemporaryDirectory() as tmpdir:
                result = create_clips("in.mp4", [(0, 10)], tmpdir)
        assert os.path.dirname(result[0]) == tmpdir


# ---------------------------------------------------------------------------
# Python fallback (no ffmpeg)
# ---------------------------------------------------------------------------

class TestPythonFallback:
    """Tests for the OpenCV fallback used when ffmpeg is not available."""

    @pytest.fixture(autouse=True)
    def no_ffmpeg(self, monkeypatch):
        import video_clipper
        monkeypatch.setattr(video_clipper, "_use_python_clipper", lambda: True)

    def _make_fake_cap(self, width=320, height=240, fps=30.0, frames=10):
        """Return a mock cv2.VideoCapture that yields `frames` black frames."""
        frame = MagicMock()
        reads = [(True, frame)] * frames + [(False, None)]
        cap = MagicMock()
        cap.get.side_effect = lambda prop: {
            0: fps,    # CAP_PROP_FPS
            3: width,  # CAP_PROP_FRAME_WIDTH
            4: height, # CAP_PROP_FRAME_HEIGHT
        }.get(prop, 0)
        cap.read.side_effect = reads
        return cap

    def test_extract_clip_uses_cv2_not_subprocess(self):
        mock_cap = self._make_fake_cap(frames=5)
        mock_writer = MagicMock()
        with patch("subprocess.run") as mock_sub, \
             patch("cv2.VideoCapture", return_value=mock_cap), \
             patch("cv2.VideoWriter", return_value=mock_writer), \
             patch("cv2.VideoWriter_fourcc", return_value=0):
            extract_clip("in.mp4", 0, 5, "out.mp4")
        mock_sub.assert_not_called()
        mock_writer.write.assert_called()

    def test_stitch_clips_uses_cv2_not_subprocess(self):
        # Each VideoCapture() call gets a fresh mock so reads don't run dry.
        mock_writer = MagicMock()
        with patch("subprocess.run") as mock_sub, \
             patch("cv2.VideoCapture", side_effect=lambda _: self._make_fake_cap(frames=3)), \
             patch("cv2.VideoWriter", return_value=mock_writer), \
             patch("cv2.VideoWriter_fourcc", return_value=0):
            stitch_clips(["a.mp4", "b.mp4"], "out.mp4")
        mock_sub.assert_not_called()
        mock_writer.release.assert_called()
