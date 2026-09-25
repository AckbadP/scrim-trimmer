"""
Tests for App._find_latest_video — the default-video-dir auto-selection scan.

Only the pure staticmethod is exercised here; it needs no Tk instance/display.
"""

import os
import time
from unittest.mock import patch

import pytest

import gui
from gui import App


def _touch(path, mtime_offset=0, content=b"x" * 10):
    with open(path, "wb") as f:
        f.write(content)
    now = time.time()
    os.utime(path, (now + mtime_offset, now + mtime_offset))


class TestFindLatestVideo:
    def test_returns_none_for_empty_dir(self, tmp_path):
        path, skipped = App._find_latest_video(str(tmp_path))
        assert path is None
        assert skipped is None

    def test_ignores_non_video_extensions(self, tmp_path):
        _touch(tmp_path / "notes.txt", mtime_offset=-100)
        video = tmp_path / "clip.mp4"
        _touch(video, mtime_offset=-100)
        with patch("gui.is_growing", return_value=False):
            path, skipped = App._find_latest_video(str(tmp_path))
        assert path == str(video)
        assert skipped is None

    def test_picks_newest_finished_video(self, tmp_path):
        older = tmp_path / "older.mkv"
        newer = tmp_path / "newer.mp4"
        _touch(older, mtime_offset=-100)
        _touch(newer, mtime_offset=-1)  # very fresh mtime
        with patch("gui.is_growing", return_value=False):
            path, skipped = App._find_latest_video(str(tmp_path))
        assert path == str(newer)
        assert skipped is None

    def test_a_fresh_but_stable_file_is_not_skipped(self, tmp_path):
        """
        Regression test: the old implementation rejected any candidate whose
        mtime was under ~10s old, regardless of whether it was still being
        written. A recorder can keep advancing a file's mtime (flush/finalize)
        long after the last frame was captured, so a fresh mtime alone must
        not disqualify it — only a genuinely growing size should.
        """
        video = tmp_path / "just_finished.mkv"
        _touch(video, mtime_offset=0)  # mtime is "now"
        with patch("gui.is_growing", return_value=False):
            path, skipped = App._find_latest_video(str(tmp_path))
        assert path == str(video)
        assert skipped is None

    def test_skips_growing_file_and_reports_it(self, tmp_path):
        growing = tmp_path / "recording.mkv"
        finished = tmp_path / "finished.mkv"
        _touch(growing, mtime_offset=-1)
        _touch(finished, mtime_offset=-100)

        def _is_growing(path):
            return path == str(growing)

        with patch("gui.is_growing", side_effect=_is_growing):
            path, skipped = App._find_latest_video(str(tmp_path))
        assert path == str(finished)
        assert skipped == "recording.mkv"

    def test_falls_back_to_newest_when_everything_looks_growing(self, tmp_path):
        video = tmp_path / "only.mp4"
        _touch(video, mtime_offset=-1)
        with patch("gui.is_growing", return_value=True):
            path, skipped = App._find_latest_video(str(tmp_path))
        assert path == str(video)
        assert skipped is None
