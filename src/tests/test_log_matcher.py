"""
Tests for log_matcher.py — pure-logic functions and detect_t0 integration.
"""

import os
import sys
import tempfile
import threading
import unittest.mock as mock

import numpy as np

from log_matcher import _normalize, _find_best_t0, _load_unique_messages, detect_t0


# ---------------------------------------------------------------------------
# _normalize
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_lowercases(self):
        assert _normalize("Hello World") == "hello world"

    def test_removes_punctuation(self):
        assert _normalize("we'll be just a minute!") == "we ll be just a minute"

    def test_collapses_whitespace(self):
        assert _normalize("too   many   spaces") == "too many spaces"

    def test_empty_string(self):
        assert _normalize("") == ""

    def test_only_punctuation(self):
        assert _normalize("!!!") == ""

    def test_mixed(self):
        assert _normalize("CD CD CD") == "cd cd cd"


# ---------------------------------------------------------------------------
# _find_best_t0
# ---------------------------------------------------------------------------

class TestFindBestT0:
    def test_single_candidate(self):
        assert _find_best_t0([3600]) == 3600

    def test_clear_majority(self):
        # Ten candidates at 3600, one outlier
        candidates = [3600] * 10 + [5000]
        assert _find_best_t0(candidates) == 3600

    def test_cluster_within_2s(self):
        # Candidates at 3600, 3601, 3602 all count as the same cluster
        candidates = [3600, 3601, 3602, 3601, 3600]
        result = _find_best_t0(candidates)
        assert abs(result - 3601) <= 2

    def test_prefers_higher_on_tie(self):
        # Two equally sized clusters; the higher one should win
        candidates = [100, 100, 200, 200]
        assert _find_best_t0(candidates) == 200

    def test_real_world_spread(self):
        # Most candidates cluster near true t0 (3725), a few stragglers below
        candidates = [3725, 3724, 3725, 3726, 3700, 3680, 3720, 3725, 3723]
        result = _find_best_t0(candidates)
        assert abs(result - 3725) <= 2


# ---------------------------------------------------------------------------
# _load_unique_messages
# ---------------------------------------------------------------------------

def _write_log(content: str, encoding: str = "utf-16") -> str:
    fd, path = tempfile.mkstemp(suffix=".txt")
    with os.fdopen(fd, "wb") as f:
        f.write(content.encode(encoding))
    return path


_LOG_A = """\
\ufeff
 Session started: 2026.03.25 01:00:00

\ufeff[ 2026.03.25 01:07:49 ] Alice > we are ready to fight
\ufeff[ 2026.03.25 01:09:00 ] Bob > short
\ufeff[ 2026.03.25 01:10:00 ] Alice > duplicate message here
"""

_LOG_B = """\
\ufeff
 Session started: 2026.03.25 01:00:00

\ufeff[ 2026.03.25 01:11:00 ] Carol > unique message in log b
\ufeff[ 2026.03.25 01:12:00 ] Alice > duplicate message here
"""


class TestLoadUniqueMessages:
    def test_returns_unique_only(self):
        path_a = _write_log(_LOG_A)
        path_b = _write_log(_LOG_B)
        try:
            result = _load_unique_messages([path_a, path_b])
            # "duplicate message here" appears twice → excluded
            for norm_msg in result:
                assert "duplicate" not in norm_msg
        finally:
            os.unlink(path_a)
            os.unlink(path_b)

    def test_excludes_short_messages(self):
        path = _write_log(_LOG_A)
        try:
            result = _load_unique_messages([path])
            # "short" is 5 chars after normalisation → excluded
            for norm_msg in result:
                assert norm_msg != "short"
        finally:
            os.unlink(path)

    def test_includes_long_unique(self):
        path_a = _write_log(_LOG_A)
        path_b = _write_log(_LOG_B)
        try:
            result = _load_unique_messages([path_a, path_b])
            assert "we are ready to fight" in result
            assert "unique message in log b" in result
        finally:
            os.unlink(path_a)
            os.unlink(path_b)

    def test_game_seconds_correct(self):
        path = _write_log(_LOG_A)
        try:
            result = _load_unique_messages([path])
            # 01:07:49 → 1*3600 + 7*60 + 49 = 4069
            assert result["we are ready to fight"] == 4069
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# detect_t0
# ---------------------------------------------------------------------------

def _make_cap_mock(fps=1.0, total_frames=120, frame_ok=True):
    """Build a mock cv2.VideoCapture that yields blank frames."""
    cap = mock.MagicMock()
    cap.isOpened.return_value = True
    cap.get.side_effect = lambda prop: fps if prop == 5 else total_frames  # CAP_PROP_FPS=5, FRAME_COUNT=7
    blank = np.zeros((100, 100, 3), dtype=np.uint8)
    cap.read.return_value = (frame_ok, blank)
    return cap


# Log content whose single unique message has game_sec = 1*3600+7*60+49 = 4069
_DETECT_LOG = """\
\ufeff
 Session started: 2026.03.25 01:00:00

\ufeff[ 2026.03.25 01:07:49 ] Alice > we are ready to fight
"""


class TestDetectT0:
    """detect_t0 tests — cv2, crop_chat_region and run_ocr_on_region are mocked."""

    def _write_detect_log(self):
        return _write_log(_DETECT_LOG)

    def _run(self, log_path, ocr_text, *, fps=1.0, total_frames=120,
             sample_interval=30, cancel_event=None, progress_callback=None,
             frame_ok=True):
        cap_mock = _make_cap_mock(fps=fps, total_frames=total_frames, frame_ok=frame_ok)
        blank = np.zeros((100, 100, 3), dtype=np.uint8)
        with mock.patch("log_matcher.cv2.VideoCapture", return_value=cap_mock), \
             mock.patch("log_matcher.crop_chat_region", return_value=blank), \
             mock.patch("log_matcher.run_ocr_on_region", return_value=ocr_text):
            return detect_t0(
                [log_path],
                "fake.mp4",
                sample_interval=sample_interval,
                cancel_event=cancel_event,
                progress_callback=progress_callback,
            )

    # -- happy path ----------------------------------------------------------

    def test_returns_correct_t0(self):
        # Message at game_sec=4069, first visible at video_sec=30 → t0=4069-30=4039.
        # OCR returns the match only on the 2nd sample (video_sec=30), so that
        # candidate (4039) is the only one and must be returned.
        path = self._write_detect_log()
        call_num = [0]

        def ocr_side_effect(_region):
            call_num[0] += 1
            return "we are ready to fight" if call_num[0] == 2 else "unrelated text"

        cap_mock = _make_cap_mock(fps=1.0, total_frames=120)
        blank = np.zeros((100, 100, 3), dtype=np.uint8)
        try:
            with mock.patch("log_matcher.cv2.VideoCapture", return_value=cap_mock), \
                 mock.patch("log_matcher.crop_chat_region", return_value=blank), \
                 mock.patch("log_matcher.run_ocr_on_region", side_effect=ocr_side_effect):
                t0 = detect_t0([path], "fake.mp4", sample_interval=30)
            assert t0 == 4039
        finally:
            os.unlink(path)

    # -- multiple match frames: keeps highest candidate per message ----------

    def test_keeps_highest_candidate_per_message(self):
        # Message appears at video_sec=0 and video_sec=30 (still on screen).
        # Best (highest) candidate = 4069 - 0 = 4069.
        # Candidate from sec 30 = 4039 (lower bound, should be discarded).
        path = self._write_detect_log()
        try:
            # sample_interval=30, total_frames=120 → samples at 0,30,60,90
            t0 = self._run(path, "we are ready to fight",
                           fps=1.0, total_frames=120, sample_interval=30)
            # The returned t0 must be the highest candidate (4069), not a lower one
            assert t0 == 4069
        finally:
            os.unlink(path)

    # -- no matches ----------------------------------------------------------

    def test_raises_when_no_match(self):
        path = self._write_detect_log()
        try:
            try:
                self._run(path, "completely unrelated ocr output xyz")
                assert False, "Expected RuntimeError"
            except RuntimeError as exc:
                assert "__cancelled__" not in str(exc)
        finally:
            os.unlink(path)

    # -- video cannot be opened ----------------------------------------------

    def test_raises_on_bad_video(self):
        path = self._write_detect_log()
        try:
            cap_mock = mock.MagicMock()
            cap_mock.isOpened.return_value = False
            with mock.patch("log_matcher.cv2.VideoCapture", return_value=cap_mock):
                try:
                    detect_t0([path], "nonexistent.mp4")
                    assert False, "Expected ValueError"
                except ValueError:
                    pass
        finally:
            os.unlink(path)

    # -- cancel event --------------------------------------------------------

    def test_cancel_raises_runtime_error(self):
        path = self._write_detect_log()
        cancel = threading.Event()
        cancel.set()
        try:
            try:
                self._run(path, "we are ready to fight", cancel_event=cancel)
                assert False, "Expected RuntimeError"
            except RuntimeError as exc:
                assert "__cancelled__" in str(exc)
        finally:
            os.unlink(path)

    # -- progress callback ---------------------------------------------------

    def test_progress_callback_called(self):
        path = self._write_detect_log()
        calls = []
        try:
            self._run(path, "we are ready to fight",
                      fps=1.0, total_frames=60, sample_interval=30,
                      progress_callback=lambda cur, tot: calls.append((cur, tot)))
            # Two samples (sec 0 and sec 30) → two callback invocations
            assert len(calls) == 2
            assert calls[-1] == (2, 2)
        finally:
            os.unlink(path)

    # -- midnight wrap -------------------------------------------------------

    def test_midnight_wrap(self):
        # A log message at 00:01:00 (game_sec=60) that appears at video_sec=3700
        # gives t0_candidate = 60 - 3700 = -3640, which is < -3600
        # → should be wrapped: -3640 + 86400 = 82760
        _midnight_log = """\
\ufeff
 Session started: 2026.03.25 00:00:00

\ufeff[ 2026.03.25 00:01:00 ] Alice > the midnight wrap test message
"""
        path = _write_log(_midnight_log)
        try:
            # fps=1, total_frames=7400 → duration=7400, sample at 3700 with interval=3700
            t0 = self._run(path, "the midnight wrap test message",
                           fps=1.0, total_frames=7400, sample_interval=3700)
            assert t0 == 82760
        finally:
            os.unlink(path)

    # -- unreadable frames are skipped --------------------------------------

    def test_unreadable_frames_skipped(self):
        # cap.read() returns (False, ...) → no candidates → RuntimeError
        path = self._write_detect_log()
        try:
            try:
                self._run(path, "we are ready to fight", frame_ok=False)
                assert False, "Expected RuntimeError"
            except RuntimeError:
                pass
        finally:
            os.unlink(path)

    # -- verbose output covers print branches (lines 114, 160, 197) ---------

    def test_verbose_mode(self, capsys):
        path = self._write_detect_log()
        cap_mock = _make_cap_mock(fps=1.0, total_frames=120)
        blank = np.zeros((100, 100, 3), dtype=np.uint8)
        try:
            with mock.patch("log_matcher.cv2.VideoCapture", return_value=cap_mock), \
                 mock.patch("log_matcher.crop_chat_region", return_value=blank), \
                 mock.patch("log_matcher.run_ocr_on_region", return_value="we are ready to fight"):
                detect_t0([path], "fake.mp4", sample_interval=30, verbose=True)
            out = capsys.readouterr().out
            assert "unique log message" in out
            assert "t0=" in out
        finally:
            os.unlink(path)
