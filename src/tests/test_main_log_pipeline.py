"""
test_main_log_pipeline.py — End-to-end integration tests for the log pipeline.

Tests call main.run() with --chat-log and --t0 arguments and verify the
complete output: final_output.mp4 and chapter timestamps.

Chapter timestamps are exact (log parsing is deterministic, unlike OCR).

All tests are @pytest.mark.integration and require:
  - caseN.mkv and caseN.txt in resources/
  - tesseract-ocr in PATH (checked by main._check_dependencies)

Skip all integration tests with: pytest -m "not integration"
"""

import os
import shutil
import types

import pytest

import main as main_module

_RES = os.path.join(os.path.dirname(__file__), "resources")


def _video(n: int) -> str:
    return os.path.join(_RES, f"case{n}.mkv")


def _log(n: int) -> str:
    return os.path.join(_RES, f"case{n}.txt")


def _make_args(video, output, chat_logs, t0):
    return types.SimpleNamespace(
        video=video,
        output=output,
        chat_region=[0.0, 0.35, 0.15, 1.0],
        threads=1,
        ram_cap_gb=2,
        verbose=False,
        chat_logs=chat_logs,
        t0=t0,
        chapters_dir=None,
        run_without_ffmpeg=True,
        force_ocr=False,
    )


def _run_log_pipeline(n: int, t0_str: str, tmp_path_factory):
    """Run the log pipeline for case N, return (chapters_content, output_dir)."""
    v = _video(n)
    l = _log(n)
    if not os.path.exists(v):
        pytest.skip(f"Video not found: {v}")
    if not os.path.exists(l):
        pytest.skip(f"Log not found: {l}")
    if shutil.which("tesseract") is None:
        pytest.skip("tesseract not found in PATH")

    out = tmp_path_factory.mktemp(f"case{n}_log")
    args = _make_args(v, str(out), [l], t0_str)
    chapters, _ = main_module.run(args)
    return chapters, out


# ---------------------------------------------------------------------------
# Case 1 — Clear CD to WF
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def case1_log(tmp_path_factory):
    return _run_log_pipeline(1, "17:43:55", tmp_path_factory)


@pytest.mark.integration
class TestCase1LogMain:
    """case1: CD at 4 s, WF at 38 s → 1 clip (4→38, 34 s)."""

    _CHAPTERS = "0:00 Match 1\n"

    def test_final_output_exists(self, case1_log):
        _, out = case1_log
        assert os.path.exists(os.path.join(out, "final_output.mp4"))

    def test_chapter_timestamps(self, case1_log):
        chapters, _ = case1_log
        assert chapters == self._CHAPTERS, f"Expected {self._CHAPTERS!r}, got {chapters!r}"


# ---------------------------------------------------------------------------
# Case 2 — Clear CD to GF
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def case2_log(tmp_path_factory):
    return _run_log_pipeline(2, "17:49:56", tmp_path_factory)


@pytest.mark.integration
class TestCase2LogMain:
    """case2: CD at 5 s, gf at 35 s (counts as WF) → 1 clip (5→35, 30 s)."""

    _CHAPTERS = "0:00 Match 1\n"

    def test_final_output_exists(self, case2_log):
        _, out = case2_log
        assert os.path.exists(os.path.join(out, "final_output.mp4"))

    def test_chapter_timestamps(self, case2_log):
        chapters, _ = case2_log
        assert chapters == self._CHAPTERS, f"Expected {self._CHAPTERS!r}, got {chapters!r}"


# ---------------------------------------------------------------------------
# Case 3 — CD to WF to CD to GF
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def case3_log(tmp_path_factory):
    return _run_log_pipeline(3, "17:52:22", tmp_path_factory)


@pytest.mark.integration
class TestCase3LogMain:
    """case3: 2 CDs (8, 59 s), 2 WFs (41, 95 s) → 2 clips (8→41=33 s, 59→95=36 s)."""

    # clip 1 duration = 41-8 = 33 s → cumulative at clip 2 = 33 s
    _CHAPTERS = "0:00 Match 1\n0:33 Match 2\n"

    def test_final_output_exists(self, case3_log):
        _, out = case3_log
        assert os.path.exists(os.path.join(out, "final_output.mp4"))

    def test_chapter_timestamps(self, case3_log):
        chapters, _ = case3_log
        assert chapters == self._CHAPTERS, f"Expected {self._CHAPTERS!r}, got {chapters!r}"


# ---------------------------------------------------------------------------
# Case 4 — WF to CD to WF (initial WF before recording)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def case4_log(tmp_path_factory):
    return _run_log_pipeline(4, "17:55:48", tmp_path_factory)


@pytest.mark.integration
class TestCase4LogMain:
    """case4: pre-recording WF excluded; CD at 9 s, WF at 33 s → 1 clip (9→33, 24 s)."""

    _CHAPTERS = "0:00 Match 1\n"

    def test_final_output_exists(self, case4_log):
        _, out = case4_log
        assert os.path.exists(os.path.join(out, "final_output.mp4"))

    def test_chapter_timestamps(self, case4_log):
        chapters, _ = case4_log
        assert chapters == self._CHAPTERS, f"Expected {self._CHAPTERS!r}, got {chapters!r}"


# ---------------------------------------------------------------------------
# Case 5 — Multiple CD / WF / GF Variations
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def case5_log(tmp_path_factory):
    return _run_log_pipeline(5, "17:58:04", tmp_path_factory)


@pytest.mark.integration
class TestCase5LogMain:
    """case5: 8 CD variants, 7 WF/GF variants → 1 clip (57→65, 8 s)."""

    _CHAPTERS = "0:00 Match 1\n"

    def test_final_output_exists(self, case5_log):
        _, out = case5_log
        assert os.path.exists(os.path.join(out, "final_output.mp4"))

    def test_chapter_timestamps(self, case5_log):
        chapters, _ = case5_log
        assert chapters == self._CHAPTERS, f"Expected {self._CHAPTERS!r}, got {chapters!r}"


# ---------------------------------------------------------------------------
# Case 6 — CD to WF Surrounded by Random Text
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def case6_log(tmp_path_factory):
    return _run_log_pipeline(6, "18:01:54", tmp_path_factory)


@pytest.mark.integration
class TestCase6LogMain:
    """case6: CDs at 22, 73 s; WFs at 109, 113 s → 1 clip (73→109, 36 s)."""

    _CHAPTERS = "0:00 Match 1\n"

    def test_final_output_exists(self, case6_log):
        _, out = case6_log
        assert os.path.exists(os.path.join(out, "final_output.mp4"))

    def test_chapter_timestamps(self, case6_log):
        chapters, _ = case6_log
        assert chapters == self._CHAPTERS, f"Expected {self._CHAPTERS!r}, got {chapters!r}"
