"""
test_main_ocr_pipeline.py — End-to-end integration tests for the OCR pipeline.

Tests call main.run() without --chat-log arguments, exercising the full
frame-extraction → OCR → analysis → clip → stitch pipeline.

Because OCR timestamps vary by ±1–2 s from the true event time, exact chapter
content is not asserted.  Instead tests verify:
  - The correct number of clip files is produced.
  - final_output.mp4 exists.
  - The chapters file has the expected number of entries.

Note: Cases 3 and 5 have OCR-undetectable events (see test_ag7_clips.py for
details), so their expected clip counts reflect what the OCR pipeline can
achieve rather than the log-file ground truth.

All tests are @pytest.mark.integration and require:
  - caseN.mkv in resources/
  - tesseract-ocr in PATH

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


def _make_args(video, output):
    return types.SimpleNamespace(
        video=video,
        output=output,
        chat_region=[0.0, 0.35, 0.15, 1.0],
        threads=1,
        ram_cap_gb=2,
        verbose=False,
        chat_logs=None,
        t0=None,
        chapters_dir=None,
        run_without_ffmpeg=True,
        force_ocr=False,
    )


def _run_ocr_pipeline(n: int, tmp_path_factory):
    """Run the OCR pipeline for case N, return (chapters_content, output_dir, clip_files)."""
    v = _video(n)
    if not os.path.exists(v):
        pytest.skip(f"Video not found: {v}")
    if shutil.which("tesseract") is None:
        pytest.skip("tesseract not found in PATH")

    out = tmp_path_factory.mktemp(f"case{n}_ocr")
    args = _make_args(v, str(out))
    chapters = main_module.run(args)
    clips = sorted(f for f in os.listdir(out) if f.startswith("clip_") and f.endswith(".mp4"))
    return chapters, out, clips


# ---------------------------------------------------------------------------
# Case 1 — Clear CD to WF
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def case1_ocr(tmp_path_factory):
    return _run_ocr_pipeline(1, tmp_path_factory)


@pytest.mark.integration
class TestCase1OcrMain:
    """case1.mkv: CD at 4 s, WF at 38 s → 1 clip."""

    _CLIP_COUNT = 1

    def test_clip_count(self, case1_ocr):
        _, _, clips = case1_ocr
        assert len(clips) == self._CLIP_COUNT, f"Expected {self._CLIP_COUNT} clip(s), got {clips}"

    def test_final_output_exists(self, case1_ocr):
        _, out, _ = case1_ocr
        assert os.path.exists(os.path.join(out, "final_output.mp4"))

    def test_chapter_count(self, case1_ocr):
        chapters, _, _ = case1_ocr
        assert chapters is not None
        lines = [l for l in chapters.splitlines() if l.strip()]
        assert len(lines) == self._CLIP_COUNT, f"Expected {self._CLIP_COUNT} chapter(s), got {lines}"


# ---------------------------------------------------------------------------
# Case 2 — Clear CD to GF
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def case2_ocr(tmp_path_factory):
    return _run_ocr_pipeline(2, tmp_path_factory)


@pytest.mark.integration
class TestCase2OcrMain:
    """case2.mkv: CD at 5 s, gf at 35 s (counts as WF) → 1 clip."""

    _CLIP_COUNT = 1

    def test_clip_count(self, case2_ocr):
        _, _, clips = case2_ocr
        assert len(clips) == self._CLIP_COUNT, f"Expected {self._CLIP_COUNT} clip(s), got {clips}"

    def test_final_output_exists(self, case2_ocr):
        _, out, _ = case2_ocr
        assert os.path.exists(os.path.join(out, "final_output.mp4"))

    def test_chapter_count(self, case2_ocr):
        chapters, _, _ = case2_ocr
        assert chapters is not None
        lines = [l for l in chapters.splitlines() if l.strip()]
        assert len(lines) == self._CLIP_COUNT, f"Expected {self._CLIP_COUNT} chapter(s), got {lines}"


# ---------------------------------------------------------------------------
# Case 3 — CD to WF to CD to GF
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def case3_ocr(tmp_path_factory):
    return _run_ocr_pipeline(3, tmp_path_factory)


@pytest.mark.integration
class TestCase3OcrMain:
    """case3.mkv: OCR detects 2 CDs but only 1 WF → 1 clip.
    The second WF (GF command at 95 s) is undetectable via OCR because its
    chat line carries a timestamp already recorded by the first WF event."""

    _CLIP_COUNT = 1

    def test_clip_count(self, case3_ocr):
        _, _, clips = case3_ocr
        assert len(clips) == self._CLIP_COUNT, f"Expected {self._CLIP_COUNT} clip(s), got {clips}"

    def test_final_output_exists(self, case3_ocr):
        _, out, _ = case3_ocr
        assert os.path.exists(os.path.join(out, "final_output.mp4"))

    def test_chapter_count(self, case3_ocr):
        chapters, _, _ = case3_ocr
        assert chapters is not None
        lines = [l for l in chapters.splitlines() if l.strip()]
        assert len(lines) == self._CLIP_COUNT, f"Expected {self._CLIP_COUNT} chapter(s), got {lines}"


# ---------------------------------------------------------------------------
# Case 4 — WF to CD to WF (initial WF before recording)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def case4_ocr(tmp_path_factory):
    return _run_ocr_pipeline(4, tmp_path_factory)


@pytest.mark.integration
class TestCase4OcrMain:
    """case4.mkv: pre-recording WF excluded by OCR dedup; CD at 9, WF at 33 → 1 clip."""

    _CLIP_COUNT = 1

    def test_clip_count(self, case4_ocr):
        _, _, clips = case4_ocr
        assert len(clips) == self._CLIP_COUNT, f"Expected {self._CLIP_COUNT} clip(s), got {clips}"

    def test_final_output_exists(self, case4_ocr):
        _, out, _ = case4_ocr
        assert os.path.exists(os.path.join(out, "final_output.mp4"))

    def test_chapter_count(self, case4_ocr):
        chapters, _, _ = case4_ocr
        assert chapters is not None
        lines = [l for l in chapters.splitlines() if l.strip()]
        assert len(lines) == self._CLIP_COUNT, f"Expected {self._CLIP_COUNT} chapter(s), got {lines}"


# ---------------------------------------------------------------------------
# Case 5 — Multiple CD / WF / GF Variations
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def case5_ocr(tmp_path_factory):
    return _run_ocr_pipeline(5, tmp_path_factory)


@pytest.mark.integration
class TestCase5OcrMain:
    """case5.mkv: OCR detects 4 of 8 CD variants and 4 of 7 WF variants → 1 clip."""

    _CLIP_COUNT = 1

    def test_clip_count(self, case5_ocr):
        _, _, clips = case5_ocr
        assert len(clips) == self._CLIP_COUNT, f"Expected {self._CLIP_COUNT} clip(s), got {clips}"

    def test_final_output_exists(self, case5_ocr):
        _, out, _ = case5_ocr
        assert os.path.exists(os.path.join(out, "final_output.mp4"))

    def test_chapter_count(self, case5_ocr):
        chapters, _, _ = case5_ocr
        assert chapters is not None
        lines = [l for l in chapters.splitlines() if l.strip()]
        assert len(lines) == self._CLIP_COUNT, f"Expected {self._CLIP_COUNT} chapter(s), got {lines}"


# ---------------------------------------------------------------------------
# Case 6 — CD to WF Surrounded by Random Text
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def case6_ocr(tmp_path_factory):
    return _run_ocr_pipeline(6, tmp_path_factory)


@pytest.mark.integration
class TestCase6OcrMain:
    """case6.mkv: CDs at 22, 73 s; WFs at 109, 113 s; embedded wf/WF ignored → 1 clip."""

    _CLIP_COUNT = 1

    def test_clip_count(self, case6_ocr):
        _, _, clips = case6_ocr
        assert len(clips) == self._CLIP_COUNT, f"Expected {self._CLIP_COUNT} clip(s), got {clips}"

    def test_final_output_exists(self, case6_ocr):
        _, out, _ = case6_ocr
        assert os.path.exists(os.path.join(out, "final_output.mp4"))

    def test_chapter_count(self, case6_ocr):
        chapters, _, _ = case6_ocr
        assert chapters is not None
        lines = [l for l in chapters.splitlines() if l.strip()]
        assert len(lines) == self._CLIP_COUNT, f"Expected {self._CLIP_COUNT} chapter(s), got {lines}"
