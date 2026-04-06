"""
test_0329_ocr.py — OCR pipeline comparison for the 2026-03-29 session.

Runs the full OCR pipeline (no chat log) on resources/2026-03-29 11-59-41.mkv
using a parallel worker pool and checks that the detected clips match the
log-file ground truth from test_0329_session.py.

Resource limits (enforced in the parallel runner):
    - max_workers=8   (≤8 threads)
    - batch_size=64   (≤64 frames in memory at once ≈ 50 MB per batch)

Memory budget: 64 frames × ~0.8 MB/frame (cropped+preprocessed PIL) + Tesseract
subprocess overhead × 8 workers ≈ well under 1 GB.  The full-frame arrays are
released immediately after cropping, so the large raw frame buffer is never held
for more than one frame at a time.

@pytest.mark.integration — skip with: pytest -m "not integration"
"""

import os
import sys

import cv2
import pytest
import pytesseract
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ocr_processor import crop_chat_region, preprocess_for_ocr
from chat_analyzer import analyze_frames, pair_cd_wf

# ---------------------------------------------------------------------------
# Paths and ground-truth from the log-file pipeline
# ---------------------------------------------------------------------------

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_VIDEO_PATH = os.path.join(_PROJECT_ROOT, "resources", "2026-03-29 11-59-41.mkv")

# Ground truth from test_0329_session.py (log-file pipeline)
_EXPECTED_PAIRS = [
    (1320, 1948),
    (2295, 2563),
    (3151, 3612),
    (3915, 4363),
    (4781, 5335),
    (5632, 6180),
    (6636, 6799),
    (7211, 7347),
]

# OCR is noisier than log parsing; allow this many seconds of drift per boundary.
_TOLERANCE = 60


# ---------------------------------------------------------------------------
# Parallel OCR runner
# ---------------------------------------------------------------------------

def _ocr_frame(args):
    """OCR one preprocessed PIL image; return the text string."""
    pil_img = args
    return pytesseract.image_to_string(pil_img, config="--psm 6 --oem 3")


def _run_ocr_pipeline_parallel(
    video_path: str,
    max_workers: int = 8,
    batch_size: int = 64,
) -> list:
    """
    Run OCR on every frame (1 fps) of video_path using a thread pool.

    Frames are extracted sequentially with a single VideoCapture pass (seeking
    one frame per second), immediately cropped to the chat region and
    preprocessed, then submitted to the thread pool in batches.  The full BGR
    frame is deleted right after cropping so only the small preprocessed PIL
    image (≈0.8 MB) is held per pending OCR job.

    Peak memory ≈ batch_size × 0.8 MB (PIL images) + 8 Tesseract subprocesses.
    With batch_size=64 that is ~50 MB, well within the 10 GB limit.

    Args:
        video_path:  Path to the video file.
        max_workers: Maximum OCR worker threads (≤8).
        batch_size:  Frames per batch (controls peak memory).

    Returns:
        List of (second, ocr_text) tuples in chronological order.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = int(total_frames / fps) + 1

    frame_texts = []
    batch_secs = []
    batch_pils = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for second in range(duration):
            frame_num = int(second * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                continue
            region = crop_chat_region(frame)
            pil_img = preprocess_for_ocr(region)
            del frame  # free the large BGR array immediately

            batch_secs.append(second)
            batch_pils.append(pil_img)

            if len(batch_pils) >= batch_size:
                texts = list(pool.map(_ocr_frame, batch_pils))
                frame_texts.extend(zip(batch_secs, texts))
                batch_secs = []
                batch_pils = []

        if batch_pils:
            texts = list(pool.map(_ocr_frame, batch_pils))
            frame_texts.extend(zip(batch_secs, texts))

    cap.release()
    return frame_texts  # chronological order (seconds increase monotonically)


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestOcrVsLogPipeline:
    """
    Verify the OCR pipeline (no log file) on the 0329 video produces clips
    that match the log-file ground truth within ±60 s tolerance.
    """

    @pytest.fixture(scope="class")
    def ocr_pairs(self):
        if not os.path.exists(_VIDEO_PATH):
            pytest.skip(f"Video not found: {_VIDEO_PATH}")
        frame_texts = _run_ocr_pipeline_parallel(
            _VIDEO_PATH, max_workers=8, batch_size=64
        )
        cd_times, wf_times = analyze_frames(frame_texts)
        pairs = pair_cd_wf(cd_times, wf_times)
        return pairs

    # ------------------------------------------------------------------

    def test_detects_eight_clips(self, ocr_pairs):
        assert len(ocr_pairs) == 8, (
            f"Expected 8 clips, got {len(ocr_pairs)}: {ocr_pairs}"
        )

    def test_clip_01(self, ocr_pairs):
        """CD ~1320 → WF ~1948"""
        cd_exp, wf_exp = _EXPECTED_PAIRS[0]
        assert ocr_pairs, "No pairs produced"
        cd_got, wf_got = ocr_pairs[0]
        assert abs(cd_got - cd_exp) <= _TOLERANCE, (
            f"Clip 1 CD: got {cd_got}, expected ~{cd_exp} (±{_TOLERANCE})"
        )
        assert abs(wf_got - wf_exp) <= _TOLERANCE, (
            f"Clip 1 WF: got {wf_got}, expected ~{wf_exp} (±{_TOLERANCE})"
        )

    def test_clip_02(self, ocr_pairs):
        """CD ~2295 → WF ~2563"""
        cd_exp, wf_exp = _EXPECTED_PAIRS[1]
        assert len(ocr_pairs) >= 2
        cd_got, wf_got = ocr_pairs[1]
        assert abs(cd_got - cd_exp) <= _TOLERANCE, (
            f"Clip 2 CD: got {cd_got}, expected ~{cd_exp} (±{_TOLERANCE})"
        )
        assert abs(wf_got - wf_exp) <= _TOLERANCE, (
            f"Clip 2 WF: got {wf_got}, expected ~{wf_exp} (±{_TOLERANCE})"
        )

    # TODO: may actually be fine
    def test_clip_03(self, ocr_pairs):
        """CD ~3151 → WF ~3612"""
        cd_exp, wf_exp = _EXPECTED_PAIRS[2]
        assert len(ocr_pairs) >= 3
        cd_got, wf_got = ocr_pairs[2]
        assert abs(cd_got - cd_exp) <= _TOLERANCE, (
            f"Clip 3 CD: got {cd_got}, expected ~{cd_exp} (±{_TOLERANCE})"
        )
        assert abs(wf_got - wf_exp) <= _TOLERANCE, (
            f"Clip 3 WF: got {wf_got}, expected ~{wf_exp} (±{_TOLERANCE})"
        )

    def test_clip_04(self, ocr_pairs):
        """CD ~3915 → WF ~4363"""
        cd_exp, wf_exp = _EXPECTED_PAIRS[3]
        assert len(ocr_pairs) >= 4
        cd_got, wf_got = ocr_pairs[3]
        assert abs(cd_got - cd_exp) <= _TOLERANCE, (
            f"Clip 4 CD: got {cd_got}, expected ~{cd_exp} (±{_TOLERANCE})"
        )
        assert abs(wf_got - wf_exp) <= _TOLERANCE, (
            f"Clip 4 WF: got {wf_got}, expected ~{wf_exp} (±{_TOLERANCE})"
        )

    # TODO: may actually be fine
    def test_clip_05(self, ocr_pairs):
        """CD ~4781 → WF ~5335"""
        cd_exp, wf_exp = _EXPECTED_PAIRS[4]
        assert len(ocr_pairs) >= 5
        cd_got, wf_got = ocr_pairs[4]
        assert abs(cd_got - cd_exp) <= _TOLERANCE, (
            f"Clip 5 CD: got {cd_got}, expected ~{cd_exp} (±{_TOLERANCE})"
        )
        assert abs(wf_got - wf_exp) <= _TOLERANCE, (
            f"Clip 5 WF: got {wf_got}, expected ~{wf_exp} (±{_TOLERANCE})"
        )

    def test_clip_06(self, ocr_pairs):
        """CD ~5632 → WF ~6180"""
        cd_exp, wf_exp = _EXPECTED_PAIRS[5]
        assert len(ocr_pairs) >= 6
        cd_got, wf_got = ocr_pairs[5]
        assert abs(cd_got - cd_exp) <= _TOLERANCE, (
            f"Clip 6 CD: got {cd_got}, expected ~{cd_exp} (±{_TOLERANCE})"
        )
        assert abs(wf_got - wf_exp) <= _TOLERANCE, (
            f"Clip 6 WF: got {wf_got}, expected ~{wf_exp} (±{_TOLERANCE})"
        )

    def test_clip_07(self, ocr_pairs):
        """CD ~6636 → WF ~6799"""
        cd_exp, wf_exp = _EXPECTED_PAIRS[6]
        assert len(ocr_pairs) >= 7
        cd_got, wf_got = ocr_pairs[6]
        assert abs(cd_got - cd_exp) <= _TOLERANCE, (
            f"Clip 7 CD: got {cd_got}, expected ~{cd_exp} (±{_TOLERANCE})"
        )
        assert abs(wf_got - wf_exp) <= _TOLERANCE, (
            f"Clip 7 WF: got {wf_got}, expected ~{wf_exp} (±{_TOLERANCE})"
        )

    def test_clip_08(self, ocr_pairs):
        """CD ~7211 → WF ~7347"""
        cd_exp, wf_exp = _EXPECTED_PAIRS[7]
        assert len(ocr_pairs) >= 8
        cd_got, wf_got = ocr_pairs[7]
        assert abs(cd_got - cd_exp) <= _TOLERANCE, (
            f"Clip 8 CD: got {cd_got}, expected ~{cd_exp} (±{_TOLERANCE})"
        )
        assert abs(wf_got - wf_exp) <= _TOLERANCE, (
            f"Clip 8 WF: got {wf_got}, expected ~{wf_exp} (±{_TOLERANCE})"
        )
