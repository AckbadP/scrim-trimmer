"""
log_matcher.py - Auto-detect t0 by matching chat log entries against video OCR.

t0 is the EVE game time (seconds since midnight UTC) at video second 0.
It allows log timestamps to be converted to video timestamps without the user
having to note the game clock manually.

Strategy
--------
1. Load all log entries; build a lookup of normalised message text → game_sec
   for messages that appear exactly once (unique messages = unambiguous anchors).
2. Sample one frame every `sample_interval` seconds and run OCR on the chat
   region.
3. For every unique log message found in the OCR output of frame V, record:
       t0_candidate = game_sec - V
   Because a message is still visible on screen seconds after it was sent,
   each candidate is a lower bound on the true t0.  The candidate produced by
   the sample frame closest to when the message first appeared is the tightest
   bound and will equal t0 (to within ±1 second).
4. Find the densest cluster among all candidates (majority vote, ±2 s window),
   preferring higher values within a tie — higher values are closer to the true
   t0.
"""

import re
import sys
import time
from typing import Dict, List, Optional

import cv2
import numpy as np

from chat_log_parser import read_chat_log
from ocr_processor import crop_chat_region, run_ocr_on_region


# Messages shorter than this after normalisation are skipped — too short to
# match reliably against noisy OCR text.
_MIN_MSG_LEN = 8


def _normalize(text: str) -> str:
    """Lowercase; keep only letters, digits, spaces; collapse whitespace."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9 ]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def _load_unique_messages(log_paths: List[str]) -> Dict[str, int]:
    """
    Return {normalised_message: game_seconds} for messages that appear
    exactly once across all log files.  Duplicates are excluded because they
    would produce conflicting t0 candidates.
    """
    all_entries: dict = {}
    for path in log_paths:
        for ts, player, msg in read_chat_log(path):
            key = (ts, player, msg)
            all_entries[key] = (ts, player, msg)

    msg_to_secs: dict = {}
    for ts, _player, msg in all_entries.values():
        norm = _normalize(msg)
        if len(norm) < _MIN_MSG_LEN:
            continue
        game_sec = ts.hour * 3600 + ts.minute * 60 + ts.second
        msg_to_secs.setdefault(norm, []).append(game_sec)

    return {msg: secs[0] for msg, secs in msg_to_secs.items() if len(secs) == 1}


def _find_best_t0(candidates: List[int]) -> int:
    """
    Return the t0 value with the densest cluster within ±2 seconds.
    Ties are broken by preferring the higher value (tighter lower bound).
    """
    best_t0 = candidates[0]
    best_score = 0
    for c in candidates:
        score = sum(1 for x in candidates if abs(x - c) <= 2)
        if score > best_score or (score == best_score and c > best_t0):
            best_score = score
            best_t0 = c
    return best_t0


def detect_t0(
    log_paths: List[str],
    video_path: str,
    chat_region: tuple = (0.0, 0.35, 0.15, 1.0),
    sample_interval: int = 30,
    verbose: bool = False,
    progress_callback=None,
    cancel_event=None,
) -> int:
    """
    Auto-detect t0 (EVE game seconds-since-midnight UTC at video second 0).

    Args:
        log_paths:       Paths to one or more EVE chat log files.
        video_path:      Path to the video file (.mp4 or .mkv).
        chat_region:     (x1, y1, x2, y2) as fractions of frame dimensions.
        sample_interval: OCR one frame every N seconds (default 30).
        verbose:         Print per-match detail.

    Returns:
        t0 as an integer (seconds since midnight UTC).

    Raises:
        RuntimeError if no log messages could be matched in the video.
    """
    unique_msgs = _load_unique_messages(log_paths)
    if verbose:
        print(f"  {len(unique_msgs)} unique log message(s) available for matching")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = int(total_frames / fps)

    # Best (highest) t0 candidate seen so far for each unique message.
    # A message that is still on screen at sample frame V gives candidate
    # game_sec - V, which is a lower bound on t0.  The first frame the message
    # appears in yields the highest (tightest) lower bound.  Later frames of
    # the same message give progressively lower (looser) bounds, which only
    # corrupt the density calculation.  Keeping only the highest candidate per
    # message prevents stale detections from forming false dense clusters.
    best_per_msg: Dict[str, int] = {}
    frames_sampled = 0

    sample_seconds = list(range(0, duration, sample_interval))
    total_samples = len(sample_seconds)
    start_time = time.monotonic()
    for sample_num, video_sec in enumerate(sample_seconds, start=1):
        if cancel_event is not None and cancel_event.is_set():
            cap.release()
            raise RuntimeError("__cancelled__")
        frame_number = int(video_sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if not ret:
            continue

        frames_sampled += 1
        region = crop_chat_region(frame, *chat_region)
        del frame
        ocr_text = run_ocr_on_region(region)
        norm_ocr = _normalize(ocr_text)

        for norm_msg, game_sec in unique_msgs.items():
            if norm_msg in norm_ocr:
                t0_candidate = game_sec - video_sec
                # Handle midnight wrap (log spans 23:xx → 00:xx UTC)
                if t0_candidate < -3600:
                    t0_candidate += 86400
                if verbose:
                    print(f"    [{video_sec:5d}s] '{norm_msg[:50]}' → t0={t0_candidate}")
                # Keep only the highest candidate for this message (first appearance).
                if norm_msg not in best_per_msg or t0_candidate > best_per_msg[norm_msg]:
                    best_per_msg[norm_msg] = t0_candidate

        if not verbose:
            elapsed = time.monotonic() - start_time
            pct = sample_num / total_samples
            if progress_callback is not None:
                progress_callback(sample_num, total_samples)
            else:
                filled = int(30 * pct)
                bar = "#" * filled + "-" * (30 - filled)
                eta_str = ""
                if sample_num > 1 and elapsed > 0:
                    remaining = (total_samples - sample_num) * elapsed / sample_num
                    m, s = divmod(int(remaining), 60)
                    eta_str = f"  ETA {m}:{s:02d}"
                sys.stdout.write(f"\r  [{bar}] {pct*100:5.1f}%  {sample_num}/{total_samples}{eta_str}  ")
                sys.stdout.flush()

    if not verbose and progress_callback is None:
        sys.stdout.write("\n")

    cap.release()

    if not best_per_msg:
        raise RuntimeError(
            f"Could not auto-detect t0: no chat log messages were found in "
            f"{frames_sampled} sampled frame(s).\n"
            "Adjust the chat region selection or provide --t0 manually."
        )

    candidates = list(best_per_msg.values())
    t0 = _find_best_t0(candidates)

    if verbose:
        print(
            f"  Detected t0={t0}  "
            f"({t0 // 3600:02d}:{(t0 % 3600) // 60:02d}:{t0 % 60:02d} UTC)  "
            f"from {len(candidates)} candidate(s) across {frames_sampled} frame(s)"
        )

    return t0
