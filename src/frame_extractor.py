"""
frame_extractor.py - Extract one frame per second from a video file.
"""

import os
import shutil
import subprocess
import time
from typing import Generator, Optional, Tuple

import cv2
import numpy as np


def extract_frames(
    video_path: str,
) -> Generator[Tuple[int, np.ndarray], None, None]:
    """
    Yield (second, frame) tuples, one per second of video.

    Args:
        video_path: Path to the video file (.mp4 or .mkv).

    Yields:
        (second, frame) where second is the integer timestamp (0, 1, 2, ...)
        and frame is a BGR numpy array.
    """
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0:
            raise ValueError(f"Invalid FPS ({fps}) in video: {video_path}")

        # Decode sequentially: advance frame-by-frame, grabbing only the frame
        # nearest to each whole second.  This avoids a costly keyframe seek on
        # every iteration (the old cap.set(CAP_PROP_POS_FRAMES, …) approach).
        frames_per_second = fps
        next_target = 0          # the next whole-second index we want to yield
        current_frame = 0        # how many frames we have read so far

        while current_frame < total_frames:
            target_frame = int(round(next_target * frames_per_second))
            if target_frame >= total_frames:
                break

            # Skip frames between the current position and the target by
            # grabbing (cheap: demux without decode) rather than seeking.
            while current_frame < target_frame:
                if not cap.grab():
                    return
                current_frame += 1

            ret, frame = cap.read()
            if not ret:
                break
            current_frame += 1

            yield next_target, frame
            next_target += 1
    finally:
        # Runs even if the generator is abandoned mid-iteration (GeneratorExit
        # from a `break`/cancel in the consumer, or any exception raised at
        # the `yield`), so the capture handle is never leaked.
        cap.release()


def probe_duration(video_path: str, timeout: float = 30.0) -> Optional[float]:
    """
    Return the container-reported duration in seconds via ffprobe, or None if
    ffprobe is unavailable, times out, or the file has no valid duration
    (e.g. a recording that is still being written and has no index yet).
    """
    if shutil.which("ffprobe") is None:
        return None
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=nw=1:nk=1",
        video_path,
    ]
    try:
        result = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0:
        return None
    try:
        value = float(result.stdout.strip())
    except ValueError:
        return None
    return value if value > 0 else None


def get_video_duration(video_path: str) -> float:
    """
    Return the duration of the video in seconds.

    Prefers ffprobe's container-reported duration; falls back to OpenCV's
    frame-count/fps estimate, which is only reliable for a fully-finalized
    file (a container still being written has no valid frame count/index).
    """
    probed = probe_duration(video_path)
    if probed is not None:
        return probed

    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return total_frames / fps if fps > 0 else 0.0
    finally:
        cap.release()


def check_source_video(video_path: str, grow_check_wait: float = 1.5) -> Optional[str]:
    """
    Sanity-check a source video before starting a long pipeline run.

    Returns a human-readable warning string if the file looks like it is
    still being actively written to (e.g. the game/recorder is still
    running), or None if the file looks finalized.

    This is a heuristic, not a guarantee: a slow/paused recorder can look
    finalized for a `grow_check_wait`-second window, and a fast one could
    theoretically finish between the two duration checks below. It exists to
    catch the common case cheaply, not to be authoritative.
    """
    try:
        size_before = os.path.getsize(video_path)
        time.sleep(grow_check_wait)
        size_after = os.path.getsize(video_path)
    except OSError:
        return None
    if size_after > size_before:
        return (
            "The source video appears to still be growing (it changed size "
            f"in the last {grow_check_wait:.1f}s) — the recorder may still be "
            "running. Processing an in-progress recording is unreliable and "
            "can make clip extraction extremely slow, since the container's "
            "index isn't written until the recording is stopped. Stop the "
            "recording, then re-select the file."
        )

    probed = probe_duration(video_path)
    if probed is None:
        return (
            "Could not determine the video's duration from its container "
            "(ffprobe found no valid index). This usually means the file "
            "was not finalized — e.g. the recorder was still running or was "
            "killed rather than stopped. Clip extraction can be extremely "
            "slow or fail entirely. Re-export/finalize the recording, or "
            "re-select the finished file, before continuing."
        )
    return None
