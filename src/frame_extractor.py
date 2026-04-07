"""
frame_extractor.py - Extract one frame per second from a video file.
"""

import cv2
from typing import Generator, Tuple
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
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        cap.release()
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

        # Skip frames between the current position and the target by grabbing
        # (cheap: demux without decode) rather than seeking.
        while current_frame < target_frame:
            if not cap.grab():
                cap.release()
                return
            current_frame += 1

        ret, frame = cap.read()
        if not ret:
            break
        current_frame += 1

        yield next_target, frame
        next_target += 1

    cap.release()


def get_video_duration(video_path: str) -> float:
    """Return the duration of the video in seconds."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames / fps if fps > 0 else 0.0
