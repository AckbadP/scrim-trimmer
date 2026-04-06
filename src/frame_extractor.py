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

    duration_seconds = int(total_frames / fps) + 1

    for second in range(duration_seconds):
        frame_number = int(second * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if ret:
            yield second, frame

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
