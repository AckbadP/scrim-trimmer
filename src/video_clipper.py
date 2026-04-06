"""
video_clipper.py - Extract clips and stitch them using ffmpeg.

Falls back to OpenCV-based Python processing when ffmpeg is not available.
Note: the Python fallback re-encodes video and does NOT preserve audio.
"""

import os
import shutil
import subprocess
import tempfile
from typing import List, Tuple


_force_python: bool = False


def set_force_python(value: bool) -> None:
    """Force the Python/OpenCV fallback even when ffmpeg is available."""
    global _force_python
    _force_python = value


def _use_python_clipper() -> bool:
    return _force_python or shutil.which("ffmpeg") is None


# ---------------------------------------------------------------------------
# ffmpeg implementations
# ---------------------------------------------------------------------------

def _extract_clip_ffmpeg(
    input_path: str,
    start_sec: int,
    end_sec: int,
    output_path: str,
) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-ss", str(start_sec),
        "-to", str(end_sec),
        "-i", input_path,
        "-c", "copy",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed extracting clip {start_sec}–{end_sec}s:\n{result.stderr}"
        )


def _stitch_clips_ffmpeg(clip_paths: List[str], output_path: str) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False
    ) as f:
        for path in clip_paths:
            f.write(f"file '{os.path.abspath(path)}'\n")
        concat_list_path = f.name

    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_list_path,
            "-c", "copy",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg concat failed:\n{result.stderr}")
    finally:
        os.unlink(concat_list_path)


# ---------------------------------------------------------------------------
# Python/OpenCV fallback implementations (no audio, re-encodes)
# ---------------------------------------------------------------------------

def _extract_clip_python(
    input_path: str,
    start_sec: int,
    end_sec: int,
    output_path: str,
) -> None:
    """Extract a clip using OpenCV. Re-encodes; audio is not preserved."""
    import cv2

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for _ in range(end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)

    cap.release()
    writer.release()


def _stitch_clips_python(clip_paths: List[str], output_path: str) -> None:
    """Concatenate clips using OpenCV. Re-encodes; audio is not preserved."""
    import cv2

    cap0 = cv2.VideoCapture(clip_paths[0])
    fps = cap0.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap0.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for clip_path in clip_paths:
        cap = cv2.VideoCapture(clip_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
        cap.release()

    writer.release()


# ---------------------------------------------------------------------------
# Public API — dispatches to ffmpeg or Python fallback
# ---------------------------------------------------------------------------

def extract_clip(
    input_path: str,
    start_sec: int,
    end_sec: int,
    output_path: str,
) -> None:
    """
    Extract a clip from a video file.

    Uses ffmpeg (stream-copy, with audio) when available, otherwise falls back
    to OpenCV (re-encodes, no audio).

    Args:
        input_path: Source video path (.mp4 or .mkv).
        start_sec: Clip start time in seconds.
        end_sec: Clip end time in seconds.
        output_path: Destination path for the extracted clip.
    """
    if not _use_python_clipper():
        _extract_clip_ffmpeg(input_path, start_sec, end_sec, output_path)
    else:
        if _force_python:
            print("  [info] Python clipper forced — re-encoding, no audio")
        else:
            print("  [warning] ffmpeg not found — using Python fallback (no audio, re-encodes)")
        _extract_clip_python(input_path, start_sec, end_sec, output_path)


def stitch_clips(clip_paths: List[str], output_path: str) -> None:
    """
    Concatenate multiple clips into a single video.

    Uses ffmpeg (stream-copy, with audio) when available, otherwise falls back
    to OpenCV (re-encodes, no audio).

    Args:
        clip_paths: Ordered list of clip paths to concatenate.
        output_path: Destination path for the final stitched video.
    """
    if not clip_paths:
        raise ValueError("No clips to stitch.")

    if not _use_python_clipper():
        _stitch_clips_ffmpeg(clip_paths, output_path)
    else:
        if _force_python:
            print("  [info] Python clipper forced — re-encoding, no audio")
        else:
            print("  [warning] ffmpeg not found — using Python fallback (no audio, re-encodes)")
        _stitch_clips_python(clip_paths, output_path)


def create_clips(
    input_path: str,
    pairs: List[Tuple[int, int]],
    output_dir: str,
) -> List[str]:
    """
    Extract all (cd, wf) clips from the input video.

    Args:
        input_path: Source video path (.mp4 or .mkv).
        pairs: List of (start_sec, end_sec) tuples.
        output_dir: Directory to save individual clips.

    Returns:
        Ordered list of clip file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    clip_paths = []

    for i, (start, end) in enumerate(pairs, start=1):
        clip_path = os.path.join(output_dir, f"clip_{i:03d}.mp4")
        print(f"  Extracting clip {i}: {start}s → {end}s → {clip_path}")
        extract_clip(input_path, start, end, clip_path)
        clip_paths.append(clip_path)

    return clip_paths
