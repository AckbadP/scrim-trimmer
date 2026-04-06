"""
main.py - CLI entry point for the EVE AT Practice Trimmer.

Pipeline:
  1. Extract 1 frame/sec from the input video
  2. Run OCR on the left chat window region of each frame
  3. Detect new CD and WF occurrences, pair them up
  4. Extract CD→WF clips with ffmpeg
  5. Stitch all clips into a final output video
"""

import argparse
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

from frame_extractor import extract_frames, get_video_duration
from ocr_processor import crop_chat_region, run_ocr_on_region
from chat_analyzer import analyze_frames, pair_cd_wf
from chat_log_parser import parse_chat_logs, game_time_to_seconds
from log_matcher import detect_t0
import video_clipper
from video_clipper import create_clips, stitch_clips


def _check_dependencies(args) -> None:
    """Verify required external tools are available before starting a long job."""
    if not getattr(args, "force_python_clipper", False) and shutil.which("ffmpeg") is None:
        print("Error: 'ffmpeg' not found in PATH. Install ffmpeg and try again.", file=sys.stderr)
        sys.exit(1)
    if shutil.which("tesseract") is None:
        print("Error: 'tesseract' not found in PATH. Install tesseract-ocr and try again.", file=sys.stderr)
        sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Trim EVE Online AT practice footage using OCR-detected CD/WF commands."
    )
    parser.add_argument("video", help="Path to the input video file (.mp4 or .mkv)")
    parser.add_argument(
        "--output", "-o",
        default="out",
        help="Output directory for clips and final video (default: out/)",
    )
    parser.add_argument(
        "--chat-region",
        type=float,
        nargs=4,
        metavar=("X1", "Y1", "X2", "Y2"),
        default=[0.0, 0.35, 0.15, 1.0],
        help="Chat window region as fractions: x1 y1 x2 y2 (default: 0.0 0.35 0.15 1.0)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        metavar="N",
        help="Number of OCR worker threads (default: number of CPU cores)",
    )
    parser.add_argument(
        "--ram-cap-gb",
        type=int,
        default=10,
        metavar="GB",
        dest="ram_cap_gb",
        help="Maximum GB of video frames to hold in memory at once (default: 10)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print OCR text and detection info per frame",
    )
    parser.add_argument(
        "--chat-log",
        action="append",
        metavar="FILE",
        dest="chat_logs",
        help="EVE chat log file to use instead of OCR (may be specified multiple times)",
    )
    parser.add_argument(
        "--t0",
        metavar="HH:MM:SS",
        help=(
            "EVE game time (UTC) at video second 0. "
            "Required with --chat-log unless it can be auto-detected from the video."
        ),
    )
    parser.add_argument(
        "--chapters-dir",
        default=None,
        metavar="DIR",
        dest="chapters_dir",
        help="Directory to save the YouTube chapters .txt file (default: same as output dir)",
    )
    parser.add_argument(
        "--force-python-clipper",
        action="store_true",
        dest="force_python_clipper",
        help=(
            "Use the Python/OpenCV clipper instead of ffmpeg. "
            "Slower, re-encodes video, and strips audio. "
            "Useful when ffmpeg is not available."
        ),
    )
    parser.add_argument(
        "--force-ocr",
        action="store_true",
        dest="force_ocr",
        help=(
            "Force OCR pipeline even when --chat-log is provided. "
            "Ignores chat logs and detects CD/WF events via OCR instead."
        ),
    )
    return parser.parse_args()


def _fmt_duration(seconds: float) -> str:
    """Format seconds as h:mm:ss or m:ss."""
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def write_chapter_timestamps(
    pairs: List[Tuple[int, int]],
    output_path: str,
    chapters_dir: str | None = None,
) -> str:
    """
    Write a YouTube-compatible chapter timestamps file.

    Each line is "<timestamp> Match <n>", where the timestamp is the
    cumulative start time of that clip in the stitched output video.
    The first chapter always starts at 0:00, as required by YouTube.

    Args:
        pairs: Ordered list of (start_sec, end_sec) tuples used to build clips.
        output_path: Path to the stitched output video; used to derive the
                     default .txt filename.
        chapters_dir: Directory to write the .txt file. If None or empty,
                      the file is saved alongside output_path.

    Returns:
        Path to the written .txt file.
    """
    base_name = os.path.splitext(os.path.basename(output_path))[0] + "_chapters.txt"
    if chapters_dir:
        os.makedirs(chapters_dir, exist_ok=True)
        txt_path = os.path.join(chapters_dir, base_name)
    else:
        txt_path = os.path.join(os.path.dirname(output_path), base_name)
    cumulative = 0
    lines = []
    for i, (start, end) in enumerate(pairs, start=1):
        lines.append(f"{_fmt_duration(cumulative)} Match {i}")
        cumulative += end - start

    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    return txt_path


def _progress_bar(current: int, total: int, elapsed: float, width: int = 30) -> str:
    """Return a one-line progress string with bar, percent, and ETA."""
    pct = current / total if total > 0 else 0
    filled = int(width * pct)
    bar = "#" * filled + "-" * (width - filled)
    eta_str = ""
    if current > 0 and elapsed > 0:
        rate = current / elapsed  # frames per second
        remaining = (total - current) / rate
        eta_str = f"  ETA {_fmt_duration(remaining)}"
    return f"\r  [{bar}] {pct*100:5.1f}%  {current}/{total}s{eta_str}  "


def run(args) -> None:
    """Run the trimmer pipeline with a pre-built args namespace."""
    video_clipper.set_force_python(getattr(args, "force_python_clipper", False))
    _check_dependencies(args)

    args.video = os.path.expanduser(args.video)
    args.output = os.path.expanduser(args.output)
    if args.chapters_dir:
        args.chapters_dir = os.path.expanduser(args.chapters_dir)
    if args.chat_logs:
        args.chat_logs = [os.path.expanduser(p) for p in args.chat_logs]

    if not os.path.isfile(args.video):
        print(f"Error: video file not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    duration = int(get_video_duration(args.video))
    print(f"Processing: {args.video}  ({_fmt_duration(duration)})")

    # --- Chat log mode: skip OCR entirely ---
    if args.chat_logs and not getattr(args, "force_ocr", False):
        if args.t0:
            try:
                t0_sec = game_time_to_seconds(args.t0)
            except ValueError as e:
                print(f"Error: invalid --t0 value: {e}", file=sys.stderr)
                sys.exit(1)
            t0_source = f"t0={args.t0} (provided)"
        else:
            print(f"\n[0/4] Auto-detecting t0 from chat log(s) "
                  f"(sampling 1 frame per 30 s)...")
            try:
                t0_sec = detect_t0(
                    args.chat_logs,
                    args.video,
                    chat_region=tuple(args.chat_region),
                    verbose=args.verbose,
                    progress_callback=getattr(args, 'progress_callback', None),
                )
            except (RuntimeError, ValueError) as e:
                print(f"Error: {e}", file=sys.stderr)
                sys.exit(1)
            hh = t0_sec // 3600
            mm = (t0_sec % 3600) // 60
            ss = t0_sec % 60
            t0_source = f"t0={hh:02d}:{mm:02d}:{ss:02d} (auto-detected)"

        output_dir = args.output
        os.makedirs(output_dir, exist_ok=True)

        print(f"Chat log mode: {t0_source}  ({len(args.chat_logs)} log file(s))")
        print("\n[1/4] Parsing chat log(s)...")
        cd_times, wf_times = parse_chat_logs(args.chat_logs, t0_sec, duration)
        print(f"  Found {len(cd_times)} CD(s) at: {cd_times}")
        print(f"  Found {len(wf_times)} WF(s) at: {wf_times}")

        pairs = pair_cd_wf(cd_times, wf_times)
        print(f"  Paired {len(pairs)} clip(s): {pairs}")

        paired_cds = {cd for cd, _ in pairs}
        paired_wfs = {wf for _, wf in pairs}
        orphan_cds = [t for t in cd_times if t not in paired_cds]
        orphan_wfs = [t for t in wf_times if t not in paired_wfs]
        if orphan_cds:
            print(f"  Warning: {len(orphan_cds)} CD(s) skipped: {orphan_cds}", file=sys.stderr)
        if orphan_wfs:
            print(f"  Warning: {len(orphan_wfs)} WF(s) skipped: {orphan_wfs}", file=sys.stderr)

        if not pairs:
            print("\nNo CD→WF pairs found. Nothing to clip.", file=sys.stderr)
            sys.exit(0)

        print(f"\n[2/4] Extracting {len(pairs)} clip(s)...")
        clip_paths = create_clips(args.video, pairs, output_dir)

        final_output = os.path.join(output_dir, "final_output.mp4")
        print(f"\n[3/4] Stitching clips into {final_output}...")
        stitch_clips(clip_paths, final_output)

        chapters_path = write_chapter_timestamps(pairs, final_output, getattr(args, "chapters_dir", None))

        print(f"\nDone! Final video: {final_output}")
        print(f"Chapter timestamps:  {chapters_path}")
        print(f"Individual clips saved to: {output_dir}/")
        with open(chapters_path) as _f:
            return _f.read()

    # --- OCR mode ---
    x1, y1, x2, y2 = args.chat_region
    print(f"Chat region: ({x1:.2f},{y1:.2f}) → ({x2:.2f},{y2:.2f})")

    # Step 1 & 2: Extract frames and run OCR
    workers = args.threads if (getattr(args, "threads", None) and args.threads > 0) else (os.cpu_count() or 4)
    print(f"\n[1/4] Extracting frames and running OCR ({duration} frames, {workers} threads)...")
    start_time = time.monotonic()

    results: dict = {}
    results_lock = threading.Lock()
    completed_count = [0]
    sem: threading.Semaphore | None = None

    def _make_callback(s: int):
        def callback(fut):
            with results_lock:
                results[s] = fut.result()
                completed_count[0] += 1
                if not args.verbose:
                    elapsed = time.monotonic() - start_time
                    sys.stdout.write(_progress_bar(completed_count[0], duration, elapsed))
                    sys.stdout.flush()
            sem.release()
        return callback

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for second, frame in extract_frames(args.video):
            region = crop_chat_region(frame, *args.chat_region)
            del frame
            if sem is None:
                ram_cap_bytes = (getattr(args, "ram_cap_gb", None) or 10) * 1024 ** 3
                max_in_flight = max(workers * 2, int(ram_cap_bytes // region.nbytes))
                sem = threading.Semaphore(max_in_flight)
            sem.acquire()
            executor.submit(run_ocr_on_region, region).add_done_callback(_make_callback(second))

    elapsed = time.monotonic() - start_time
    frame_texts = sorted(results.items())

    if not args.verbose:
        sys.stdout.write("\n")
    else:
        for second, text in frame_texts:
            print(f"  [{second}s] OCR: {text[:100].replace(chr(10), ' ')}")

    rate = len(frame_texts) / elapsed if elapsed > 0 else 0
    print(f"  Done. {len(frame_texts)} frames in {_fmt_duration(elapsed)} ({rate:.1f} fps)")

    # Step 3: Detect CD/WF and pair them
    print("\n[2/4] Analyzing chat for CD/WF commands...")
    cd_times, wf_times = analyze_frames(frame_texts, verbose=args.verbose)
    print(f"  Found {len(cd_times)} CD(s) at: {cd_times}")
    print(f"  Found {len(wf_times)} WF(s) at: {wf_times}")

    pairs = pair_cd_wf(cd_times, wf_times)
    print(f"  Paired {len(pairs)} clip(s): {pairs}")

    # Warn about unpaired events so users know if something was missed
    paired_cds = {cd for cd, _ in pairs}
    paired_wfs = {wf for _, wf in pairs}
    orphan_cds = [t for t in cd_times if t not in paired_cds]
    orphan_wfs = [t for t in wf_times if t not in paired_wfs]
    if orphan_cds:
        print(f"  Warning: {len(orphan_cds)} CD(s) had no matching WF and were skipped: {orphan_cds}", file=sys.stderr)
    if orphan_wfs:
        print(f"  Warning: {len(orphan_wfs)} WF(s) had no preceding CD and were skipped: {orphan_wfs}", file=sys.stderr)

    if not pairs:
        print("\nNo CD→WF pairs found. Nothing to clip.", file=sys.stderr)
        sys.exit(0)

    # Step 4: Extract clips
    print(f"\n[3/4] Extracting {len(pairs)} clip(s)...")
    clip_paths = create_clips(args.video, pairs, output_dir)

    # Step 5: Stitch clips
    final_output = os.path.join(output_dir, "final_output.mp4")
    print(f"\n[4/4] Stitching clips into {final_output}...")
    stitch_clips(clip_paths, final_output)

    chapters_path = write_chapter_timestamps(pairs, final_output)

    print(f"\nDone! Final video: {final_output}")
    print(f"Chapter timestamps:  {chapters_path}")
    print(f"Individual clips saved to: {output_dir}/")
    with open(chapters_path) as _f:
        return _f.read()


def main():
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
