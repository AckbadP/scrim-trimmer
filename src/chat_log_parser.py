"""
chat_log_parser.py - Parse EVE Online local-chat log files to find CD/WF commands.

EVE exports chat logs as UTF-16 LE files.  Each message line has the form:
    [ YYYY.MM.DD HH:MM:SS ] PlayerName > message text

Multiple log files may cover different (non-overlapping) portions of the same
session; pass them all in together.

Usage:
    cd_times, wf_times = parse_chat_logs(
        log_paths,
        t0_seconds,   # EVE game-time seconds at video second 0
        duration,     # total video duration in seconds
    )
"""

import re
from datetime import datetime, timezone
from typing import List, Tuple, Optional


# Match: [ 2026.03.25 01:07:49 ] PlayerName > message
_LOG_LINE_RE = re.compile(
    r'\[\s*(\d{4}\.\d{2}\.\d{2}\s+\d{2}:\d{2}:\d{2})\s*\]\s+(.+?)\s+>\s+(.+)'
)

_FIRST_WORD_CD = re.compile(r'^\s*CD\b', re.IGNORECASE)
_FIRST_WORD_WF = re.compile(r'^\s*WF\b', re.IGNORECASE)


def _parse_log_line(line: str) -> Optional[Tuple[datetime, str, str]]:
    """Parse a single chat log line; return (timestamp, player, message) or None."""
    line = line.strip().lstrip('\ufeff')
    m = _LOG_LINE_RE.match(line)
    if not m:
        return None
    ts_str, player, message = m.group(1), m.group(2).strip(), m.group(3).strip()
    try:
        ts = datetime.strptime(ts_str.strip(), '%Y.%m.%d %H:%M:%S').replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    return ts, player, message


def read_chat_log(path: str) -> List[Tuple[datetime, str, str]]:
    """Read one EVE chat log file; return list of (timestamp, player, message)."""
    with open(path, 'rb') as f:
        raw = f.read()
    try:
        text = raw.decode('utf-16')
    except UnicodeDecodeError:
        text = raw.decode('utf-16-le', errors='replace')
    entries = []
    for line in text.split('\n'):
        parsed = _parse_log_line(line)
        if parsed:
            entries.append(parsed)
    return entries


def parse_chat_logs(
    log_paths: List[str],
    t0_game_seconds: float,
    video_duration: float,
) -> Tuple[List[int], List[int]]:
    """
    Extract CD and WF video timestamps from EVE chat log files.

    Args:
        log_paths: Paths to one or more EVE chat log files (any order; will be
            merged and sorted chronologically).
        t0_game_seconds: EVE game time (seconds since midnight UTC on the log
            date) at video second 0.  Compute as:
                t0_game_seconds = hh * 3600 + mm * 60 + ss
            where HH:MM:SS is the EVE game time when video recording began.
        video_duration: Total video length in seconds; events outside
            [0, duration] are discarded.

    Returns:
        (cd_timestamps, wf_timestamps): lists of integer video seconds.
    """
    # Collect all entries from all files, deduplicate by (timestamp, player,
    # message) in case files overlap slightly.
    all_entries: dict = {}
    for path in log_paths:
        for ts, player, msg in read_chat_log(path):
            key = (ts, player, msg)
            all_entries[key] = (ts, player, msg)

    sorted_entries = sorted(all_entries.values(), key=lambda x: x[0])

    cd_timestamps: List[int] = []
    wf_timestamps: List[int] = []

    for ts, player, msg in sorted_entries:
        words = msg.strip().split()
        if not words:
            continue
        # Strip surrounding non-word characters before classifying (e.g. "-CD-" → "CD").
        # Skip tokens that are pure punctuation (e.g. "*****" in "***** CD ******")
        # and use the first token that has at least one word character.
        first = ''
        for w in words:
            stripped = re.sub(r'^\W+|\W+$', '', w).upper()
            if stripped:
                first = stripped
                break
        is_cd = first in ('CD', 'COUNTDOWN')
        is_wf = first in ('WF', 'GF')
        if not is_cd and not is_wf:
            continue

        # Convert game timestamp to video seconds.
        # game_sec = HH*3600 + MM*60 + SS on the UTC date of the log entry.
        game_sec = ts.hour * 3600 + ts.minute * 60 + ts.second
        video_sec = game_sec - t0_game_seconds

        # Handle midnight wrap: if game_sec < t0, the log crosses midnight.
        if video_sec < -3600:
            video_sec += 86400  # add 24 hours

        if video_sec < 0 or video_sec > video_duration:
            continue

        if is_cd:
            cd_timestamps.append(int(video_sec))
        else:
            wf_timestamps.append(int(video_sec))

    return cd_timestamps, wf_timestamps


def game_time_to_seconds(time_str: str) -> int:
    """
    Convert an EVE game time string to seconds-since-midnight.

    Accepted formats: "HH:MM:SS" or "HH:MM".
    """
    parts = time_str.strip().split(':')
    if len(parts) == 3:
        h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
    elif len(parts) == 2:
        h, m, s = int(parts[0]), int(parts[1]), 0
    else:
        raise ValueError(f"Invalid game time format: {time_str!r}")
    return h * 3600 + m * 60 + s
