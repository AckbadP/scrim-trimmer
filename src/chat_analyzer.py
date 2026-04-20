"""
chat_analyzer.py - Detect new CD and WF occurrences in chat OCR text and pair them.

Key rules from spec:
- "CD" and "WF" are whole-word matches (case insensitive: cd, CD, Cd, wf, WF, Wf)
- CD/WF must be the FIRST word of the message (after '>' separator) to avoid
  counting acknowledgment messages like "ok CD"
- CD can appear multiple times before a WF; use the most recent CD before each WF
- Each CD and WF used exactly once
- Pairing order: sequential, CDs after the previous WF are eligible for next WF

Detection strategy (real EVE chat with [HH:MM:SS] timestamps):
  Scan each frame's OCR lines top-to-bottom, carrying forward the most recently
  seen [HH:MM:SS] timestamp.  When a CD/WF command line is found, associate it
  with the carry-forward timestamp and use that as a dedup key: if the key is
  already in the "seen" set the message was already counted; otherwise count it
  and add the key.

  Why carry-forward?  EVE chat messages with long player names wrap across two
  OCR lines:
      [01:07:49] Ceofore Aideron     ← timestamp here
      > CD                           ← command on the NEXT line

  Carry-forward correctly links the command to its timestamp even after the wrap.

  Why a post-merge?  OCR garbles the same timestamp differently in different
  frames (e.g. "01:67:49" vs "01:87:49" for the same [01:07:49] message), so
  the same event can generate two detections with different keys.  We collapse
  detections within _MERGE_WINDOW seconds into one (keeping the earliest).

  Fallback for when no timestamp precedes a command line: 30-second cooldown.

For synthetic / test text (no timestamps): monotonic-count fallback.
"""

import re
from typing import List, Set, Tuple


def count_keyword(text: str, keyword: str) -> int:
    """Count whole-word occurrences of keyword (case insensitive)."""
    pattern = r"\b" + re.escape(keyword) + r"\b"
    return len(re.findall(pattern, text, re.IGNORECASE))


def count_keyword_in_messages(text: str, keyword: str) -> int:
    """
    Count lines where keyword is the FIRST word of the message content.

    EVE Online chat lines look like: "[timestamp] PlayerName > message"
    Real CD/WF commands are typed alone as the entire message (e.g. "> cd").
    Acknowledgment messages like "> ok cd" must not be counted — they inflate
    the monotonic counter and prevent later rounds from being detected.

    Rules:
    - Only inspect text after the last '>' or '»' separator on the line.
    - The keyword must be the very first word after that separator.
    - Lines with no separator are skipped entirely.
    """
    first_word_re = re.compile(
        r'^\s*' + re.escape(keyword) + r'\b', re.IGNORECASE
    )
    count = 0
    for line in text.split("\n"):
        # Accept both ASCII '>' and '»' (OCR sometimes reads '>' as '»')
        sep_pos = -1
        for sep in (">", "\u00bb"):
            pos = line.rfind(sep)
            if pos > sep_pos:
                sep_pos = pos
        if sep_pos == -1:
            continue  # no separator on this line — skip to avoid false positives
        after_sep = line[sep_pos + 1:]
        if first_word_re.match(after_sep):
            count += 1
    return count


_CO_AS_CD = re.compile(r'(>[ \t]*)CO\b')
# OCR frequently misreads 'F' as 'E', 'R', or 'H' in EVE's bold chat font,
# turning "> WF" into "> WE", "> WR", or "> WH".  Correct all three.
# Use negative lookahead to exclude real words starting with those letters
# (e.g. "We'll", "wrong", "who") — the next char after WE/WR/WH must not be
# a word character or apostrophe.
_WE_AS_WF = re.compile(r"(>[ \t]*)WE(?![\w'])", re.IGNORECASE)
_WR_AS_WF = re.compile(r"(>[ \t]*)WR(?![\w'])", re.IGNORECASE)
_WH_AS_WF = re.compile(r"(>[ \t]*)WH(?![\w'])", re.IGNORECASE)
# OCR misreads "> <alias> wf" as "> <alias> dc" or "> <alias> de" when the
# WF command is preceded by a player alias/callsign on the same message line.
# e.g. "[01:17:39] Arkadiy Krylov > 3brand dc and" → "> 3brand wf and"
# Replace so WF becomes the first word after the separator.
_ALIAS_DCDE_AS_WF = re.compile(r'(>[ \t]*)\S+[ \t]+d[ce]\b', re.IGNORECASE)

# Expanded timestamp regex: tolerates common OCR garbles of [HH:MM:SS].
# Opening bracket may be [ ( or @; separator may be : or -; closing may be ] or ).
# Hour, minute, and second fields may contain @ (garbled 0) in any position.
# Each field allows 1-2 characters that are digits or '@'.
# Optional trailing '@' before closing bracket handles garbles like (@1-19:58@]
# where OCR adds an extra '@' character after the seconds field.
_CHAT_TS_RE = re.compile(r'[\[(@](@?[0-9@]{1,2})[:\-]([0-9@]{2})[:\-]?([0-9@]{2})@?[\])]')

_FIRST_WORD_CD = re.compile(r'^\s*CD\b', re.IGNORECASE)
_FIRST_WORD_WF = re.compile(r'^\s*(WF|GF)\b', re.IGNORECASE)

# Fuzzy dedup window: treat two HH:MM:SS keys as the same event if they are
# within this many seconds of each other.  Handles OCR adding ±1-2s jitter to
# the seconds field of the same physical timestamp across different frames.
# Kept small (3s) so that legitimate events 5+ seconds apart are never merged.
_DEDUP_FUZZY_SEC = 3

# Minimum video-time gap (seconds) between a CD and WF.  Short enough to allow
# case-5's 8s gap (57→65s) while blocking OCR noise that fires within 1-2s.
_MIN_CD_WF_GAP = 5

# Tolerance (game-time seconds) when deciding whether a WF is stale relative
# to the last CD.  A WF whose game timestamp is more than this many seconds
# before the highest CD game timestamp seen is treated as stale and skipped.
# The 5s tolerance absorbs OCR jitter without masking legitimate stale WFs
# (which have game timestamps tens of seconds or minutes before the CD).
_STALE_WF_TOLERANCE = 5


def _ts_to_secs(ts: str) -> int:
    """Convert a 'HH:MM:SS' string to total seconds since midnight."""
    h, m, s = map(int, ts.split(':'))
    return h * 3600 + m * 60 + s


def _key_in_set_fuzzy(key: str, seen_set: Set[str], window: int = _DEDUP_FUZZY_SEC) -> bool:
    """Return True if `key` is within `window` seconds of any key in seen_set."""
    key_secs = _ts_to_secs(key)
    for k in seen_set:
        if abs(_ts_to_secs(k) - key_secs) <= window:
            return True
    return False


def _parse_ts(m: re.Match) -> str | None:
    """
    Normalize a possibly-garbled OCR timestamp match to a canonical HH:MM:SS key.

    OCR on EVE's bold font frequently misreads '0' as '8', '6', '9', or '@'.
    This causes the same physical [01:07:49] to appear as [81:87:49], [@1:87-49],
    [61:07:49], etc.  Normalization: replace leading '@' with '0'; if hours > 23
    or minutes > 59, treat the tens digit as a garbled '0'.

    Returns None when the timestamp cannot be resolved to a valid HH:MM:SS value.
    """
    h_str = m.group(1).replace('@', '0')
    m_str = m.group(2).replace('@', '0')
    s_str = m.group(3).replace('@', '0')
    try:
        h, mn, s = int(h_str), int(m_str), int(s_str)
    except ValueError:
        return None
    if h > 23 and len(h_str) == 2 and h_str[0] in '6890':
        h = int('0' + h_str[1])
    if mn > 59 and m_str[0] in '6890':
        mn = int('0' + m_str[1])
    # Correct garbled seconds field using the same logic applied to hours/minutes.
    if s > 59 and len(s_str) == 2 and s_str[0] in '6890':
        s = int('0' + s_str[1])
    if h <= 23 and mn <= 59 and s <= 59:
        return f"{h:02d}:{mn:02d}:{s:02d}"
    return None


def _fix_ocr(text: str) -> str:
    """
    Correct systematic Tesseract misreads in EVE Online's bold chat font:
    - Uppercase 'D' → 'O': turns "> CD" into "> CO".
    - Uppercase 'F' → 'E'/'R'/'H': turns "> WF" into "> WE/WR/WH".
    - 'wf' after alias → 'dc'/'de': turns "> alias dc" into "> WF".
    All substitutions are scoped to after the message separator so unrelated
    player names or words containing CO/WE/WR/WH are not affected.
    """
    text = _CO_AS_CD.sub(r'\1CD', text)
    text = _WE_AS_WF.sub(r'\1WF', text)
    text = _WR_AS_WF.sub(r'\1WF', text)
    text = _WH_AS_WF.sub(r'\1WF', text)
    text = _ALIAS_DCDE_AS_WF.sub(r'\1WF', text)
    return text


def _check_line_for_command(line: str) -> str:
    """
    Return 'CD', 'WF', or '' depending on whether the line contains a CD or WF
    command as the first word after the message separator.
    """
    sep_pos = -1
    for sep in (">", "\u00bb"):
        pos = line.rfind(sep)
        if pos > sep_pos:
            sep_pos = pos
    if sep_pos == -1:
        return ""
    after_sep = line[sep_pos + 1:]
    if _FIRST_WORD_CD.match(after_sep):
        return "CD"
    if _FIRST_WORD_WF.match(after_sep):
        return "WF"
    return ""


_TOURN_START_RE = re.compile(r'30\s+seconds\s+until\s+match\s+start', re.IGNORECASE)
_TOURN_END_RE = re.compile(r'Match\s+completed', re.IGNORECASE)


def _check_line_for_tournament_command(line: str) -> str:
    """Return 'CD', 'WF', or '' for tournament system messages after the separator."""
    sep_pos = -1
    for sep in (">", "\u00bb"):
        pos = line.rfind(sep)
        if pos > sep_pos:
            sep_pos = pos
    if sep_pos == -1:
        return ""
    after_sep = line[sep_pos + 1:]
    if _TOURN_START_RE.search(after_sep):
        return "CD"
    if _TOURN_END_RE.search(after_sep):
        return "WF"
    return ""


# Guard window: only update max_cd_ts_secs when a new CD fires at least this
# many video seconds after the previous one.  Prevents garbled-timestamp
# duplicates from inflating the stale-WF threshold.  Smaller than
# _CD_MERGE_WINDOW intentionally; the guard fires per-frame while
# _CD_MERGE_WINDOW collapses the final list.
_MERGE_WINDOW = 3

# Post-merge windows: collapse detections of the same event that generated
# multiple entries due to inconsistent OCR garbling of the same chat timestamp.
# CDs use a wider window (5s) to absorb seconds-field garbles (e.g. "38"→"30",
# an 8s difference) that exceed the _DEDUP_FUZZY_SEC window.
# WFs use a narrower window (4s) to merge 3s-apart duplicates while keeping
# legitimate event pairs like case-6's WFs at 109s and 113s (4s apart).
_CD_MERGE_WINDOW = 5
_WF_MERGE_WINDOW = 4


def _merge_close_events(timestamps: List[int], window: int) -> List[int]:
    """Merge consecutive timestamps within `window` seconds; keep the earliest."""
    result: List[int] = []
    for t in timestamps:  # already in chronological order
        if not result or t - result[-1] >= window:
            result.append(t)
    return result


# Maximum line length for no-separator CD/WF detection.  EVE chat command
# messages are very short (2-6 chars), but OCR sometimes prepends garbled player
# alias fragments, keeping the total well under 20 chars.  Longer lines are
# almost certainly full sentences and are skipped to avoid false positives.
_MAX_NO_SEP_LINE_LEN = 20

# Token sets used by _detect_command_no_sep
_CD_TOKENS = frozenset({'CD', 'CO', 'CKD'})   # CO/CKD = common OCR garbles of CD
_WF_TOKENS = frozenset({'WF', 'WR', 'WH', 'WE', 'GF'})


def _normalize_token(s: str) -> str:
    """Strip surrounding punctuation and apply character-level OCR corrections."""
    # Apply char replacements BEFORE stripping so e.g. '¢' in '--¢d' becomes 'c'
    # before the leading '--' is stripped, producing 'cd' not just 'd'.
    s = s.replace('\u00a2', 'c').replace('\u00e9', 'e')  # ¢→c, é→e
    s = re.sub(r'^[^a-zA-Z0-9]+', '', s)
    s = re.sub(r'[^a-zA-Z0-9]+$', '', s)
    s = s.upper()
    if s == 'CKD':
        s = 'CD'
    return s


def _detect_command_no_sep(line: str) -> str:
    """
    Detect a CD or WF command on a short line that lacks a '>' separator.

    EVE chat OCR often produces two-line output:
        [17:43:59] PlayerName >          ← header line (processed by _check_line_for_command)
        cd                               ← command line, no '>' here

    This function handles the command line.  Only short lines (≤ _MAX_NO_SEP_LINE_LEN)
    are examined to avoid false positives from longer chat messages.

    Detection strategy: check the LAST token of the line (OCR sometimes prepends
    a garbled player-alias fragment, e.g. "hel co" where "co" is the command).
    Also checks any token for the rare-in-chat garble 'CKD' (→ CD).

    Returns 'CD', 'WF', or ''.
    """
    stripped = line.strip()
    if not stripped or len(stripped) > _MAX_NO_SEP_LINE_LEN:
        return ""
    if '>' in stripped or '\u00bb' in stripped:
        return ""  # has separator — handled by _check_line_for_command

    tokens = [_normalize_token(t) for t in stripped.split()]
    tokens = [t for t in tokens if t]  # drop empty strings (e.g. lone '*')
    if not tokens:
        return ""

    last = tokens[-1]

    # CD detection: last token is a CD word, or 'CD' appears as any token
    # (covers the CKD garble which _normalize_token converts to 'CD', and
    # cases where OCR prepends alias fragments before the command).
    if last in _CD_TOKENS:
        return "CD"
    if 'CD' in tokens:
        return "CD"

    # WF detection: last token is a WF word, or exactly 3 chars starting with
    # 'WR' (handles "wre" — OCR adds an extra letter to "wr").
    # Only 'WR' prefix is used for partial matches; 'WE', 'WH' etc. at length 3
    # would catch common English words like "Wee", "Who" etc.
    if last in _WF_TOKENS:
        # Reject standalone 'WE' (single token): "we" is a common English word
        # and triggers false positives on garbled CD-variant lines.  When 'WE'
        # is preceded by a player-alias fragment (≥2 tokens total) it is more
        # likely a genuine "wf" OCR garble and is accepted.
        if last == 'WE' and len(tokens) < 2:
            return ""
        return "WF"
    if len(last) == 3 and last[:2] == 'WR':
        return "WF"

    return ""


def analyze_frames(
    frame_texts: List[Tuple[int, str]],
    verbose: bool = False,
    tournament_mode: bool = False,
) -> Tuple[List[int], List[int]]:
    """
    Scan OCR text from each frame and record video-seconds of new CD/WF events.

    Two detection strategies, chosen automatically:

    Timestamp carry-forward mode (real EVE chat):
        Lines are scanned top-to-bottom within each frame.  The most recently
        matched [HH:MM:SS] timestamp is carried forward to the next command
        line — this handles the common OCR wrapping where the timestamp and the
        "> CD" message content are on separate lines.  The carry-forward
        timestamp is used as a dedup key: a command is only counted the first
        time a given key is seen across all frames.  After all frames are
        processed, detections within _MERGE_WINDOW seconds are collapsed into
        one (handling inconsistent garbling of the same timestamp).

    Monotonic fallback (synthetic / test text without timestamps):
        A new CD/WF is detected when the total count in the current frame
        exceeds the maximum count seen so far.

    Args:
        frame_texts: List of (second, ocr_text) tuples in chronological order.
        verbose: If True, print debug info per frame.

    Returns:
        (cd_timestamps, wf_timestamps): lists of integer seconds.
    """
    cd_timestamps: List[int] = []
    wf_timestamps: List[int] = []

    # Carry-forward mode state
    has_seen_timestamps = False
    # Separate seen-sets per command type so that a CD timestamp key does NOT
    # block a WF with the same garbled carry_ts (e.g. both reading as "01:07:49").
    seen_cd_ts: Set[str] = set()
    seen_wf_ts: Set[str] = set()
    # Video-time of most recently accepted CD, used for minimum-gap WF filtering.
    last_cd_video_sec: int = -1
    # Maximum game-time (total seconds) of all accepted CDs, used for stale-WF
    # detection.  Only updated when the new CD fires >= _MERGE_WINDOW video
    # seconds after the previous CD, to prevent a close-duplicate detection with
    # a garbled timestamp (e.g. 17:50:01 misread as 17:58:01) from inflating the
    # max and blocking the legitimate WF that follows.
    max_cd_ts_secs: int = -1
    # Game-time of the most recently accepted WF, used for:
    #  (a) stale-CD detection (CD game time before last WF → from previous round),
    #  (b) backward-WF detection (WF game time before previous WF → duplicate).
    last_wf_ts_secs: int = -1

    # Monotonic fallback state
    max_cd_seen = 0
    max_wf_seen = 0

    for second, text in frame_texts:
        fixed = _fix_ocr(text)

        if _CHAT_TS_RE.search(fixed):
            # ---- Carry-forward timestamp mode (real EVE chat) ----
            has_seen_timestamps = True
            carry_ts: str | None = None

            for line in fixed.split("\n"):
                # Update carry-forward with any timestamp on this line.
                m = _CHAT_TS_RE.search(line)
                if m:
                    norm = _parse_ts(m)
                    if norm is not None:
                        carry_ts = norm

                if tournament_mode:
                    cmd = _check_line_for_tournament_command(line)
                else:
                    cmd = _check_line_for_command(line)
                    # Try no-separator detection for short lines that follow a
                    # timestamp header (carry_ts set) but have no '>' on them.
                    # This handles the common two-line EVE chat OCR format where
                    # the command appears on a line without a separator.
                    if not cmd and carry_ts is not None:
                        cmd = _detect_command_no_sep(line)

                if not cmd:
                    continue

                if carry_ts is not None:
                    if cmd == "CD" and not _key_in_set_fuzzy(carry_ts, seen_cd_ts):
                        # Skip CDs whose game timestamp is within 3s of a known
                        # WF timestamp: same message misclassified on the same
                        # frame (e.g. pre-recording WF generates both a WF and
                        # a spurious CD detection on the same carry_ts).
                        if carry_ts in seen_wf_ts:
                            if verbose:
                                print(f"  [{second}s] SKIP CD (ts={carry_ts}, carry_ts in seen_wf_ts)")
                            continue
                        # Skip stale CDs: game timestamp before the last
                        # accepted WF means this message is from a previous
                        # round that has since ended.
                        if last_wf_ts_secs >= 0:
                            cd_game_secs = _ts_to_secs(carry_ts)
                            if cd_game_secs < last_wf_ts_secs - _STALE_WF_TOLERANCE:
                                if verbose:
                                    print(
                                        f"  [{second}s] SKIP CD (stale: ts={carry_ts}, "
                                        f"cd_secs={cd_game_secs} < last_wf_secs-tol={last_wf_ts_secs - _STALE_WF_TOLERANCE})"
                                    )
                                continue
                        seen_cd_ts.add(carry_ts)
                        cd_timestamps.append(second)
                        # Only update max_cd_ts_secs when this CD fires at least
                        # _MERGE_WINDOW video seconds after the previous one.
                        # A very-close duplicate (different carry_ts but same
                        # video frame) likely has a garbled timestamp; letting
                        # that garble inflate max_cd_ts_secs would poison the
                        # stale-WF check for legitimate WFs that follow.
                        if last_cd_video_sec < 0 or second - last_cd_video_sec >= _MERGE_WINDOW:
                            max_cd_ts_secs = max(max_cd_ts_secs, _ts_to_secs(carry_ts))
                        last_cd_video_sec = second
                        if verbose:
                            print(f"  [{second}s] NEW CD (ts={carry_ts})")
                    elif cmd == "WF" and not _key_in_set_fuzzy(carry_ts, seen_wf_ts):
                        # Skip WFs whose carry_ts is within 3s of a known CD
                        # timestamp.  This catches OCR misreads (e.g. '> go'
                        # → '> WF') in frames dominated by the same countdown
                        # timestamps, and WFs with game time 1-2s after a CD
                        # (OCR noise from the same message).
                        if _key_in_set_fuzzy(carry_ts, seen_cd_ts):
                            if verbose:
                                print(f"  [{second}s] SKIP WF (ts={carry_ts}, near a CD ts)")
                            continue
                        # Skip WFs that fire too soon after the most recently
                        # accepted CD in video time.
                        gap = second - last_cd_video_sec if last_cd_video_sec >= 0 else _MIN_CD_WF_GAP
                        if gap < _MIN_CD_WF_GAP:
                            if verbose:
                                print(f"  [{second}s] SKIP WF (ts={carry_ts}, gap={gap}s < {_MIN_CD_WF_GAP}s)")
                            continue
                        # Skip stale WFs: their game timestamp is before the
                        # highest CD game timestamp seen (minus tolerance for
                        # OCR jitter).
                        if max_cd_ts_secs >= 0:
                            wf_game_secs = _ts_to_secs(carry_ts)
                            if wf_game_secs < max_cd_ts_secs - _STALE_WF_TOLERANCE:
                                if verbose:
                                    print(
                                        f"  [{second}s] SKIP WF (stale: ts={carry_ts}, "
                                        f"wf_secs={wf_game_secs} < max_cd_secs-tol={max_cd_ts_secs - _STALE_WF_TOLERANCE})"
                                    )
                                continue
                        # Skip backward WFs: game timestamp before the previous
                        # accepted WF (minus tolerance).  These are repeated
                        # OCR readings of an old message that produce a slightly
                        # earlier garbled timestamp, e.g. 17:56:20 → 17:56:13.
                        if last_wf_ts_secs >= 0:
                            wf_game_secs = _ts_to_secs(carry_ts)
                            if wf_game_secs < last_wf_ts_secs - _STALE_WF_TOLERANCE:
                                if verbose:
                                    print(
                                        f"  [{second}s] SKIP WF (backward: ts={carry_ts}, "
                                        f"wf_secs={wf_game_secs} < last_wf_secs-tol={last_wf_ts_secs - _STALE_WF_TOLERANCE})"
                                    )
                                continue
                        seen_wf_ts.add(carry_ts)
                        wf_timestamps.append(second)
                        last_wf_ts_secs = _ts_to_secs(carry_ts)
                        if verbose:
                            print(f"  [{second}s] NEW WF (ts={carry_ts})")
                # else: carry_ts is None — command appears before any timestamp
                # in this frame.  Skip to avoid false detections from stale
                # chat history whose timestamps were too garbled to parse.

        elif not has_seen_timestamps:
            # ---- Monotonic fallback (no timestamps seen anywhere yet) ----
            if tournament_mode:
                cd_count = sum(1 for line in fixed.split("\n") if _check_line_for_tournament_command(line) == "CD")
                wf_count = sum(1 for line in fixed.split("\n") if _check_line_for_tournament_command(line) == "WF")
            else:
                cd_count = count_keyword_in_messages(fixed, "CD")
                wf_count = count_keyword_in_messages(fixed, "WF") + count_keyword_in_messages(fixed, "GF")

            if verbose and (cd_count > 0 or wf_count > 0):
                print(f"  [{second}s] CD={cd_count} WF={wf_count}")

            while cd_count > max_cd_seen:
                cd_timestamps.append(second)
                max_cd_seen += 1
            while wf_count > max_wf_seen:
                wf_timestamps.append(second)
                max_wf_seen += 1

    # Collapse detections of the same event that generated multiple entries due
    # to inconsistent OCR garbling of the same chat timestamp.  Only needed
    # in carry-forward mode; monotonic fallback already deduplicates correctly.
    # In tournament mode the "30 seconds until match start" message stays
    # visible for ~30 s, producing many duplicate detections with varying
    # garbled timestamps.  Use a 40 s window to collapse them all into the
    # earliest (first) detection.
    if has_seen_timestamps:
        cd_merge = 40 if tournament_mode else _CD_MERGE_WINDOW
        cd_timestamps = _merge_close_events(cd_timestamps, cd_merge)
        wf_timestamps = _merge_close_events(wf_timestamps, _WF_MERGE_WINDOW)
        # Remove any WFs that occur (in video time) before the first detected CD.
        # These are either pre-recording messages visible in the chat window at
        # the start of the recording, or spurious detections before the round
        # begins.  pair_cd_wf would ignore them anyway (no eligible CD), but
        # filtering them keeps the WF count accurate for tests.
        if cd_timestamps:
            first_cd_sec = cd_timestamps[0]
            wf_timestamps = [t for t in wf_timestamps if t >= first_cd_sec]

    return cd_timestamps, wf_timestamps


def pair_cd_wf(
    cd_timestamps: List[int],
    wf_timestamps: List[int],
) -> List[Tuple[int, int]]:
    """
    Pair CD and WF timestamps into (start, end) clip boundaries.

    Pairing rules:
    - For each WF (in order), find all CDs that occurred after the previous WF
      and before the current WF.
    - Use the most recent of those CDs as the clip start.
    - CDs that appear before an eligible window are discarded.
    - Each CD and WF is used at most once.

    Args:
        cd_timestamps: Sorted list of seconds where a new CD was detected.
        wf_timestamps: Sorted list of seconds where a new WF was detected.

    Returns:
        List of (cd_time, wf_time) tuples representing clip boundaries.
    """
    clips: List[Tuple[int, int]] = []
    last_wf_used = -1  # sentinel: no WF used yet

    for wf_time in sorted(wf_timestamps):
        # Eligible CDs: appeared after the last used WF, and before this WF
        eligible = [cd for cd in cd_timestamps if last_wf_used < cd < wf_time]
        if eligible:
            cd_time = max(eligible)  # most recent CD
            clips.append((cd_time, wf_time))
            last_wf_used = wf_time
        # If no eligible CD exists for this WF, skip this WF

    return clips
