"""
Tests for chat_log_parser.py
"""

import sys
import os
import tempfile

from chat_log_parser import read_chat_log, parse_chat_logs, game_time_to_seconds, _parse_log_line


# ---------------------------------------------------------------------------
# game_time_to_seconds
# ---------------------------------------------------------------------------

class TestGameTimeToSeconds:
    def test_hh_mm_ss(self):
        assert game_time_to_seconds("01:07:49") == 1 * 3600 + 7 * 60 + 49

    def test_hh_mm(self):
        assert game_time_to_seconds("01:07") == 1 * 3600 + 7 * 60

    def test_midnight(self):
        assert game_time_to_seconds("00:00:00") == 0

    def test_end_of_day(self):
        assert game_time_to_seconds("23:59:59") == 23 * 3600 + 59 * 60 + 59


# ---------------------------------------------------------------------------
# read_chat_log
# ---------------------------------------------------------------------------

def _write_log(content: str, encoding: str = 'utf-16') -> str:
    """Write content as a UTF-16 chat log to a temp file; return the path."""
    fd, path = tempfile.mkstemp(suffix='.txt')
    with os.fdopen(fd, 'wb') as f:
        f.write(content.encode(encoding))
    return path


_SAMPLE_LOG = """\
\ufeff
 - - - - - - - - -
 Channel ID:      local
 Channel Name:    Local
 Listener:        Test Player
 Session started: 2026.03.25 01:00:00
 - - - - - - - - -

\ufeff[ 2026.03.25 01:07:49 ] Ceofore Aideron > CD
\ufeff[ 2026.03.25 01:14:56 ] Ceofore Aideron > CD
\ufeff[ 2026.03.25 01:17:39 ] Arkadiy Krylov > 3brand wf
\ufeff[ 2026.03.25 01:19:50 ] Rima Ambraelle > wf
\ufeff[ 2026.03.25 01:20:00 ] Some Player > ok WF
"""


class TestReadChatLog:
    def test_reads_messages(self):
        path = _write_log(_SAMPLE_LOG)
        try:
            entries = read_chat_log(path)
            assert len(entries) == 5
        finally:
            os.unlink(path)

    def test_correct_timestamp(self):
        path = _write_log(_SAMPLE_LOG)
        try:
            entries = read_chat_log(path)
            ts, player, msg = entries[0]
            assert ts.hour == 1 and ts.minute == 7 and ts.second == 49
            assert player == "Ceofore Aideron"
            assert msg == "CD"
        finally:
            os.unlink(path)

    def test_skips_non_message_lines(self):
        path = _write_log(_SAMPLE_LOG)
        try:
            entries = read_chat_log(path)
            # Only lines matching [ ts ] player > msg should be returned
            players = [e[1] for e in entries]
            assert "Channel ID" not in players
            assert "Session started" not in players
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# parse_chat_logs
# ---------------------------------------------------------------------------

class TestParseChatLogs:
    def _make_log(self, messages):
        """messages: list of (HH:MM:SS, player, text)"""
        lines = ["\ufeff\n Session started: 2026.03.25 00:00:00\n"]
        for ts, player, msg in messages:
            lines.append(f"\ufeff[ 2026.03.25 {ts} ] {player} > {msg}")
        path = _write_log("\n".join(lines))
        return path

    def test_cd_and_wf_detected(self):
        path = self._make_log([
            ("01:07:49", "Ceofore", "CD"),
            ("01:14:56", "Ceofore", "CD"),
            ("01:19:50", "Rima", "wf"),
        ])
        # t0 = 01:05:00 = 3900s; video 0-3600s
        t0 = game_time_to_seconds("01:05:00")
        try:
            cd_times, wf_times = parse_chat_logs([path], t0, 3600)
            # CD at 01:07:49 → video t = 1:07:49 - 1:05:00 = 2:49 = 169s
            # CD at 01:14:56 → video t = 9:56 = 596s
            # WF at 01:19:50 → video t = 14:50 = 890s
            assert 169 in cd_times
            assert 596 in cd_times
            assert 890 in wf_times
        finally:
            os.unlink(path)

    def test_events_outside_duration_excluded(self):
        path = self._make_log([
            ("01:07:49", "P", "CD"),   # t=169s
            ("01:30:00", "P", "WF"),   # t=1500s → beyond 600s duration
        ])
        t0 = game_time_to_seconds("01:05:00")
        try:
            cd_times, wf_times = parse_chat_logs([path], t0, 600)
            assert 169 in cd_times
            assert len(wf_times) == 0  # WF at t=1500 is out of range
        finally:
            os.unlink(path)

    def test_events_before_t0_excluded(self):
        path = self._make_log([
            ("01:00:00", "P", "CD"),   # before T0
            ("01:07:49", "P", "WF"),   # after T0 → t=169s
        ])
        t0 = game_time_to_seconds("01:05:00")
        try:
            cd_times, wf_times = parse_chat_logs([path], t0, 3600)
            assert len(cd_times) == 0
            assert 169 in wf_times
        finally:
            os.unlink(path)

    def test_ok_wf_not_detected(self):
        # "ok WF" must not be counted — WF is not the first word
        path = self._make_log([
            ("01:07:49", "P", "CD"),
            ("01:14:56", "P", "ok WF"),
        ])
        t0 = game_time_to_seconds("01:05:00")
        try:
            cd_times, wf_times = parse_chat_logs([path], t0, 3600)
            assert len(wf_times) == 0
        finally:
            os.unlink(path)

    def test_multi_file_merge(self):
        # Two non-overlapping log files covering different periods
        path1 = self._make_log([
            ("01:07:49", "Ceofore", "CD"),
            ("01:14:56", "Ceofore", "CD"),
        ])
        path2 = self._make_log([
            ("01:19:50", "Rima", "wf"),
        ])
        t0 = game_time_to_seconds("01:05:00")
        try:
            cd_times, wf_times = parse_chat_logs([path1, path2], t0, 3600)
            assert len(cd_times) == 2
            assert len(wf_times) == 1
        finally:
            os.unlink(path1)
            os.unlink(path2)

    def test_cd_cd_cd_counts_once(self):
        # "CD CD CD" message — only first word is CD, so counts as 1 CD
        path = self._make_log([
            ("02:22:10", "Arkadiy", "CD CD CD"),
        ])
        t0 = game_time_to_seconds("01:05:00")
        try:
            cd_times, wf_times = parse_chat_logs([path], t0, 7200)
            # Should produce 1 CD, not 3
            assert len(cd_times) == 1
        finally:
            os.unlink(path)

    def test_duplicate_entries_across_overlapping_files_deduplicated(self):
        # Both files contain the same message (overlapping export windows).
        # The entry must appear only once in the output.
        path1 = self._make_log([
            ("01:07:49", "Ceofore", "CD"),
            ("01:19:50", "Rima", "wf"),
        ])
        path2 = self._make_log([
            ("01:07:49", "Ceofore", "CD"),   # duplicate
        ])
        t0 = game_time_to_seconds("01:05:00")
        try:
            cd_times, wf_times = parse_chat_logs([path1, path2], t0, 3600)
            assert len(cd_times) == 1
        finally:
            os.unlink(path1)
            os.unlink(path2)

    def test_midnight_wrap_excluded(self):
        # video_sec = game_sec - t0; if that result is < -3600 the code adds
        # 86400 (midnight crossing).  A message whose adjusted video_sec still
        # falls outside [0, duration] must be excluded.
        # Simulate: t0=1*3600=3600 (01:00:00), event at 00:30:00 → video_sec=-1800
        # -1800 > -3600 so no wrap is applied → excluded (video_sec < 0)
        path = self._make_log([
            ("00:30:00", "P", "CD"),  # before t0, no wrap applied
        ])
        t0 = game_time_to_seconds("01:00:00")
        try:
            cd_times, _ = parse_chat_logs([path], t0, 3600)
            assert len(cd_times) == 0
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# game_time_to_seconds — error handling
# ---------------------------------------------------------------------------

class TestGameTimeToSecondsErrors:
    def test_invalid_format_raises_value_error(self):
        import pytest
        with pytest.raises(ValueError, match="Invalid game time"):
            game_time_to_seconds("01:07:49:00")

    def test_single_component_raises_value_error(self):
        import pytest
        with pytest.raises(ValueError):
            game_time_to_seconds("3600")


# ---------------------------------------------------------------------------
# _parse_log_line
# ---------------------------------------------------------------------------

class TestParseLogLine:
    def test_valid_line_returns_tuple(self):
        line = "[ 2026.03.25 01:07:49 ] Ceofore Aideron > CD"
        result = _parse_log_line(line)
        assert result is not None
        ts, player, msg = result
        assert ts.hour == 1 and ts.minute == 7 and ts.second == 49
        assert player == "Ceofore Aideron"
        assert msg == "CD"

    def test_non_message_line_returns_none(self):
        assert _parse_log_line(" Channel ID:      local") is None
        assert _parse_log_line("") is None
        assert _parse_log_line(" - - - - - - - - -") is None

    def test_bom_stripped(self):
        # Lines in UTF-16 logs often start with a BOM character (\ufeff)
        line = "\ufeff[ 2026.03.25 01:07:49 ] Player > hello"
        result = _parse_log_line(line)
        assert result is not None
        _, player, msg = result
        assert player == "Player"
        assert msg == "hello"

    def test_whitespace_trimmed_from_player_and_message(self):
        line = "[ 2026.03.25 01:07:49 ]  Padded Player  >  trimmed message  "
        result = _parse_log_line(line)
        assert result is not None
        _, player, msg = result
        assert player == "Padded Player"
        assert msg == "trimmed message"
