"""
Tests for log_matcher.py — pure-logic functions only (no video/OCR required).
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from log_matcher import _normalize, _find_best_t0, _load_unique_messages


# ---------------------------------------------------------------------------
# _normalize
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_lowercases(self):
        assert _normalize("Hello World") == "hello world"

    def test_removes_punctuation(self):
        assert _normalize("we'll be just a minute!") == "we ll be just a minute"

    def test_collapses_whitespace(self):
        assert _normalize("too   many   spaces") == "too many spaces"

    def test_empty_string(self):
        assert _normalize("") == ""

    def test_only_punctuation(self):
        assert _normalize("!!!") == ""

    def test_mixed(self):
        assert _normalize("CD CD CD") == "cd cd cd"


# ---------------------------------------------------------------------------
# _find_best_t0
# ---------------------------------------------------------------------------

class TestFindBestT0:
    def test_single_candidate(self):
        assert _find_best_t0([3600]) == 3600

    def test_clear_majority(self):
        # Ten candidates at 3600, one outlier
        candidates = [3600] * 10 + [5000]
        assert _find_best_t0(candidates) == 3600

    def test_cluster_within_2s(self):
        # Candidates at 3600, 3601, 3602 all count as the same cluster
        candidates = [3600, 3601, 3602, 3601, 3600]
        result = _find_best_t0(candidates)
        assert abs(result - 3601) <= 2

    def test_prefers_higher_on_tie(self):
        # Two equally sized clusters; the higher one should win
        candidates = [100, 100, 200, 200]
        assert _find_best_t0(candidates) == 200

    def test_real_world_spread(self):
        # Most candidates cluster near true t0 (3725), a few stragglers below
        candidates = [3725, 3724, 3725, 3726, 3700, 3680, 3720, 3725, 3723]
        result = _find_best_t0(candidates)
        assert abs(result - 3725) <= 2


# ---------------------------------------------------------------------------
# _load_unique_messages
# ---------------------------------------------------------------------------

def _write_log(content: str, encoding: str = "utf-16") -> str:
    fd, path = tempfile.mkstemp(suffix=".txt")
    with os.fdopen(fd, "wb") as f:
        f.write(content.encode(encoding))
    return path


_LOG_A = """\
\ufeff
 Session started: 2026.03.25 01:00:00

\ufeff[ 2026.03.25 01:07:49 ] Alice > we are ready to fight
\ufeff[ 2026.03.25 01:09:00 ] Bob > short
\ufeff[ 2026.03.25 01:10:00 ] Alice > duplicate message here
"""

_LOG_B = """\
\ufeff
 Session started: 2026.03.25 01:00:00

\ufeff[ 2026.03.25 01:11:00 ] Carol > unique message in log b
\ufeff[ 2026.03.25 01:12:00 ] Alice > duplicate message here
"""


class TestLoadUniqueMessages:
    def test_returns_unique_only(self):
        path_a = _write_log(_LOG_A)
        path_b = _write_log(_LOG_B)
        try:
            result = _load_unique_messages([path_a, path_b])
            # "duplicate message here" appears twice → excluded
            for norm_msg in result:
                assert "duplicate" not in norm_msg
        finally:
            os.unlink(path_a)
            os.unlink(path_b)

    def test_excludes_short_messages(self):
        path = _write_log(_LOG_A)
        try:
            result = _load_unique_messages([path])
            # "short" is 5 chars after normalisation → excluded
            for norm_msg in result:
                assert norm_msg != "short"
        finally:
            os.unlink(path)

    def test_includes_long_unique(self):
        path_a = _write_log(_LOG_A)
        path_b = _write_log(_LOG_B)
        try:
            result = _load_unique_messages([path_a, path_b])
            assert "we are ready to fight" in result
            assert "unique message in log b" in result
        finally:
            os.unlink(path_a)
            os.unlink(path_b)

    def test_game_seconds_correct(self):
        path = _write_log(_LOG_A)
        try:
            result = _load_unique_messages([path])
            # 01:07:49 → 1*3600 + 7*60 + 49 = 4069
            assert result["we are ready to fight"] == 4069
        finally:
            os.unlink(path)
