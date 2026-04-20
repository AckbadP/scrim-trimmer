"""
Tests for chat_analyzer.py

Covers:
- count_keyword: whole-word matching, case insensitivity
- analyze_frames: monotonic count detection, multiple occurrences
- pair_cd_wf: pairing rules, most-recent-CD logic, WF with no CD, etc.
"""

import sys
import os

from unittest.mock import MagicMock

from chat_analyzer import (
    count_keyword,
    count_keyword_in_messages,
    analyze_frames,
    pair_cd_wf,
    _fix_ocr,
    _CHAT_TS_RE,
    _parse_ts,
    _key_in_set_fuzzy,
    _MIN_CD_WF_GAP,
    _DEDUP_FUZZY_SEC,
    _check_line_for_command,
    _merge_close_events,
    _detect_command_no_sep,
)


# ---------------------------------------------------------------------------
# _fix_ocr
# ---------------------------------------------------------------------------

class TestFixOcr:
    def test_co_to_cd_after_arrow(self):
        assert _fix_ocr("> CO") == "> CD"

    def test_co_to_cd_with_spaces(self):
        assert _fix_ocr(">  CO") == ">  CD"

    def test_we_to_wf_after_arrow(self):
        assert _fix_ocr("> WE") == "> WF"

    def test_we_to_wf_lowercase(self):
        assert _fix_ocr("> we") == "> WF"

    def test_we_to_wf_mixed_case(self):
        assert _fix_ocr("> We") == "> WF"

    def test_we_not_replaced_before_arrow(self):
        # "We Player > hello" — 'We' before separator must not be changed
        assert _fix_ocr("We Player > hello") == "We Player > hello"

    def test_co_not_replaced_before_arrow(self):
        assert _fix_ocr("CO Player > hello") == "CO Player > hello"

    def test_no_false_positive_wee(self):
        # "> wee" must not be changed (not whole-word WE)
        assert _fix_ocr("> wee") == "> wee"

    def test_no_false_positive_contraction(self):
        # "> We'll" must not be changed — "We'll be just a minute" is NOT a WF command
        line = "> We'll be just a minute"
        assert _fix_ocr(line) == line

    def test_real_ocr_snippet_wf(self):
        # Simulate wrapped OCR line: timestamp on one line, command on next
        text = "[01:19:50] Rima Ambraelle\n> we"
        fixed = _fix_ocr(text)
        assert "> WF" in fixed

    def test_wr_to_wf_after_arrow(self):
        # OCR misreads 'F' as 'R': "> wr" → "> WF"
        assert _fix_ocr("> wr") == "> WF"

    def test_wr_to_wf_uppercase(self):
        assert _fix_ocr("> WR") == "> WF"

    def test_wr_not_replaced_in_wrong(self):
        # "> wrong" must not be changed — 'wr' is followed by 'o' (word char)
        assert _fix_ocr("> wrong") == "> wrong"

    def test_wr_not_replaced_before_arrow(self):
        assert _fix_ocr("WR Player > hello") == "WR Player > hello"

    def test_wh_to_wf_after_arrow(self):
        # OCR misreads 'F' as 'H': "> wh" → "> WF"
        assert _fix_ocr("> wh") == "> WF"

    def test_wh_to_wf_uppercase(self):
        assert _fix_ocr("> WH") == "> WF"

    def test_wh_not_replaced_in_who(self):
        # "> who" must not be changed — 'wh' is followed by 'o' (word char)
        assert _fix_ocr("> who") == "> who"

    def test_wh_not_replaced_before_arrow(self):
        assert _fix_ocr("WH Player > hello") == "WH Player > hello"

    def test_analyze_frames_detects_wr_as_wf(self):
        # "> wr" OCR garble (seen in ag7 section_03) flows through analyze_frames
        ts_text = "[01:31:33] Rima Ambraelle\n> wr"
        cd_text = "[01:26:23] 3brand > CD"
        frames = [
            (11,  cd_text),
            (317, ts_text),   # gap=306s > _MIN_CD_WF_GAP; carry_ts '01:31' ≠ '01:26'
        ]
        cd_times, wf_times = analyze_frames(frames)
        assert wf_times == [317]

    def test_analyze_frames_detects_wh_as_wf(self):
        # "> wh" OCR garble (seen in ag7 section_03 second 318)
        ts_text = "[01:31:33] Rima Ambraelle\n> wh"
        cd_text = "[01:26:23] 3brand > CD"
        frames = [
            (11,  cd_text),
            (318, ts_text),
        ]
        cd_times, wf_times = analyze_frames(frames)
        assert wf_times == [318]

    def test_trailing_at_in_timestamp_matched(self):
        # (@1-19:58@] has a trailing '@' before ']' — regex must still match
        m = _CHAT_TS_RE.search("(@1-19:58@] Rima Ambraelle")
        assert m is not None
        assert _parse_ts(m) == "01:19:58"

    def test_analyze_frames_detects_we_as_wf(self):
        # Ensure the WE→WF correction flows through analyze_frames
        ts_text = "[01:19:50] Rima Ambraelle\n> we"
        cd_text = "[01:07:49] Ceofore Aideron\n> CD"
        frames = [
            (100, cd_text),
            (400, ts_text),   # gap=300s > _MIN_CD_WF_GAP; carry_ts '01:19' ≠ '01:07'
        ]
        cd_times, wf_times = analyze_frames(frames)
        assert wf_times == [400]

    def test_alias_dc_to_wf(self):
        # "> 3brand dc" is an OCR garble of "> 3brand wf" — must be corrected
        assert "> WF" in _fix_ocr("> 3brand dc")

    def test_alias_de_to_wf(self):
        assert "> WF" in _fix_ocr("> callsign de")

    def test_alias_dc_case_insensitive(self):
        assert "> WF" in _fix_ocr("> Alias DC")

    def test_alias_dc_not_matched_before_arrow(self):
        # "dc" before separator must not be changed
        assert _fix_ocr("3brand dc > hello") == "3brand dc > hello"

    def test_analyze_frames_detects_alias_dc_as_wf(self):
        # Real pattern: "[01:17:39] Arkadiy Krylov > 3brand dc and"
        ts_text = "[01:17:39] Arkadiy Krylov\n> 3brand dc and"
        cd_text = "[01:07:49] Ceofore Aideron\n> CD"
        frames = [
            (100, cd_text),
            (765, ts_text),
        ]
        cd_times, wf_times = analyze_frames(frames)
        assert wf_times == [765]


# ---------------------------------------------------------------------------
# _key_in_set_fuzzy
# ---------------------------------------------------------------------------

class TestKeyInSetFuzzy:
    def test_exact_match(self):
        assert _key_in_set_fuzzy("01:07:00", {"01:07:00"}) is True

    def test_within_window(self):
        assert _key_in_set_fuzzy("01:07:02", {"01:07:00"}) is True  # 2s diff ≤ 3s window
        assert _key_in_set_fuzzy("01:07:03", {"01:07:00"}) is True  # 3s diff = window

    def test_outside_window(self):
        assert _key_in_set_fuzzy("01:07:04", {"01:07:00"}) is False  # 4s > 3s window

    def test_empty_set(self):
        assert _key_in_set_fuzzy("01:07:00", set()) is False

    def test_hour_boundary(self):
        assert _key_in_set_fuzzy("01:00:01", {"00:59:59"}) is True  # 2s diff

    def test_fuzzy_dedup_blocks_garbled_cd(self):
        # OCR adds ±1-2s jitter to the same physical timestamp on different frames.
        # Two detections of the same CD within 3s of each other → blocked.
        frames = [
            (600, "[01:07:49] Player\n> CD"),    # key '01:07:49' → detected
            (601, "[01:07:50] Player\n> CD"),    # key '01:07:50' → blocked (1s jitter)
            (602, "[01:07:51] Player\n> CD"),    # key '01:07:51' → blocked (2s jitter)
        ]
        cd_times, _ = analyze_frames(frames)
        assert cd_times == [600]

    def test_fuzzy_dedup_blocks_garbled_wf(self):
        # Two detections of the same WF within 3s of each other → keep only first.
        wf_765 = "[01:17:39] Player\n> 3brand dc"  # garble of wf — becomes WF via _fix_ocr
        wf_766 = "[01:17:40] Player\n> WF"          # 1s jitter — blocked
        cd_text = "[01:07:49] Player\n> CD"
        frames = [
            (100, cd_text),
            (765, wf_765),
            (766, wf_766),
        ]
        _, wf_times = analyze_frames(frames)
        assert wf_times == [765]


# ---------------------------------------------------------------------------
# min-gap WF suppression
# ---------------------------------------------------------------------------

class TestMinGapWfSuppression:
    """WF detected within _MIN_CD_WF_GAP seconds of the most recent CD is suppressed."""

    def test_wf_too_soon_after_cd_skipped(self):
        # CD fires at t=2314, garbled-duplicate CD fires at t=2323,
        # then a stale WF fires at t=2332 (9s after last CD) → must be skipped.
        cd1 = "[01:41:49] Working Class\nMan > CD"
        cd2 = "[01:01:49] Working Class\nMan > CD"   # garbled dup, key '01:01'
        wf_stale = "[01:31:33] Rima\n> WF"           # stale WF, key '01:31'
        frames = [
            (2314, cd1),
            (2323, cd2),
            (2332, wf_stale),
        ]
        _, wf_times = analyze_frames(frames)
        assert wf_times == []  # stale WF suppressed

    def test_wf_at_gap_boundary_accepted(self):
        # CD at t=100, WF at t=100+_MIN_CD_WF_GAP → accepted (gap == threshold)
        cd = "[01:07:49] Player\n> CD"
        wf = "[01:17:39] Player\n> WF"
        frames = [
            (100, cd),
            (100 + _MIN_CD_WF_GAP, wf),
        ]
        _, wf_times = analyze_frames(frames)
        assert wf_times == [100 + _MIN_CD_WF_GAP]

    def test_wf_before_any_cd_accepted(self):
        # No CD has been accepted yet → WF should be accepted regardless of gap
        wf = "[01:07:41] Working Class\nMan > WF got this"
        frames = [(160, wf)]
        _, wf_times = analyze_frames(frames)
        assert wf_times == [160]

    def test_stale_wf_does_not_steal_cd(self):
        # Stale WF fires 9s after CD; a legitimate later WF should still pair.
        cd_real    = "[01:41:49] Player\n> CD"      # key '01:41:49'
        cd_garbled = "[01:01:49] Player\n> CD"      # garbled dup, key '01:01:49'
        wf_stale   = "[01:31:33] Rima\n> WF"        # stale: game time 01:31:33 < 01:41:49
        wf_legit   = "[01:49:11] Rima\n> WF"        # correct WF, game time after all CDs
        frames = [
            (2314, cd_real),
            (2323, cd_garbled),
            (2332, wf_stale),    # stale by game timestamp → suppressed
            (2788, wf_legit),    # game time > max CD → accepted
        ]
        cd_times, wf_times = analyze_frames(frames)
        assert wf_times == [2788]
        # cd_garbled has a distinct HH:MM:SS key so both CDs are detected;
        # _MERGE_WINDOW is small, so both survive the merge.  pair_cd_wf picks
        # the most recent eligible CD (2323) before the WF at 2788.
        assert pair_cd_wf(cd_times, wf_times) == [(2323, 2788)]

    def test_wf_with_nearby_but_distinct_cd_ts_not_blocked(self):
        # carry_ts '01:49' is only 2 min from CD carry_ts '01:47' but NOT an
        # exact match.  The exact-match guard must not block the legitimate WF.
        # (Models the real video case where a spurious CD@229 carry_ts='01:47'
        # gets merged away post-scan but the legitimate WF@2788 carry_ts='01:49'
        # must still fire.)
        cd = "[01:47:00] Player\n> CD"       # carry_ts='01:47' → in seen_cd_ts
        wf = "[01:49:00] Rima\n> WF"         # carry_ts='01:49' ≠ '01:47' exactly
        frames = [
            (229, cd),
            (784, wf),    # gap=555s > 120; carry_ts '01:49' not in seen_cd_ts → accepted
        ]
        cd_times, wf_times = analyze_frames(frames)
        assert 784 in wf_times


# ---------------------------------------------------------------------------
# count_keyword
# ---------------------------------------------------------------------------

class TestCountKeyword:
    def test_simple_match(self):
        assert count_keyword("cd wf", "cd") == 1

    def test_uppercase_match(self):
        assert count_keyword("CD WF", "CD") == 1

    def test_case_insensitive(self):
        # keyword "CD" should match "cd", "Cd", "CD"
        assert count_keyword("cd CD Cd cD", "CD") == 4

    def test_no_partial_match(self):
        # "CDx" or "xCD" must not match
        assert count_keyword("CDx xCD xCDx", "CD") == 0

    def test_multiple_occurrences(self):
        assert count_keyword("cd cd cd", "cd") == 3

    def test_wf_no_partial(self):
        assert count_keyword("wfw WFW", "WF") == 0

    def test_wf_whole_word(self):
        assert count_keyword("wf WF", "WF") == 2

    def test_empty_string(self):
        assert count_keyword("", "CD") == 0

    def test_cd_in_sentence(self):
        text = "[01:23:45] PlayerName > cd ?"
        assert count_keyword(text, "CD") == 1

    def test_wf_in_sentence(self):
        text = "[01:25:00] PlayerName > wf now"
        assert count_keyword(text, "WF") == 1


# ---------------------------------------------------------------------------
# count_keyword_in_messages
# ---------------------------------------------------------------------------

MSG = "[01:23:45] Player > {}"  # template for a realistic chat line


class TestCountKeywordInMessages:
    def test_cd_after_arrow(self):
        assert count_keyword_in_messages(MSG.format("cd"), "CD") == 1

    def test_wf_after_arrow(self):
        assert count_keyword_in_messages(MSG.format("wf"), "WF") == 1

    def test_case_insensitive(self):
        assert count_keyword_in_messages(MSG.format("CD"), "cd") == 1
        assert count_keyword_in_messages(MSG.format("WF"), "wf") == 1

    def test_keyword_before_arrow_ignored(self):
        # "Wf PlayerName > Panic" — false positive pattern from OCR noise
        assert count_keyword_in_messages("Wf PlayerName > Panic", "WF") == 0

    def test_cd_before_arrow_ignored(self):
        assert count_keyword_in_messages("CD Player > hello", "CD") == 0

    def test_no_arrow_line_skipped(self):
        # Lines without '>' are skipped entirely
        assert count_keyword_in_messages("cd wf", "CD") == 0
        assert count_keyword_in_messages("cd wf", "WF") == 0

    def test_guillemet_separator(self):
        # OCR sometimes reads '>' as '»'
        assert count_keyword_in_messages("Player \u00bb cd", "CD") == 1

    def test_ok_cd_after_arrow(self):
        # "ok cd" is an acknowledgment, not a command — must not be counted
        assert count_keyword_in_messages(MSG.format("ok cd"), "CD") == 0

    def test_multiple_messages(self):
        text = "\n".join([MSG.format("hello"), MSG.format("cd"), MSG.format("wf")])
        assert count_keyword_in_messages(text, "CD") == 1
        assert count_keyword_in_messages(text, "WF") == 1

    def test_partial_word_not_matched(self):
        assert count_keyword_in_messages(MSG.format("cdx xcD"), "CD") == 0

    def test_real_false_positive_pattern(self):
        # From actual OCR: "[01:07:16] Ceofore ® Kibitt\nWf Aideron > Panic"
        text = "[01:07:16] Ceofore \u00ae Kibitt\nWf Aideron > Panic"
        assert count_keyword_in_messages(text, "WF") == 0


# ---------------------------------------------------------------------------
# analyze_frames  (uses count_keyword_in_messages internally)
# ---------------------------------------------------------------------------

class TestAnalyzeFrames:
    def test_no_cd_no_wf(self):
        frames = [(5, "Player > hello world"), (10, "Player > nothing here")]
        cd_times, wf_times = analyze_frames(frames)
        assert cd_times == []
        assert wf_times == []

    def test_persistent_chat_not_double_counted(self):
        # CD appears at t=10, remains visible at t=11, t=12 — must only record once
        frames = [
            (10, "Player > cd"),
            (11, "Player > cd"),   # same CD still visible
            (12, "Player > cd"),
        ]
        cd_times, wf_times = analyze_frames(frames)
        assert cd_times == [10]

    def test_two_cds_before_wf(self):
        frames = [
            (5,  "Player > cd"),
            (10, "Player > cd\nPlayer > cd"),    # second CD appears
            (20, "Player > cd\nPlayer > cd\nPlayer > wf"), # WF appears
        ]
        cd_times, wf_times = analyze_frames(frames)
        assert cd_times == [5, 10]
        assert wf_times == [20]

    def test_ocr_noise_miss_then_reappear(self):
        # t=10 correctly sees CD=1, t=11 OCR misses it (count=0), t=12 OCR sees it again
        # Should NOT re-detect at t=12 since max_seen=1
        frames = [
            (10, "Player > cd"),
            (11, ""),   # OCR noise: missed
            (12, "Player > cd"), # CD reappears in OCR but wasn't new
        ]
        cd_times, _ = analyze_frames(frames)
        assert cd_times == [10]

    def test_second_cd_after_wf(self):
        frames = [
            (10, "Player > cd"),
            (20, "Player > cd\nPlayer > wf"),
            (30, "Player > cd\nPlayer > wf\nPlayer > cd"),   # new CD after WF
            (40, "Player > cd\nPlayer > wf\nPlayer > cd\nPlayer > wf"), # new WF
        ]
        cd_times, wf_times = analyze_frames(frames)
        assert cd_times == [10, 30]
        assert wf_times == [20, 40]

    def test_case_insensitive_detection(self):
        frames = [
            (5,  "Player > CD"),
            (10, "Player > CD\nPlayer > WF"),
        ]
        cd_times, wf_times = analyze_frames(frames)
        assert cd_times == [5]
        assert wf_times == [10]

    def test_keyword_before_arrow_not_detected(self):
        # OCR noise: "Wf PlayerName > Panic" — should not trigger WF detection
        frames = [
            (10, "Wf Player > Panic"),
            (20, "Wf Player > Panic\nWf Player > Panic"),
        ]
        cd_times, wf_times = analyze_frames(frames)
        assert wf_times == []


# ---------------------------------------------------------------------------
# pair_cd_wf
# ---------------------------------------------------------------------------

class TestPairCdWf:
    def test_simple_one_pair(self):
        pairs = pair_cd_wf([10], [20])
        assert pairs == [(10, 20)]

    def test_two_pairs_sequential(self):
        # CD1→WF1, CD2→WF2
        pairs = pair_cd_wf([10, 30], [20, 40])
        assert pairs == [(10, 20), (30, 40)]

    def test_two_cds_before_wf_uses_most_recent(self):
        # CD1=5, CD2=15, WF=25 → should use CD2 (most recent)
        pairs = pair_cd_wf([5, 15], [25])
        assert pairs == [(15, 25)]

    def test_cd_before_and_after_wf(self):
        # CD1=5, CD2=10, WF1=20, CD3=30, WF2=40
        # WF1 uses CD2 (most recent before WF1); CD1 is discarded
        # WF2 uses CD3 (after WF1)
        pairs = pair_cd_wf([5, 10, 30], [20, 40])
        assert pairs == [(10, 20), (30, 40)]

    def test_wf_with_no_preceding_cd(self):
        # WF at 5, CD at 10 — no eligible CD before WF
        pairs = pair_cd_wf([10], [5])
        assert pairs == []

    def test_extra_cds_discarded(self):
        # CD1=5, CD2=10, CD3=15, WF=20 — use most recent (CD3)
        pairs = pair_cd_wf([5, 10, 15], [20])
        assert pairs == [(15, 20)]

    def test_more_wfs_than_cds(self):
        # Only one CD, two WFs → only first WF gets paired
        pairs = pair_cd_wf([10], [20, 30])
        assert pairs == [(10, 20)]

    def test_empty_inputs(self):
        assert pair_cd_wf([], []) == []
        assert pair_cd_wf([10], []) == []
        assert pair_cd_wf([], [10]) == []

    def test_same_second_cd_before_wf(self):
        # CD and WF at same second: CD must be strictly before WF
        # cd < wf_time is strict, so same second means no pairing
        pairs = pair_cd_wf([10], [10])
        assert pairs == []

    def test_three_complete_pairs(self):
        # CD1=10, WF1=20, CD2=30, WF2=40, CD3=50, WF3=60
        pairs = pair_cd_wf([10, 30, 50], [20, 40, 60])
        assert pairs == [(10, 20), (30, 40), (50, 60)]


# ---------------------------------------------------------------------------
# _parse_ts
# ---------------------------------------------------------------------------

class TestParseTs:
    def _match(self, text: str):
        return _CHAT_TS_RE.search(text)

    def test_normal_timestamp(self):
        m = self._match("[01:07:49]")
        assert _parse_ts(m) == "01:07:49"

    def test_returns_hh_mm_ss(self):
        m = self._match("[23:59:59]")
        assert _parse_ts(m) == "23:59:59"

    def test_at_sign_replaces_zero_in_hour(self):
        # '@1:07:49' → h_str='01' after replace → h=1 → '01:07:49'
        m = self._match("[@1:07:49]")
        assert _parse_ts(m) == "01:07:49"

    def test_garbled_hour_tens_digit_corrected(self):
        # '81:07:49' → h=81 > 23, h_str[0]='8' ∈ '6890' → h = int('01') = 1
        m = self._match("[81:07:49]")
        assert _parse_ts(m) == "01:07:49"

    def test_garbled_minute_tens_digit_corrected(self):
        # '01:87:49' → mn=87 > 59, m_str[0]='8' ∈ '6890' → mn = int('07') = 7
        m = self._match("[01:87:49]")
        assert _parse_ts(m) == "01:07:49"

    def test_garbled_second_tens_digit_corrected(self):
        # '01:07:89' → s=89 > 59, s_str[0]='8' ∈ '6890' → s = int('09') = 9
        m = self._match("[01:07:89]")
        assert _parse_ts(m) == "01:07:09"

    def test_returns_none_for_unresolvable_timestamp(self):
        # Hour > 23 and tens digit not in '6890' → cannot correct → None
        m = self._match("[31:07:49]")
        assert _parse_ts(m) is None

    def test_midnight(self):
        m = self._match("[00:00:00]")
        assert _parse_ts(m) == "00:00:00"


# ---------------------------------------------------------------------------
# _check_line_for_command
# ---------------------------------------------------------------------------

class TestCheckLineForCommand:
    def test_returns_cd_for_cd_message(self):
        assert _check_line_for_command("Player > CD") == "CD"

    def test_returns_wf_for_wf_message(self):
        assert _check_line_for_command("Player > WF") == "WF"

    def test_case_insensitive_cd(self):
        assert _check_line_for_command("Player > cd") == "CD"

    def test_case_insensitive_wf(self):
        assert _check_line_for_command("Player > wf") == "WF"

    def test_returns_empty_for_other_message(self):
        assert _check_line_for_command("Player > hello world") == ""

    def test_returns_empty_for_no_separator(self):
        assert _check_line_for_command("CD WF no arrow") == ""

    def test_ok_cd_not_matched(self):
        assert _check_line_for_command("Player > ok CD") == ""

    def test_guillemet_separator_accepted(self):
        assert _check_line_for_command("Player \u00bb CD") == "CD"

    def test_cd_with_trailing_text(self):
        # "CD some text" — CD is still the first word
        assert _check_line_for_command("Player > CD some text") == "CD"


# ---------------------------------------------------------------------------
# _merge_close_events
# ---------------------------------------------------------------------------

class TestMergeCloseEvents:
    def test_empty_list(self):
        assert _merge_close_events([], 60) == []

    def test_single_item(self):
        assert _merge_close_events([100], 60) == [100]

    def test_items_within_window_merged_to_earliest(self):
        # 100 and 130 are 30s apart (< 60), so merged to 100
        assert _merge_close_events([100, 130], 60) == [100]

    def test_items_at_window_boundary_kept_separately(self):
        # 100 and 160 are exactly 60s apart (≥ window) → both kept
        assert _merge_close_events([100, 160], 60) == [100, 160]

    def test_items_beyond_window_kept_separately(self):
        assert _merge_close_events([100, 200], 60) == [100, 200]

    def test_chain_merges_to_first(self):
        # 100→130→150: 130-100=30 (merged), 150-100=50 (merged to same cluster)
        assert _merge_close_events([100, 130, 150], 60) == [100]

    def test_two_clusters(self):
        # 100, 130 → cluster1; 200 → cluster2 (200-100=100 ≥ 60)
        assert _merge_close_events([100, 130, 200], 60) == [100, 200]


# ---------------------------------------------------------------------------
# analyze_frames — carry_ts=None case
# ---------------------------------------------------------------------------

class TestAnalyzeFramesCarryTs:
    def test_command_before_any_timestamp_in_frame_is_skipped(self):
        # The "> CD" command appears on a line before any timestamp in the frame.
        # carry_ts is None when the command is processed → must not be counted.
        text = "> CD\n[01:07:49] Player\n> hello"
        frames = [(100, text)]
        cd_times, _ = analyze_frames(frames)
        assert cd_times == []

    def test_monotonic_fallback_disabled_after_timestamp_seen(self):
        # First frame has a real timestamp → has_seen_timestamps=True.
        # Second frame has no timestamp and contains CD/WF text;
        # the monotonic fallback must NOT fire for that second frame.
        ts_frame = "[01:07:49] Player\n> CD"
        no_ts_frame = "Player > WF"   # no EVE timestamp, monotonic would count this
        frames = [(100, ts_frame), (200, no_ts_frame)]
        _, wf_times = analyze_frames(frames)
        assert wf_times == []


# ---------------------------------------------------------------------------
# _detect_command_no_sep
# ---------------------------------------------------------------------------

class TestDetectCommandNoSep:
    """Two-line OCR: command on a short line without '>' separator."""

    def test_cd_exact(self):
        assert _detect_command_no_sep("cd") == "CD"

    def test_cd_uppercase(self):
        assert _detect_command_no_sep("CD") == "CD"

    def test_co_as_cd(self):
        # OCR frequently misreads 'D' as 'O'
        assert _detect_command_no_sep("co") == "CD"

    def test_cd_with_leading_garbage(self):
        # Player alias fragment prepended by OCR: "hel co" → last token 'co' → CD
        assert _detect_command_no_sep("hel co") == "CD"

    def test_cd_with_punctuation_prefix(self):
        # "--¢d" → normalize last token: strip '--', ¢→c → 'cd' → CD
        assert _detect_command_no_sep("--\u00a2d") == "CD"

    def test_ckd_garble_as_cd(self):
        assert _detect_command_no_sep("CKD") == "CD"

    def test_ckd_in_middle_of_line(self):
        # "he cn CKD appotte" — CKD anywhere on short line → CD
        assert _detect_command_no_sep("he cn CKD appotte") == "CD"

    def test_wf_exact(self):
        assert _detect_command_no_sep("wf") == "WF"

    def test_wr_as_wf(self):
        assert _detect_command_no_sep("wr") == "WF"

    def test_wre_partial_garble_as_wf(self):
        # "* wre" — last token 'wre' starts with 'wr', len=3 → WF
        assert _detect_command_no_sep("* wre") == "WF"

    def test_gf_as_wf(self):
        assert _detect_command_no_sep("gf") == "WF"

    def test_empty_line(self):
        assert _detect_command_no_sep("") == ""

    def test_line_too_long(self):
        assert _detect_command_no_sep("this line is definitely longer than twenty chars") == ""

    def test_line_with_separator_ignored(self):
        # Lines with '>' are handled by _check_line_for_command, not this function
        assert _detect_command_no_sep("> cd") == ""
        assert _detect_command_no_sep("Player > cd") == ""

    def test_no_cd_wf_word(self):
        assert _detect_command_no_sep("hello world") == ""

    def test_analyze_frames_detects_cd_no_sep(self):
        # Full pipeline: header line with timestamp, command on next line without '>'
        text = "[17:43:59] PlayerName >\nhel co"
        frames = [(4, text)]
        cd_times, _ = analyze_frames(frames)
        assert cd_times == [4]

    def test_analyze_frames_detects_wf_no_sep(self):
        cd = "[17:43:59] PlayerName >\nhel co"
        wf = "[17:44:33] PlayerName >\nwr"
        frames = [(4, cd), (38, wf)]
        cd_times, wf_times = analyze_frames(frames)
        assert cd_times == [4]
        assert wf_times == [38]

    def test_all_punctuation_tokens_empty(self):
        # "* * *" — every token normalizes to "" → not tokens → return ""
        assert _detect_command_no_sep("* * *") == ""

    def test_single_we_token_rejected(self):
        # Single token "WE" is rejected (common English word, needs alias context)
        assert _detect_command_no_sep("WE") == ""


# ---------------------------------------------------------------------------
# _parse_ts — ValueError branch
# ---------------------------------------------------------------------------

class TestParseTsValueError:
    def test_non_numeric_group_returns_none(self):
        # Simulate a regex match where a captured group is non-numeric after '@'
        # replacement — the except ValueError branch returns None.
        m = MagicMock()
        m.group.side_effect = lambda g: {1: "XX", 2: "07", 3: "49"}[g]
        assert _parse_ts(m) is None


# ---------------------------------------------------------------------------
# analyze_frames — verbose paths
# ---------------------------------------------------------------------------

class TestAnalyzeFramesVerbose:
    """Exercise every ``if verbose:`` branch in analyze_frames."""

    # -- NEW CD and NEW WF prints (lines 428, 475) ---------------------------

    def test_verbose_new_cd_and_wf(self, capsys):
        cd = "[01:07:49] Player\n> CD"
        wf = "[01:50:00] Rima\n> WF"
        frames = [(100, cd), (500, wf)]
        analyze_frames(frames, verbose=True)
        out = capsys.readouterr().out
        assert "NEW CD" in out
        assert "NEW WF" in out

    # -- SKIP CD: carry_ts already in seen_wf_ts (lines 401-403) ------------

    def test_verbose_skip_cd_in_seen_wf_ts(self, capsys):
        cd = "[01:07:49] P\n> CD"
        wf = "[01:50:00] Rima\n> WF"
        cd_stale = "[01:50:00] P\n> CD"  # carry_ts matches the WF ts
        frames = [(100, cd), (500, wf), (600, cd_stale)]
        analyze_frames(frames, verbose=True)
        out = capsys.readouterr().out
        assert "SKIP CD" in out and "seen_wf_ts" in out

    # -- SKIP CD: stale game timestamp (lines 408-415) ----------------------

    def test_verbose_skip_cd_stale(self, capsys):
        # WF accepted first (no preceding CD), then a CD whose game time is
        # well before the WF's game time → stale CD skipped.
        wf = "[01:50:00] Rima\n> WF"
        cd_old = "[01:30:00] P\n> CD"   # game time 01:30 < 01:50 - 5s tolerance
        frames = [(100, wf), (200, cd_old)]
        analyze_frames(frames, verbose=True)
        out = capsys.readouterr().out
        assert "SKIP CD" in out and "stale" in out

    # -- SKIP WF: near a CD ts (line 437) -----------------------------------

    def test_verbose_skip_wf_near_cd_ts(self, capsys):
        # WF carry_ts matches the CD carry_ts exactly → blocked
        cd = "[01:26:00] 3brand\n> CD"
        fake_wf = "[01:26:00] 3brand\n> WF"
        legit_wf = "[01:51:00] Rima\n> WF"
        frames = [(2004, cd), (2702, fake_wf), (2788, legit_wf)]
        analyze_frames(frames, verbose=True)
        out = capsys.readouterr().out
        assert "SKIP WF" in out and "near a CD ts" in out

    # -- SKIP WF: video-time gap too small (line 444) -----------------------

    def test_verbose_skip_wf_gap_too_small(self, capsys):
        cd = "[01:07:49] Player\n> CD"
        wf = "[01:17:39] Rima\n> WF"
        # WF fires 1s after CD (gap < _MIN_CD_WF_GAP)
        frames = [(100, cd), (101, wf)]
        analyze_frames(frames, verbose=True)
        out = capsys.readouterr().out
        assert "SKIP WF" in out and "gap=" in out

    # -- SKIP WF: stale (wf game time < max CD game time) (lines 453) ------

    def test_verbose_skip_wf_stale(self, capsys):
        cd = "[01:41:49] P\n> CD"
        wf_stale = "[01:31:33] Rima\n> WF"  # game time before CD
        frames = [(2314, cd), (2376, wf_stale)]
        analyze_frames(frames, verbose=True)
        out = capsys.readouterr().out
        assert "SKIP WF" in out and "stale" in out

    # -- SKIP WF: backward game timestamp (lines 463-470) ------------------

    def test_verbose_skip_wf_backward(self, capsys):
        cd = "[01:07:49] P\n> CD"
        wf1 = "[01:50:00] Rima\n> WF"
        wf_back = "[01:45:00] Rima\n> WF"  # game time < last WF time
        frames = [(100, cd), (500, wf1), (600, wf_back)]
        analyze_frames(frames, verbose=True)
        out = capsys.readouterr().out
        assert "SKIP WF" in out and "backward" in out

    # -- monotonic fallback verbose (line 486) ------------------------------

    def test_verbose_monotonic_fallback(self, capsys):
        # No EVE timestamps → monotonic fallback path; verbose prints CD/WF counts
        frames = [(10, "Player > cd"), (20, "Player > cd\nPlayer > wf")]
        analyze_frames(frames, verbose=True)
        out = capsys.readouterr().out
        assert "CD=" in out


# ---------------------------------------------------------------------------
# Tournament mode
# ---------------------------------------------------------------------------

class TestTournamentMode:
    def test_repeated_cd_detections_collapse_to_earliest(self):
        # "30 seconds until match start" stays visible in chat for ~30 s.
        # OCR re-detects it on subsequent frames with garbled timestamps that
        # are >3 s apart and therefore pass the fuzzy dedup check, producing
        # multiple entries in cd_timestamps (e.g. [100, 106, 112]).
        #
        # Before the fix (_CD_MERGE_WINDOW=5): these stay separate because each
        # gap (6 s) exceeds the 5 s window.  pair_cd_wf uses the most-recent CD
        # (112), so the clip misses the first ~12 s of the countdown.
        #
        # After the fix (tournament merge window=40): all three collapse into
        # the earliest detection (100), so the clip starts at the first
        # appearance of "30 seconds until match start".
        cd_first    = "[17:08:43] EVE System > 30 seconds until match start..."
        cd_redetect = "[17:08:49] EVE System > 30 seconds until match start..."
        cd_late     = "[17:08:55] EVE System > 30 seconds until match start..."
        wf          = "[17:13:34] EVE System > Match completed!"
        frames = [
            (100, cd_first),     # first detection of the countdown message
            (106, cd_redetect),  # re-detection with garbled timestamp (+6 s)
            (112, cd_late),      # another re-detection (+12 s garble)
            (400, wf),
        ]
        cd_times, wf_times = analyze_frames(frames, tournament_mode=True)
        assert cd_times == [100], f"expected single earliest CD; got {cd_times}"
        assert wf_times == [400]
        pairs = pair_cd_wf(cd_times, wf_times)
        assert pairs == [(100, 400)], f"clip must start at first detection; got {pairs}"
