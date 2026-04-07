"""
test_0329_session.py — Log-parsing tests for the six test cases in
src/tests/resources/.

Each case has a UTF-16 EVE chat log (caseN.txt) with a known t0 (EVE
game-time seconds at video second 0).  Tests verify that parse_chat_logs
+ pair_cd_wf produce the correct CD times, WF times, and clip pairs.

These tests do not require video files or OCR; they run on every pytest
invocation as long as the resource files are present.

Case summaries (from test-cases.md):
  Case 1 — Clear CD → WF:               CDs=[4],        WFs=[38],           clips=[(4, 38)]
  Case 2 — Clear CD → GF:               CDs=[5],        WFs=[35],           clips=[(5, 35)]
  Case 3 — CD→WF→CD→GF:                 CDs=[8, 59],    WFs=[41, 95],       clips=[(8, 41), (59, 95)]
  Case 4 — WF→CD→WF (pre-rec WF):       CDs=[9],        WFs=[33],           clips=[(9, 33)]
  Case 5 — Multiple CD/WF/GF variants:  CDs=[6..57]×8,  WFs=[65..99]×7,    clips=[(57, 65)]
  Case 6 — CD→WF with random text:      CDs=[22, 73],   WFs=[109, 113],     clips=[(73, 109)]
"""

import os
import sys

from chat_log_parser import parse_chat_logs
from chat_analyzer import pair_cd_wf

_RES = os.path.join(os.path.dirname(__file__), "resources")


def _log(n: int) -> str:
    return os.path.join(_RES, f"case{n}.txt")


def _parse(n: int, t0: int, duration: int):
    return parse_chat_logs([_log(n)], t0, duration)


# ---------------------------------------------------------------------------
# Case 1 — Clear CD to WF
# ---------------------------------------------------------------------------

class TestCase1LogParsing:
    """case1.txt: CD at 4 s, wf at 38 s → 1 clip (4, 38)."""

    _T0, _DUR = 63835, 50
    _CDS = [4]
    _WFS = [38]
    _CLIPS = [(4, 38)]

    def test_cd_count(self):
        cds, _ = _parse(1, self._T0, self._DUR)
        assert len(cds) == 1, f"Expected 1 CD, got {len(cds)}: {cds}"

    def test_wf_count(self):
        _, wfs = _parse(1, self._T0, self._DUR)
        assert len(wfs) == 1, f"Expected 1 WF, got {len(wfs)}: {wfs}"

    def test_cd_timestamps(self):
        cds, _ = _parse(1, self._T0, self._DUR)
        for t in self._CDS:
            assert t in cds, f"CD at {t} not found; cds={cds}"

    def test_wf_timestamps(self):
        _, wfs = _parse(1, self._T0, self._DUR)
        for t in self._WFS:
            assert t in wfs, f"WF at {t} not found; wfs={wfs}"

    def test_clips(self):
        cds, wfs = _parse(1, self._T0, self._DUR)
        pairs = pair_cd_wf(cds, wfs)
        assert pairs == self._CLIPS, f"Expected {self._CLIPS}, got {pairs}"


# ---------------------------------------------------------------------------
# Case 2 — Clear CD to GF
# ---------------------------------------------------------------------------

class TestCase2LogParsing:
    """case2.txt: CD at 5 s, gf at 35 s → 1 clip (5, 35).  GF counts as WF."""

    _T0, _DUR = 64196, 50
    _CDS = [5]
    _WFS = [35]
    _CLIPS = [(5, 35)]

    def test_cd_count(self):
        cds, _ = _parse(2, self._T0, self._DUR)
        assert len(cds) == 1, f"Expected 1 CD, got {len(cds)}: {cds}"

    def test_wf_count(self):
        _, wfs = _parse(2, self._T0, self._DUR)
        assert len(wfs) == 1, f"Expected 1 WF, got {len(wfs)}: {wfs}"

    def test_gf_recognised_as_wf(self):
        """'gf' as first word must be treated as WF."""
        _, wfs = _parse(2, self._T0, self._DUR)
        assert self._WFS[0] in wfs, (
            f"GF at {self._WFS[0]} not recognised as WF; wfs={wfs}"
        )

    def test_clips(self):
        cds, wfs = _parse(2, self._T0, self._DUR)
        pairs = pair_cd_wf(cds, wfs)
        assert pairs == self._CLIPS, f"Expected {self._CLIPS}, got {pairs}"


# ---------------------------------------------------------------------------
# Case 3 — CD to WF to CD to GF
# ---------------------------------------------------------------------------

class TestCase3LogParsing:
    """case3.txt: 2 CDs (8, 59 s), 2 WFs (41, 95 s) → 2 clips."""

    _T0, _DUR = 64342, 110
    _CDS = [8, 59]
    _WFS = [41, 95]
    _CLIPS = [(8, 41), (59, 95)]

    def test_cd_count(self):
        cds, _ = _parse(3, self._T0, self._DUR)
        assert len(cds) == 2, f"Expected 2 CDs, got {len(cds)}: {cds}"

    def test_wf_count(self):
        _, wfs = _parse(3, self._T0, self._DUR)
        assert len(wfs) == 2, f"Expected 2 WFs, got {len(wfs)}: {wfs}"

    def test_cd_timestamps(self):
        cds, _ = _parse(3, self._T0, self._DUR)
        for t in self._CDS:
            assert t in cds, f"CD at {t} not found; cds={cds}"

    def test_wf_timestamps(self):
        _, wfs = _parse(3, self._T0, self._DUR)
        for t in self._WFS:
            assert t in wfs, f"WF at {t} not found; wfs={wfs}"

    def test_clips(self):
        cds, wfs = _parse(3, self._T0, self._DUR)
        pairs = pair_cd_wf(cds, wfs)
        assert pairs == self._CLIPS, f"Expected {self._CLIPS}, got {pairs}"


# ---------------------------------------------------------------------------
# Case 4 — WF to CD to WF (initial WF before recording)
# ---------------------------------------------------------------------------

class TestCase4LogParsing:
    """case4.txt: pre-recording wf excluded; CD at 9, wf at 33 → 1 clip (9, 33)."""

    _T0, _DUR = 64548, 50
    _CDS = [9]
    _WFS = [33]
    _CLIPS = [(9, 33)]

    def test_cd_count(self):
        cds, _ = _parse(4, self._T0, self._DUR)
        assert len(cds) == 1, f"Expected 1 CD, got {len(cds)}: {cds}"

    def test_wf_count(self):
        """Pre-recording WF (video_sec < 0) must be excluded; only WF at 33 s."""
        _, wfs = _parse(4, self._T0, self._DUR)
        assert len(wfs) == 1, f"Expected 1 WF, got {len(wfs)}: {wfs}"

    def test_pre_recording_wf_excluded(self):
        _, wfs = _parse(4, self._T0, self._DUR)
        assert all(t >= 0 for t in wfs), f"Negative WF time found: {wfs}"

    def test_clips(self):
        cds, wfs = _parse(4, self._T0, self._DUR)
        pairs = pair_cd_wf(cds, wfs)
        assert pairs == self._CLIPS, f"Expected {self._CLIPS}, got {pairs}"


# ---------------------------------------------------------------------------
# Case 5 — Multiple CD / WF / GF Variations
# ---------------------------------------------------------------------------

class TestCase5LogParsing:
    """case5.txt: 8 CD variants, 7 WF/GF variants → 1 clip (57, 65)."""

    _T0, _DUR = 64684, 110
    _CDS = [6, 13, 22, 28, 35, 43, 52, 57]
    _WFS = [65, 70, 76, 84, 90, 93, 99]
    _CLIPS = [(57, 65)]

    def test_cd_count(self):
        cds, _ = _parse(5, self._T0, self._DUR)
        assert len(cds) == 8, f"Expected 8 CDs, got {len(cds)}: {cds}"

    def test_wf_count(self):
        _, wfs = _parse(5, self._T0, self._DUR)
        assert len(wfs) == 7, f"Expected 7 WFs, got {len(wfs)}: {wfs}"

    def test_all_cd_timestamps(self):
        """All 8 CD spelling variants (CD, cd, --CD--, ***** CD ****** etc.) must be detected."""
        cds, _ = _parse(5, self._T0, self._DUR)
        for t in self._CDS:
            assert t in cds, f"CD at {t} not found; cds={cds}"

    def test_all_wf_timestamps(self):
        """All 7 WF/GF variants (WF, wf, wf!, wf wf, GF, gf, gf!) must be detected."""
        _, wfs = _parse(5, self._T0, self._DUR)
        for t in self._WFS:
            assert t in wfs, f"WF at {t} not found; wfs={wfs}"

    def test_clips(self):
        """Only (57, 65): the last CD before the first WF; subsequent WFs have no CD."""
        cds, wfs = _parse(5, self._T0, self._DUR)
        pairs = pair_cd_wf(cds, wfs)
        assert pairs == self._CLIPS, f"Expected {self._CLIPS}, got {pairs}"


# ---------------------------------------------------------------------------
# Case 6 — CD to WF Surrounded by Random Text
# ---------------------------------------------------------------------------

class TestCase6LogParsing:
    """case6.txt: CDs at 22, 73; WFs at 109, 113; embedded wf/WF not detected → 1 clip."""

    _T0, _DUR = 64914, 130
    _CDS = [22, 73]
    _WFS = [109, 113]
    _CLIPS = [(73, 109)]

    def test_cd_count(self):
        cds, _ = _parse(6, self._T0, self._DUR)
        assert len(cds) == 2, f"Expected 2 CDs, got {len(cds)}: {cds}"

    def test_wf_count(self):
        """Mid-sentence wf/WF must not be counted; only standalone wf at 109, 113 s."""
        _, wfs = _parse(6, self._T0, self._DUR)
        assert len(wfs) == 2, f"Expected 2 WFs, got {len(wfs)}: {wfs}"

    def test_cd_timestamps(self):
        cds, _ = _parse(6, self._T0, self._DUR)
        for t in self._CDS:
            assert t in cds, f"CD at {t} not found; cds={cds}"

    def test_wf_timestamps(self):
        _, wfs = _parse(6, self._T0, self._DUR)
        for t in self._WFS:
            assert t in wfs, f"WF at {t} not found; wfs={wfs}"

    def test_embedded_wf_not_detected(self):
        """Messages where wf/WF is not the first word must not trigger WF detection."""
        _, wfs = _parse(6, self._T0, self._DUR)
        assert 104 not in wfs, f"Embedded 'wf' at 104 s falsely detected; wfs={wfs}"
        assert 105 not in wfs, f"Embedded 'WF' at 105 s falsely detected; wfs={wfs}"

    def test_clips(self):
        """CD at 22 s is abandoned when CD at 73 s arrives; clip is (73, 109)."""
        cds, wfs = _parse(6, self._T0, self._DUR)
        pairs = pair_cd_wf(cds, wfs)
        assert pairs == self._CLIPS, f"Expected {self._CLIPS}, got {pairs}"
