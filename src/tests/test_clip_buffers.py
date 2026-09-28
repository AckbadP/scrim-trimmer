"""
test_clip_buffers.py — Unit tests for main.apply_buffers() and main._buffers_for().

Pure-function tests; no video/tesseract/ffmpeg dependencies.
"""

import types

from main import apply_buffers, _buffers_for


class TestApplyBuffers:
    def test_no_buffers_is_noop(self):
        pairs = [(10, 20), (30, 45)]
        assert apply_buffers(pairs, 0, 0, duration=100) == pairs

    def test_positive_buffers_shift_outward(self):
        pairs = [(10, 20)]
        assert apply_buffers(pairs, -2, 3, duration=100) == [(8, 23)]

    def test_negative_end_buffer_shrinks_clip(self):
        pairs = [(10, 20)]
        assert apply_buffers(pairs, 0, -3, duration=100) == [(10, 17)]

    def test_start_clamped_to_zero(self):
        pairs = [(2, 20)]
        assert apply_buffers(pairs, -10, 0, duration=100) == [(0, 20)]

    def test_end_clamped_to_duration(self):
        pairs = [(80, 98)]
        assert apply_buffers(pairs, 0, 10, duration=100) == [(80, 100)]

    def test_collapsed_pair_is_dropped(self):
        pairs = [(10, 12), (30, 45)]
        # start buffer pushes the first pair's start past its end
        result = apply_buffers(pairs, 5, -5, duration=100)
        assert result == [(35, 40)]

    def test_multiple_pairs_all_shifted(self):
        pairs = [(10, 20), (30, 40), (50, 60)]
        assert apply_buffers(pairs, 1, -1, duration=100) == [(11, 19), (31, 39), (51, 59)]


class TestBuffersFor:
    def _args(self, **overrides):
        return types.SimpleNamespace(**overrides)

    def test_normal_mode_defaults_to_zero(self):
        args = self._args(tournament_match=False)
        assert _buffers_for(args) == (0, 0)

    def test_tournament_mode_defaults_preserve_old_plus_10_behavior(self):
        # Regression: main.py used to hardcode wf_times = [min(t + 10, duration)...]
        # in tournament mode. With no buffer attrs set (e.g. an older/minimal args
        # namespace), _buffers_for must still yield a +10s end buffer.
        args = self._args(tournament_match=True)
        assert _buffers_for(args) == (0, 10)

    def test_normal_mode_reads_explicit_values(self):
        args = self._args(tournament_match=False, buffer_start=-5, buffer_end=3)
        assert _buffers_for(args) == (-5, 3)

    def test_tournament_mode_reads_explicit_values(self):
        args = self._args(
            tournament_match=True,
            tournament_buffer_start=2,
            tournament_buffer_end=0,
        )
        assert _buffers_for(args) == (2, 0)

    def test_tournament_mode_ignores_normal_buffers(self):
        args = self._args(
            tournament_match=True,
            buffer_start=99,
            buffer_end=99,
            tournament_buffer_start=1,
            tournament_buffer_end=2,
        )
        assert _buffers_for(args) == (1, 2)
