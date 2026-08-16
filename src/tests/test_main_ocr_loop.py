"""
test_main_ocr_loop.py — Unit tests for the OCR concurrency loop in main.run().

Unlike test_main_ocr_pipeline.py, these tests mock extract_frames,
run_ocr_on_region, and friends so they run fast, deterministically, and
without needing real video/tesseract/ffmpeg. They target two regressions:

  - A semaphore permit leak: if an OCR worker raised, sem.release() was
    skipped, and enough failures would deadlock the producer forever at
    sem.acquire().
  - Cancel being a no-op during OCR: the executor was shut down with
    wait=True (no cancel_futures), so Cancel silently drained the entire
    queued backlog instead of aborting promptly.
"""

import threading
import types

import pytest

import main as main_module


def _make_args(video, output, **overrides):
    ns = types.SimpleNamespace(
        video=video,
        output=output,
        chat_region=[0.0, 0.35, 0.15, 1.0],
        threads=2,
        ram_cap_gb=1,
        verbose=False,
        chat_logs=None,
        t0=None,
        chapters_dir=None,
        run_without_ffmpeg=False,
        force_ocr=False,
        force_unfinalized=True,  # skip the check_source_video preflight
    )
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


def _fake_extract_frames_factory(n_frames, on_yield=None):
    def _fake_extract_frames(video_path):
        import numpy as np
        # Big enough that crop_chat_region's default fractions ([0.0, 0.35,
        # 0.15, 1.0]) produce a non-empty, non-degenerate region.
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        for i in range(n_frames):
            yield i, frame
            if on_yield is not None:
                on_yield(i)
    return _fake_extract_frames


class TestOcrLoopFailSafety:
    def test_all_ocr_failures_does_not_deadlock(self, monkeypatch, tmp_path):
        # Every frame's OCR raises. Before the fix, sem.release() was
        # skipped whenever fut.result() raised, so the producer would
        # eventually deadlock forever at sem.acquire() instead of finishing.
        n_frames = 20

        def _always_fails(region, timeout=120.0):
            raise RuntimeError("simulated OCR failure")

        monkeypatch.setattr(main_module, "_check_dependencies", lambda args: None)
        monkeypatch.setattr(main_module, "extract_frames", _fake_extract_frames_factory(n_frames))
        monkeypatch.setattr(main_module, "get_video_duration", lambda path: float(n_frames))
        monkeypatch.setattr(main_module, "run_ocr_on_region", _always_fails)

        video = tmp_path / "fake.mp4"
        video.write_bytes(b"0")
        args = _make_args(str(video), str(tmp_path / "out"))

        result = {}

        def _target():
            try:
                main_module.run(args)
            except SystemExit:
                pass  # "No CD->WF pairs found" is the expected clean exit here
            result["done"] = True

        t = threading.Thread(target=_target, daemon=True)
        t.start()
        t.join(timeout=15)
        assert result.get("done"), (
            "OCR loop did not complete within 15s — looks like the semaphore-leak "
            "regression (sem.release() skipped on a worker exception)."
        )

    def test_cancel_shuts_down_executor_without_waiting(self, monkeypatch, tmp_path):
        import concurrent.futures as cf

        shutdown_calls = []
        real_executor_cls = cf.ThreadPoolExecutor

        class _RecordingExecutor(real_executor_cls):
            def shutdown(self, wait=True, cancel_futures=False):
                shutdown_calls.append({"wait": wait, "cancel_futures": cancel_futures})
                return super().shutdown(wait=wait, cancel_futures=cancel_futures)

        monkeypatch.setattr(main_module, "ThreadPoolExecutor", _RecordingExecutor)

        cancel_event = threading.Event()

        def _set_cancel_after_third_frame(i):
            if i == 2:
                cancel_event.set()

        monkeypatch.setattr(main_module, "_check_dependencies", lambda args: None)
        monkeypatch.setattr(
            main_module, "extract_frames",
            _fake_extract_frames_factory(50, on_yield=_set_cancel_after_third_frame),
        )
        monkeypatch.setattr(main_module, "get_video_duration", lambda path: 50.0)
        monkeypatch.setattr(main_module, "run_ocr_on_region", lambda region, timeout=120.0: "")

        video = tmp_path / "fake.mp4"
        video.write_bytes(b"0")
        args = _make_args(str(video), str(tmp_path / "out"), cancel_event=cancel_event)

        with pytest.raises(main_module.CancelledError):
            main_module.run(args)

        assert shutdown_calls, "executor.shutdown() was never called"
        # The abort path must drop queued work rather than draining it — this
        # is what makes Cancel actually fast during OCR.
        assert shutdown_calls[-1] == {"wait": False, "cancel_futures": True}

    def test_normal_completion_waits_for_executor(self, monkeypatch, tmp_path):
        # Sanity check for the other branch of the same change: the
        # non-cancelled path must still wait for all submitted work so
        # results aren't collected before every OCR call has finished.
        import concurrent.futures as cf

        shutdown_calls = []
        real_executor_cls = cf.ThreadPoolExecutor

        class _RecordingExecutor(real_executor_cls):
            def shutdown(self, wait=True, cancel_futures=False):
                shutdown_calls.append({"wait": wait, "cancel_futures": cancel_futures})
                return super().shutdown(wait=wait, cancel_futures=cancel_futures)

        monkeypatch.setattr(main_module, "ThreadPoolExecutor", _RecordingExecutor)
        monkeypatch.setattr(main_module, "_check_dependencies", lambda args: None)
        monkeypatch.setattr(main_module, "extract_frames", _fake_extract_frames_factory(5))
        monkeypatch.setattr(main_module, "get_video_duration", lambda path: 5.0)
        monkeypatch.setattr(main_module, "run_ocr_on_region", lambda region, timeout=120.0: "")

        video = tmp_path / "fake.mp4"
        video.write_bytes(b"0")
        args = _make_args(str(video), str(tmp_path / "out"))

        with pytest.raises(SystemExit):  # no CD/WF found in empty OCR text
            main_module.run(args)

        assert shutdown_calls == [{"wait": True, "cancel_futures": False}]
