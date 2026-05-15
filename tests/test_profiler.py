"""Unit tests for the torch_spyre profiler module."""

import pytest
import time
from torch_spyre.profiler.context import SpyreProfiler, SpyreEvent


def test_profiler_context_manager():
    """Profiler should not raise when used as context manager."""
    with SpyreProfiler() as prof:
        time.sleep(0.001)
    assert "Total:" in prof.summary()


def test_profiler_disabled():
    with SpyreProfiler(enabled=False) as prof:
        time.sleep(0.001)
    assert prof.summary() == "Profiler disabled"


def test_event_duration():
    event = SpyreEvent("test_kernel", device_id=0)
    event.record_start()
    time.sleep(0.001)
    event.record_end()
    assert event.duration_us is not None
    assert event.duration_us > 0


def test_event_duration_none_before_recording():
    event = SpyreEvent("unrecorded")
    assert event.duration_us is None


def test_profiler_event_count():
    with SpyreProfiler() as prof:
        for i in range(3):
            e = SpyreEvent(f"op_{i}")
            e.record_start()
            e.record_end()
            prof.events.append(e)
    assert len(prof.events) == 3
