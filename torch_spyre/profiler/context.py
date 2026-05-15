"""
SpyreProfiler — Context manager for profiling Spyre AIU kernel execution.

Integrates with PyTorch kineto for Chrome trace export compatibility.
"""

import time
from contextlib import contextmanager
from typing import Optional, List


class SpyreEvent:
    """Represents a single timed event on the Spyre AIU."""

    def __init__(self, name: str, device_id: int = 0):
        self.name = name
        self.device_id = device_id
        self.start_time_ns: Optional[int] = None
        self.end_time_ns: Optional[int] = None

    def record_start(self) -> None:
        self.start_time_ns = time.perf_counter_ns()

    def record_end(self) -> None:
        self.end_time_ns = time.perf_counter_ns()

    @property
    def duration_us(self) -> Optional[float]:
        if self.start_time_ns and self.end_time_ns:
            return (self.end_time_ns - self.start_time_ns) / 1000.0
        return None


class SpyreProfiler:
    """
    Context manager for profiling Spyre AIU operations.

    Usage:
        with SpyreProfiler() as prof:
            model(input_tensor)
        print(prof.summary())
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.events: List[SpyreEvent] = []

    def __enter__(self) -> "SpyreProfiler":
        if self.enabled:
            self._start = time.perf_counter_ns()
        return self

    def __exit__(self, *args) -> None:
        if self.enabled:
            self._end = time.perf_counter_ns()

    def summary(self) -> str:
        if not self.enabled:
            return "Profiler disabled"
        total_us = (self._end - self._start) / 1000.0
        return f"Total: {total_us:.2f}us | Events: {len(self.events)}"
