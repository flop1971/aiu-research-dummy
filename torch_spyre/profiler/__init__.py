"""
torch_spyre.profiler — Profiling support for Spyre AIU operations.

Provides PyTorch-compatible profiling context managers and event
timing APIs for Spyre kernel execution.
"""

from torch_spyre.profiler.context import SpyreProfiler
from torch_spyre.profiler.events import SpyreEvent

__all__ = ["SpyreProfiler", "SpyreEvent"]
