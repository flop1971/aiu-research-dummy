"""
Spyre runtime allocator.

Manages allocation of tensors in Spyre AIU local memory with
NUMA-aware placement for multi-card configurations.
"""

import threading
from typing import Optional


class SpyreAllocator:
    """Thread-safe allocator for Spyre AIU tensor memory."""

    _lock = threading.Lock()
    _allocations: dict = {}

    @classmethod
    def allocate(cls, size_bytes: int, device_id: int = 0) -> int:
        """Allocate a buffer on the specified Spyre card. Returns handle."""
        with cls._lock:
            handle = id(object())
            cls._allocations[handle] = {
                "size": size_bytes,
                "device_id": device_id,
            }
            return handle

    @classmethod
    def free(cls, handle: int) -> None:
        """Release a previously allocated buffer."""
        with cls._lock:
            cls._allocations.pop(handle, None)

    @classmethod
    def evict_all(cls) -> None:
        """Evict all allocations — used in scratchpad planning pass."""
        with cls._lock:
            cls._allocations.clear()
