"""Tests for the Spyre runtime allocator."""

import pytest
from torch_spyre.runtime.allocator import SpyreAllocator


def test_allocate_returns_handle():
    handle = SpyreAllocator.allocate(1024, device_id=0)
    assert isinstance(handle, int)
    SpyreAllocator.free(handle)


def test_free_nonexistent_handle():
    """Freeing a non-existent handle should not raise."""
    SpyreAllocator.free(99999999)


def test_evict_all_clears_allocations():
    SpyreAllocator.allocate(512)
    SpyreAllocator.allocate(1024)
    SpyreAllocator.evict_all()
    assert len(SpyreAllocator._allocations) == 0


def test_allocate_multiple_devices():
    h0 = SpyreAllocator.allocate(256, device_id=0)
    h1 = SpyreAllocator.allocate(256, device_id=1)
    assert h0 != h1
    SpyreAllocator.free(h0)
    SpyreAllocator.free(h1)
