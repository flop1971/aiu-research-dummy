"""Tests for Spyre tensor copy operations."""

import pytest
from torch_spyre.tensor_layout import SpyreTensorLayout
from torch_spyre.runtime.spyre_copy import spyre__copy_from


def test_copy_empty_tensor_no_error():
    """Zero-numel tensors should return without error (regression test #44)."""
    layout = SpyreTensorLayout(shape=(0, 8), strides=(8, 1))
    spyre__copy_from(layout, b"", strides=(8, 1))   # should not raise


def test_copy_stride_rank_mismatch():
    """Mismatched stride rank should raise ValueError."""
    layout = SpyreTensorLayout(shape=(4, 8), strides=(8, 1))
    with pytest.raises(ValueError, match="Stride rank mismatch"):
        spyre__copy_from(layout, b"data", strides=(1,))   # wrong rank


def test_copy_valid_2d():
    """Valid 2D tensor copy should complete without error."""
    layout = SpyreTensorLayout(shape=(4, 8), strides=(8, 1))
    spyre__copy_from(layout, b"x" * 64, strides=(8, 1))
