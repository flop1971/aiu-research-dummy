"""Unit tests for SpyreTensorLayout."""

import pytest
from torch_spyre.tensor_layout import SpyreTensorLayout


@pytest.fixture
def simple_layout():
    return SpyreTensorLayout(shape=(4, 8), strides=(8, 1))


def test_is_contiguous(simple_layout):
    assert simple_layout.is_contiguous()


def test_non_contiguous():
    layout = SpyreTensorLayout(shape=(4, 8), strides=(16, 1))
    assert not layout.is_contiguous()


def test_hash_stability(simple_layout):
    """SpyreTensorLayout must be hashable for use as dict key."""
    d = {simple_layout: "value"}
    assert d[simple_layout] == "value"


def test_stickified_flag():
    layout = SpyreTensorLayout(
        shape=(4, 8), strides=(8, 1), is_stickified=True
    )
    assert layout.is_stickified


@pytest.mark.parametrize("shape,expected", [
    ((1, 1), True),
    ((4, 8), True),
    ((4, 8, 16), True),
])
def test_contiguous_shapes(shape, expected):
    strides = _row_major_strides(shape)
    layout = SpyreTensorLayout(shape=shape, strides=strides)
    assert layout.is_contiguous() == expected


def _row_major_strides(shape):
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return tuple(strides)
