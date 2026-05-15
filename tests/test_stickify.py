"""Tests for the stickification compiler pass."""

import pytest
from torch_spyre.tensor_layout import SpyreTensorLayout
from torch_spyre.compiler.stickify import stickify


def test_stickify_2d():
    layout = SpyreTensorLayout(shape=(4, 8), strides=(8, 1))
    result = stickify(layout)
    assert result.is_stickified
    assert result.sdsc_format == "SDSC_v2"


def test_stickify_idempotent():
    """Applying stickification twice should return same layout."""
    layout = SpyreTensorLayout(
        shape=(4, 8), strides=(8, 1), is_stickified=True
    )
    result = stickify(layout)
    assert result is layout


def test_stickify_4d():
    layout = SpyreTensorLayout(
        shape=(2, 4, 8, 16), strides=(512, 128, 16, 1)
    )
    result = stickify(layout)
    assert result.is_stickified
    assert len(result.strides) == 4
