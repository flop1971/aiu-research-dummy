"""Tests for SpyreDevice host architecture detection."""

import pytest
from unittest.mock import patch
from torch_spyre.runtime.device import SpyreDevice


def test_device_creation():
    device = SpyreDevice(device_id=0)
    assert device.device_id == 0
    assert device.host_arch in ("x86", "power", "z", "unknown")


@patch("platform.machine", return_value="x86_64")
def test_x86_detection(mock_machine):
    device = SpyreDevice()
    assert device.is_x86_host
    assert not device.is_power_host
    assert not device.is_z_host


@patch("platform.machine", return_value="ppc64le")
def test_power_detection(mock_machine):
    device = SpyreDevice()
    assert device.is_power_host
    assert not device.is_x86_host


@patch("platform.machine", return_value="s390x")
def test_z_detection(mock_machine):
    device = SpyreDevice()
    assert device.is_z_host
    assert not device.is_x86_host


def test_repr():
    device = SpyreDevice(device_id=2)
    assert "SpyreDevice" in repr(device)
    assert "id=2" in repr(device)
