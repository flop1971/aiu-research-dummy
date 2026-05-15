"""
SpyreDevice — device abstraction for the Spyre AIU accelerator.

Detects host CPU architecture (x86, POWER, Z) and configures
the device interface appropriately for each host platform.
"""

import platform
from typing import Optional


class SpyreDevice:
    """Represents a single Spyre AIU accelerator card."""

    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.host_arch = self._detect_host_arch()

    @staticmethod
    def _detect_host_arch() -> str:
        """Detect host CPU architecture for platform-specific config."""
        machine = platform.machine().lower()
        if machine in ("x86_64", "amd64", "i386", "i686"):
            return "x86"
        elif machine.startswith("ppc") or machine.startswith("powerpc"):
            return "power"
        elif machine.startswith("s390"):
            return "z"
        return "unknown"

    @property
    def is_x86_host(self) -> bool:
        return self.host_arch == "x86"

    @property
    def is_power_host(self) -> bool:
        return self.host_arch == "power"

    @property
    def is_z_host(self) -> bool:
        return self.host_arch == "z"

    def __repr__(self) -> str:
        return f"SpyreDevice(id={self.device_id}, host={self.host_arch})"
