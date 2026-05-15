"""
SpyreTensorLayout — Tensor memory layout management for the Spyre AIU.

Handles SDSC (Stacked Data in Single Column) format and stickification
of tensors for efficient AIU memory access patterns.
"""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class SpyreTensorLayout:
    """Describes how a tensor is laid out in Spyre AIU local memory."""

    shape: Tuple[int, ...]
    strides: Tuple[int, ...]
    is_stickified: bool = False
    sdsc_format: Optional[str] = None

    def __hash__(self) -> int:
        return hash((self.shape, self.strides, self.is_stickified))

    def is_contiguous(self) -> bool:
        """Return True if tensor is contiguous in AIU memory."""
        expected = 1
        for dim, stride in zip(reversed(self.shape), reversed(self.strides)):
            if stride != expected:
                return False
            expected *= dim
        return True
