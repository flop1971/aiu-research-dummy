"""
Scratchpad planning pass.

Determines optimal allocation of intermediate tensors into the
Spyre AIU scratchpad (LX) memory to minimise DMA transfers
between host and card during kernel execution.
"""

from typing import List, Dict
from torch_spyre.tensor_layout import SpyreTensorLayout


class ScratchpadPlan:
    """Represents an allocation plan for LX scratchpad memory."""

    def __init__(self, capacity_bytes: int):
        self.capacity_bytes = capacity_bytes
        self.allocations: Dict[str, int] = {}
        self._used = 0

    def can_fit(self, size_bytes: int) -> bool:
        return (self._used + size_bytes) <= self.capacity_bytes

    def allocate(self, tensor_id: str, size_bytes: int) -> bool:
        if not self.can_fit(size_bytes):
            return False
        self.allocations[tensor_id] = size_bytes
        self._used += size_bytes
        return True

    @property
    def utilisation(self) -> float:
        return self._used / self.capacity_bytes if self.capacity_bytes else 0.0


def plan_scratchpad(
    layouts: List[SpyreTensorLayout],
    capacity_bytes: int = 64 * 1024 * 1024,
) -> ScratchpadPlan:
    """
    Compute a scratchpad allocation plan for a list of tensor layouts.
    Uses a greedy first-fit strategy.
    """
    plan = ScratchpadPlan(capacity_bytes)
    for i, layout in enumerate(layouts):
        size = sum(layout.strides[0] * layout.shape[0] * 2)   # fp16 estimate
        plan.allocate(f"tensor_{i}", size)
    return plan
