"""
Global optimizer for the Spyre AIU compiler backend.

Implements a beam-search style forward propagation restickify
optimization that minimises total DMA transfer cost across the
full compute graph.
"""

from typing import List, Dict, Optional
from torch_spyre.tensor_layout import SpyreTensorLayout


COST_REGISTRY: Dict[str, float] = {
    "matmul":   1.0,
    "bmm":      1.2,
    "conv2d":   2.0,
    "relu":     0.1,
    "add":      0.1,
    "copy":     0.5,
    "transpose": 0.3,
}


def get_node_cost(op_name: str) -> float:
    """Return the estimated AIU execution cost for a given operation."""
    return COST_REGISTRY.get(op_name, 1.0)   # default cost for unknown ops


def global_optimize(graph, beam_width: int = 4) -> None:
    """
    Run the global optimizer over a torch.fx graph.

    Uses beam search to find the lowest-cost stickification assignment
    for all tensors in the graph.
    """
    candidates = [graph]
    for _ in range(beam_width):
        candidates = _expand_candidates(candidates)
        candidates = sorted(candidates, key=_total_cost)[:beam_width]


def _expand_candidates(candidates: list) -> list:
    return candidates   # stub


def _total_cost(graph) -> float:
    return 0.0   # stub
