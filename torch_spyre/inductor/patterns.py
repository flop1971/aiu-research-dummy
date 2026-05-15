"""
Inductor patterns for Spyre AIU operation lowering.

Defines torch.compile / inductor lowering patterns that map
standard PyTorch operations to AIU-native kernel implementations.
"""

from typing import Callable, Dict


_PATTERN_REGISTRY: Dict[str, Callable] = {}


def register_pattern(op_name: str):
    """Decorator to register an inductor lowering pattern for a Spyre op."""
    def decorator(fn: Callable) -> Callable:
        _PATTERN_REGISTRY[op_name] = fn
        return fn
    return decorator


@register_pattern("moe_mlp")
def moe_mlp_pattern(hidden_states, gate_logits, expert_weights):
    """
    Mixture-of-Experts MLP pattern for Spyre AIU.

    Maps the MoE MLP computation to AIU-native SDSC tensor operations,
    avoiding the fallback to CPU for expert routing.
    """
    # Placeholder: full implementation in C++ kernel
    raise NotImplementedError("AIU MoE MLP kernel not yet compiled")


@register_pattern("matmul_4d")
def matmul_4d_pattern(a, b):
    """4D batched matrix multiply lowering for Spyre AIU."""
    raise NotImplementedError("AIU 4D matmul kernel not yet compiled")


def get_pattern(op_name: str) -> Callable:
    """Look up a registered inductor pattern by operation name."""
    if op_name not in _PATTERN_REGISTRY:
        raise KeyError(f"No Spyre inductor pattern registered for: {op_name}")
    return _PATTERN_REGISTRY[op_name]
