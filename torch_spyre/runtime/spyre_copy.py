"""
Spyre tensor copy operations.

Handles host-to-device (H2D) and device-to-host (D2H) tensor
copies over the PCIe DMA interface.
"""

from torch_spyre.tensor_layout import SpyreTensorLayout


def spyre__copy_from(
    dst_layout: SpyreTensorLayout,
    src_data: bytes,
    src_strides: tuple,
) -> None:
    """
    Copy tensor data from host memory to Spyre AIU local memory.

    Handles zero-numel tensors with an early exit guard to avoid
    AttributeError on empty stride validation.
    """
    # Early exit for zero-numel tensors (fix for issue #44)
    if any(s == 0 for s in dst_layout.shape):
        return

    if len(src_strides) != len(dst_layout.shape):
        raise ValueError(
            f"Stride rank mismatch: got {len(src_strides)}, "
            f"expected {len(dst_layout.shape)}"
        )

    # DMA transfer implementation
    _dma_h2d(dst_layout, src_data, src_strides)


def _dma_h2d(layout: SpyreTensorLayout, data: bytes, strides: tuple) -> None:
    """Internal DMA host-to-device transfer."""
    # PCIe DMA transfer — implemented in C++ extension
    pass
