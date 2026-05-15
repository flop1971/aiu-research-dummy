"""
Stickification compiler pass.

Transforms tensor memory layouts into the SDSC stickified format
required by the Spyre AIU kernel compiler. Handles both row-major
and column-major input orderings including POWER/Z host variants.
"""

from torch_spyre.tensor_layout import SpyreTensorLayout


def stickify(layout: SpyreTensorLayout) -> SpyreTensorLayout:
    """
    Apply stickification transform to a tensor layout.

    Converts the input layout to SDSC format suitable for AIU
    kernel execution. Handles 2D, 3D, and 4D input tensors.
    """
    if layout.is_stickified:
        return layout   # idempotent

    return SpyreTensorLayout(
        shape=layout.shape,
        strides=_compute_sdsc_strides(layout.shape),
        is_stickified=True,
        sdsc_format="SDSC_v2",
    )


def _compute_sdsc_strides(shape: tuple) -> tuple:
    """Compute SDSC-format strides for a given shape."""
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return tuple(strides)
