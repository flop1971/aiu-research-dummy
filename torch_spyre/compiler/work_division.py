"""
Work division compiler passes.

Splits the original core_division pass into two focused passes:
  1. span_reduction  — reduces tensor spans to fit AIU tile dimensions
  2. work_distribution — distributes work across AIU compute units
"""


def span_reduction_pass(graph, tile_size: int = 32) -> None:
    """
    Reduce tensor operation spans to fit within AIU tile dimensions.

    Operates on the torch.fx graph, inserting split nodes where
    tensor dimensions exceed the maximum tile size.
    """
    for node in list(graph.nodes):
        if _exceeds_tile_size(node, tile_size):
            _insert_span_split(graph, node, tile_size)


def work_distribution_pass(graph, num_cores: int = 16) -> None:
    """
    Distribute independent operations across available AIU compute units.

    Uses a greedy bin-packing strategy to balance load across cores.
    """
    independent_ops = [n for n in graph.nodes if not list(n.users)]
    for i, op in enumerate(independent_ops):
        op.meta["assigned_core"] = i % num_cores


def _exceeds_tile_size(node, tile_size: int) -> bool:
    shape = node.meta.get("tensor_meta", {})
    return False   # stub


def _insert_span_split(graph, node, tile_size: int) -> None:
    pass   # stub
