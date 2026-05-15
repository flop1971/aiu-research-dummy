"""Tests for the work division compiler passes."""

import pytest
from torch_spyre.compiler.work_division import (
    span_reduction_pass,
    work_distribution_pass,
)


class MockGraph:
    """Minimal torch.fx-like graph stub for testing."""
    def __init__(self):
        self.nodes = []


class MockNode:
    def __init__(self, name):
        self.name = name
        self.meta = {}
        self.users = []


def test_span_reduction_pass_empty_graph():
    """span_reduction_pass should handle empty graphs."""
    graph = MockGraph()
    span_reduction_pass(graph, tile_size=32)   # should not raise


def test_work_distribution_assigns_cores():
    graph = MockGraph()
    for i in range(8):
        node = MockNode(f"op_{i}")
        graph.nodes.append(node)
    work_distribution_pass(graph, num_cores=4)
    # All nodes without users get assigned
    unassigned = [n for n in graph.nodes if "assigned_core" not in n.meta]
    assert len(unassigned) == 0
