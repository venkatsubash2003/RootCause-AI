
import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src")
)

from rootcause.graph.dependency_graph import (
    build_dependency_graph
)


def test_dependency_graph():

    records = [
        {
            "trace_id": "trace-1",
            "span_id": "span-1",
            "parent_span_id": None,
            "service": "frontend"
        },
        {
            "trace_id": "trace-1",
            "span_id": "span-2",
            "parent_span_id": "span-1",
            "service": "checkout"
        },
        {
            "trace_id": "trace-1",
            "span_id": "span-3",
            "parent_span_id": "span-2",
            "service": "cart"
        },
        {
            "trace_id": "trace-1",
            "span_id": "span-4",
            "parent_span_id": "span-2",
            "service": "payment"
        }
    ]

    graph = build_dependency_graph(records)

    assert graph.has_edge("frontend", "checkout")
    assert graph.has_edge("checkout", "cart")
    assert graph.has_edge("checkout", "payment")

    assert graph.number_of_nodes() == 4
    assert graph.number_of_edges() == 3

    assert (
        graph["checkout"]["cart"]["call_count"]
        == 1
    )
