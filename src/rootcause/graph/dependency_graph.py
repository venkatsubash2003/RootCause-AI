
import networkx as nx


def build_dependency_graph(records):
    """
    Build a directed service dependency graph
    from normalized OpenTelemetry spans.
    """

    graph = nx.DiGraph()

    # Build span lookup within each trace
    span_lookup = {}

    for record in records:
        trace_id = record["trace_id"]
        span_id = record["span_id"]

        if not trace_id or not span_id:
            continue

        span_lookup[(trace_id, span_id)] = record

        service = record.get("service")

        if service:
            graph.add_node(service)

    # Find relationships between services
    for record in records:

        trace_id = record.get("trace_id")
        parent_id = record.get("parent_span_id")
        child_service = record.get("service")

        if not trace_id or not parent_id:
            continue

        if not child_service:
            continue

        parent = span_lookup.get(
            (trace_id, parent_id)
        )

        if parent is None:
            continue

        parent_service = parent.get("service")

        if not parent_service:
            continue

        # We only need cross-service dependencies
        if parent_service == child_service:
            continue

        if graph.has_edge(
            parent_service, child_service
        ):
            graph[parent_service][child_service][
                "call_count"
            ] += 1

        else:
            graph.add_edge(
                parent_service,
                child_service,
                call_count=1
            )

    return graph
