
import csv
from pathlib import Path

import matplotlib.pyplot as plt

from dependency_graph import build_dependency_graph


def load_spans(csv_path):
    with open(csv_path, newline="") as file:
        return list(csv.DictReader(file))


def main():
    csv_path = Path(
        "data/processed/checkout_spans.csv"
    )

    records = load_spans(csv_path)

    graph = build_dependency_graph(records)

    print("\nService Dependency Graph")
    print("=" * 45)

    print("Services:", graph.number_of_nodes())
    print("Dependencies:", graph.number_of_edges())

    print("\nObserved service relationships:")

    for source, target, data in graph.edges(
        data=True
    ):
        print(
            f"{source} -> {target} "
            f"(observed spans: {data['call_count']})"
        )


import matplotlib.pyplot as plt
import networkx as nx


def visualize_graph(graph):

    if graph.number_of_nodes() == 0:
        print("No services to visualize")
        return

    plt.figure(figsize=(11, 7))

    positions = nx.spring_layout(
        graph,
        seed=42
    )

    nx.draw_networkx(
        graph,
        positions,
        with_labels=True,
        node_size=2300,
        font_size=10,
        arrows=True,
        arrowsize=18
    )

    plt.title("RootCause AI - Service Dependencies")
    plt.axis("off")
    plt.tight_layout()

    plt.savefig(
        "data/processed/service_graph.png",
        dpi=150
    )

    plt.show()


if __name__ == "__main__":
    main()
    visualize_graph(graph)
