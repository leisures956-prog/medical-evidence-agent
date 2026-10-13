"""Render the JSON evidence graph as a presentation-ready PNG.

This is an optional showcase script. The API itself only depends on the JSON
graph writer; install networkx and matplotlib only when you want this image.
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D


NODE_STYLES = {
    "Paper": {"color": "#2F6B9A", "shape": "o", "legend": "论文 Paper"},
    "Disease": {"color": "#D85C5C", "shape": "h", "legend": "疾病 Disease"},
    "Method": {"color": "#49A078", "shape": "s", "legend": "方法 Method"},
    "Dataset": {"color": "#8A63B8", "shape": "D", "legend": "数据集 Dataset"},
}

RELATION_NAMES = {
    "studies_disease": "研究疾病",
    "uses_method": "采用方法",
    "uses_dataset": "使用数据集",
}


def _short_label(label: str, width: int = 24) -> str:
    return textwrap.fill(str(label), width=width, break_long_words=False)


def _load_graph(path: Path) -> nx.DiGraph:
    payload = json.loads(path.read_text(encoding="utf-8"))
    graph = nx.DiGraph()

    for node in payload.get("nodes", []):
        graph.add_node(node["id"], **node)

    for edge in payload.get("edges", []):
        source = edge["source"]
        target = edge["target"]
        relation = edge["relation"]
        if graph.has_edge(source, target):
            graph[source][target]["relations"].append(relation)
        else:
            graph.add_edge(source, target, relations=[relation])

    return graph


def _layout(graph: nx.DiGraph) -> dict[str, tuple[float, float]]:
    """Use a clean hub layout for the small demo and spring layout otherwise."""

    if len(graph) <= 6 and any(data.get("type") == "Paper" for _, data in graph.nodes(data=True)):
        positions: dict[str, tuple[float, float]] = {}
        paper_nodes = [node for node, data in graph.nodes(data=True) if data.get("type") == "Paper"]
        other_types = {"Method": [], "Disease": [], "Dataset": []}
        for node, data in graph.nodes(data=True):
            if data.get("type") in other_types:
                other_types[data["type"]].append(node)

        for index, node in enumerate(paper_nodes):
            positions[node] = (0.0, (index - (len(paper_nodes) - 1) / 2) * 1.35)

        slots = {
            "Method": (2.45, 0.0),
            "Disease": (-2.25, -1.35),
            "Dataset": (-2.25, 1.35),
        }
        for entity_type, entity_nodes in other_types.items():
            base_x, base_y = slots[entity_type]
            for index, node in enumerate(entity_nodes):
                positions[node] = (base_x, base_y + (index - (len(entity_nodes) - 1) / 2) * 0.7)
        if len(positions) == len(graph):
            return positions

    return nx.spring_layout(graph, seed=42, k=1.7, iterations=300)


def render(input_path: Path, output_path: Path) -> Path:
    graph = _load_graph(input_path)
    if not graph.nodes:
        raise ValueError("知识图谱没有节点，无法绘图。")

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    figure, axis = plt.subplots(figsize=(14, 8), facecolor="white")
    positions = _layout(graph)

    for entity_type, style in NODE_STYLES.items():
        selected = [
            node
            for node, data in graph.nodes(data=True)
            if data.get("type") == entity_type
        ]
        if selected:
            nx.draw_networkx_nodes(
                graph,
                positions,
                nodelist=selected,
                node_color=style["color"],
                node_shape=style["shape"],
                node_size=2800,
                edgecolors="white",
                linewidths=2,
                alpha=0.96,
                ax=axis,
            )

    nx.draw_networkx_edges(
        graph,
        positions,
        edge_color="#667085",
        width=1.8,
        alpha=0.85,
        arrows=True,
        arrowsize=22,
        arrowstyle="-|>",
        min_source_margin=26,
        min_target_margin=26,
        connectionstyle="arc3,rad=0.06",
        ax=axis,
    )

    # Put readable labels outside the colored marks so long titles are not clipped.
    label_offsets = {
        "Paper": (0.0, -0.56),
        "Method": (0.0, 0.56),
        "Disease": (0.0, -0.56),
        "Dataset": (0.0, 0.56),
    }
    for node, data in graph.nodes(data=True):
        x, y = positions[node]
        dx, dy = label_offsets.get(data.get("type"), (0.0, -0.56))
        axis.text(
            x + dx,
            y + dy,
            _short_label(data.get("label", node)),
            ha="center",
            va="top" if dy < 0 else "bottom",
            fontsize=10,
            color="#344054",
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": "#D0D5DD",
                "alpha": 0.95,
            },
            clip_on=False,
        )

    edge_labels = {
        (source, target): " / ".join(
            RELATION_NAMES.get(relation, relation)
            for relation in data.get("relations", [])
        )
        for source, target, data in graph.edges(data=True)
    }
    nx.draw_networkx_edge_labels(
        graph,
        positions,
        edge_labels=edge_labels,
        font_size=9,
        font_color="#344054",
        rotate=False,
        bbox={
            "boxstyle": "round,pad=0.2",
            "facecolor": "white",
            "edgecolor": "#D0D5DD",
            "alpha": 0.92,
        },
        ax=axis,
    )

    legend = [
        Line2D(
            [0],
            [0],
            marker=style["shape"],
            color="none",
            markerfacecolor=style["color"],
            markeredgecolor="white",
            markersize=12,
            label=style["legend"],
        )
        for entity_type, style in NODE_STYLES.items()
        if any(data.get("type") == entity_type for _, data in graph.nodes(data=True))
    ]
    axis.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=max(1, len(legend)),
        frameon=False,
        fontsize=10,
    )

    xs = [x for x, _ in positions.values()]
    ys = [y for _, y in positions.values()]
    axis.set_xlim(min(xs) - 1.25, max(xs) + 1.25)
    axis.set_ylim(min(ys) - 1.05, max(ys) + 1.05)
    axis.set_title("PubMed 摘要级证据知识图谱", fontsize=18, fontweight="bold", color="#17365D", pad=22)
    axis.axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.35, facecolor="white")
    plt.close(figure)
    return output_path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="绘制阶段 8 的证据知识图谱 PNG")
    parser.add_argument(
        "--input",
        type=Path,
        default=project_root / "artifacts" / "evidence_graph_demo.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "artifacts" / "evidence_graph_visualization.png",
    )
    args = parser.parse_args()
    print(render(args.input, args.output))


if __name__ == "__main__":
    main()
