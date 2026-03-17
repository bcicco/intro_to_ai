from models import Problem, Node, Edge
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def parse_problem(file_path: str) -> Problem:
    with open(file_path) as f:
        lines = [line.strip() for line in f if line.strip()]

    section = None
    nodes: dict[int, Node] = {}
    edges: list[Edge] = []
    origin: int | None = None
    destinations: list[int] = []

    for line in lines:
        if line == "Nodes:":
            section = "nodes"
        elif line == "Edges:":
            section = "edges"
        elif line == "Origin:":
            section = "origin"
        elif line == "Destinations:":
            section = "destinations"
        elif section == "nodes":
            node_id, coords = line.split(":")
            x, y = coords.strip().strip("()").split(",")
            nodes[int(node_id)] = Node(
                id=int(node_id),
                coordinates=(int(x), int(y)),
            )
        elif section == "edges":
            edge_part, cost = line.split(":")
            start, end = edge_part.strip().strip("()").split(",")
            edge = Edge(
                start_node_id=int(start), end_node_id=int(end), cost=float(cost)
            )
            edges.append(edge)
            nodes[int(start)].edges.append(edge)
        elif section == "origin":
            origin = int(line)
        elif section == "destinations":
            destinations = [int(d.strip()) for d in line.split(";")]

    return Problem(nodes=list(nodes.values()), origin=origin, destinations=destinations)


def visualise(problem: Problem):
    node_map = {n.id: n for n in problem.nodes}

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title("Pathfinding Problem")
    ax.set_aspect("equal")
    ax.axis("off")

    # Draw edges
    for node in problem.nodes:
        x1, y1 = node.coordinates
        for edge in node.edges:
            x2, y2 = node_map[edge.end_node_id].coordinates
            ax.annotate(
                "",
                xy=(x2, y2),
                xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.2),
            )
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx, my, str(edge.cost), fontsize=8, color="dimgray", ha="center")

    # Draw nodes
    for node in problem.nodes:
        x, y = node.coordinates
        if node.id == problem.origin:
            color = "limegreen"
        elif node.id in problem.destinations:
            color = "tomato"
        else:
            color = "steelblue"

        circle = plt.Circle((x, y), 0.2, color=color, zorder=3)
        ax.add_patch(circle)
        ax.text(
            x,
            y,
            str(node.id),
            fontsize=10,
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
            zorder=4,
        )

    legend = [
        mpatches.Patch(color="limegreen", label=f"Origin ({problem.origin})"),
        mpatches.Patch(color="tomato", label=f"Destinations {problem.destinations}"),
        mpatches.Patch(color="steelblue", label="Node"),
    ]
    ax.legend(handles=legend, loc="upper left")

    ax.autoscale()
    plt.tight_layout()
    plt.show()
