import heapq
from typing import Callable

from assignment1.models import Edge, Node, Problem

SolverFn = Callable[[Problem], tuple[list[int], float, int] | None]


def _build_subproblem(
    nodes: list[Node],
    source: int,
    target: int,
    removed_nodes: frozenset[int],
    removed_edges: frozenset[tuple[int, int]],
) -> Problem:
    filtered: list[Node] = []
    for node in nodes:
        if node.id in removed_nodes:
            continue
        kept_edges = [
            e for e in node.edges
            if e.end_node_id not in removed_nodes
            and (node.id, e.end_node_id) not in removed_edges
        ]
        filtered.append(Node(id=node.id, coordinates=node.coordinates, edges=kept_edges))
    return Problem(nodes=filtered, origin=source, destinations=[target])


def _dijkstra(
    adj: dict[int, list[tuple[int, float]]],
    source: int,
    target: int,
    removed_nodes: frozenset[int],
    removed_edges: frozenset[tuple[int, int]],
) -> tuple[list[int], float] | None:
    dist: dict[int, float] = {source: 0.0}
    prev: dict[int, int | None] = {source: None}
    heap: list[tuple[float, int]] = [(0.0, source)]
    visited: set[int] = set()

    while heap:
        cost, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)
        if u == target:
            path: list[int] = []
            node: int | None = target
            while node is not None:
                path.append(node)
                node = prev[node]
            return list(reversed(path)), cost
        for v, w in adj.get(u, []):
            if v in removed_nodes or (u, v) in removed_edges:
                continue
            nc = cost + w
            if nc < dist.get(v, float("inf")):
                dist[v] = nc
                prev[v] = u
                heapq.heappush(heap, (nc, v))

    return None


def yen_k_shortest(
    problem: Problem,
    k: int = 5,
    solver: SolverFn | None = None,
) -> list[tuple[list[int], float]]:
    """
    Return up to k loopless shortest paths using Yen's algorithm.

    solver: optional search function (e.g. METHODS['AS']). When provided it is
    used as the inner shortest-path oracle instead of the built-in Dijkstra,
    so the method dropdown in the GUI stays meaningful. Optimal solvers (A*,
    AS) preserve Yen's correctness; non-optimal ones (DFS, BFS) may return
    paths that are not globally ranked by cost.
    """
    adj: dict[int, list[tuple[int, float]]] = {}
    edge_cost: dict[tuple[int, int], float] = {}
    for node in problem.nodes:
        adj[node.id] = []
        for e in node.edges:
            adj[node.id].append((e.end_node_id, e.cost))
            edge_cost[(node.id, e.end_node_id)] = e.cost

    source = problem.origin
    target = problem.destinations[0]

    def path_cost(path: list[int]) -> float:
        return sum(
            edge_cost.get((path[i], path[i + 1]), float("inf"))
            for i in range(len(path) - 1)
        )

    def find_path(
        src: int,
        rem_nodes: frozenset[int],
        rem_edges: frozenset[tuple[int, int]],
    ) -> tuple[list[int], float] | None:
        if solver is not None:
            sub = _build_subproblem(problem.nodes, src, target, rem_nodes, rem_edges)
            result = solver(sub)
            if result is None:
                return None
            path, cost, _ = result
            return path, cost
        return _dijkstra(adj, src, target, rem_nodes, rem_edges)

    first = find_path(source, frozenset(), frozenset())
    if first is None:
        return []

    A: list[tuple[list[int], float]] = [first]
    B: list[tuple[float, int, list[int]]] = []
    B_paths: set[tuple[int, ...]] = set()
    _ctr = 0

    for _ in range(1, k):
        prev_path, _ = A[-1]

        for spur_idx in range(len(prev_path) - 1):
            spur_node = prev_path[spur_idx]
            root_path = prev_path[: spur_idx + 1]
            root_cost = path_cost(root_path)

            rem_edges: set[tuple[int, int]] = set()
            for p, _ in A:
                if len(p) > spur_idx and p[: spur_idx + 1] == root_path:
                    rem_edges.add((p[spur_idx], p[spur_idx + 1]))
            for _, _, p in B:
                if len(p) > spur_idx and p[: spur_idx + 1] == root_path:
                    rem_edges.add((p[spur_idx], p[spur_idx + 1]))

            rem_nodes = frozenset(root_path[:-1])
            spur_result = find_path(spur_node, rem_nodes, frozenset(rem_edges))
            if spur_result is not None:
                spur_path, spur_cost = spur_result
                total_path = root_path[:-1] + spur_path
                total_cost = root_cost + spur_cost
                key = tuple(total_path)
                if key not in B_paths and not any(p == total_path for p, _ in A):
                    B_paths.add(key)
                    heapq.heappush(B, (total_cost, _ctr, total_path))
                    _ctr += 1

        if not B:
            break

        best_cost, _, best_path = heapq.heappop(B)
        B_paths.discard(tuple(best_path))
        A.append((best_path, best_cost))

    return A
