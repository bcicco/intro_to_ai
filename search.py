import sys
from models import Problem, SearchNode
from helpers import parse_problem


class Frontier:
    def __init__(self):
        self._stack: list[SearchNode] = []

    def push(self, node: SearchNode) -> None:
        self._stack.append(node)

    def pop(self) -> SearchNode:
        return self._stack.pop()

    def is_empty(self) -> bool:
        return len(self._stack) == 0

    def __len__(self) -> int:
        return len(self._stack)
    
class BFSFrontier:
    def __init__(self):
        self._queue: list[SearchNode] = []
    
    def push(self, node: SearchNode) -> None:
        self._queue.append(node)

    def pop(self) -> SearchNode:
        return self._queue.pop(0)
    
    def is_empty(self) -> bool:
        return len(self._queue) == 0
    
    def __len__(self) -> int:
        return len(self._queue)


def dfs(problem: Problem) -> tuple[list[int], float, int] | None:
    """Returns (path, cost, nodes_created) or None if no path exists."""
    node_map = {n.id: n for n in problem.nodes}
    destinations = set(problem.destinations)
    nodes_created = 0

    frontier = Frontier()
    frontier.push(SearchNode(node_id=problem.origin, path=[problem.origin], cost=0.0))
    nodes_created += 1
    visited: set[int] = set()

    while not frontier.is_empty():
        current = frontier.pop()

        if current.node_id in visited:
            continue
        visited.add(current.node_id)

        if current.node_id in destinations:
            return current.path, current.cost, nodes_created

        node = node_map[current.node_id]
        # Push in reverse so lower-numbered edges are explored first
        for edge in reversed(node.edges):
            if edge.end_node_id not in visited:
                frontier.push(SearchNode(
                    node_id=edge.end_node_id,
                    path=current.path + [edge.end_node_id],
                    cost=current.cost + edge.cost,
                ))
                nodes_created += 1

    return None

def bfs(problem: Problem) -> tuple[list[int], float, int] | None:
    """Returns (path, cost, nodes_created) or none if no path exists."""

    node_map = {n.id: n for n in problem.nodes}
    destination = set(problem.destinations)
    nodes_created = 0

    frontier = BFSFrontier()
    frontier.push(SearchNode(node_id=problem.origin, path=[problem.origin] , cost=0.0))
    nodes_created += 1

    visited:set[int] = set()
    seen:set[int] = {problem.origin}

    while not frontier.is_empty():
        current = frontier.pop()

        if current.node_id is visited:
            continue
        visited.add(current.node_id)

        if current.node_id in destination:
            return current.path , current.cost, nodes_created
        node = node_map[current.node_id]

        for edge in sorted(node.edges, key=lambda e: e.end_node_id):
            if edge.end_node_id not in seen:
                frontier.push(SearchNode(
                    node_id=edge.end_node_id,
                    path=current.path + [edge.end_node_id],
                    cost= current.cost + edge.cost,
                ))
                seen.add(edge.end_node_id)
                nodes_created +=1
    return None


METHODS = {
    "DFS": dfs,
    "BFS": bfs,
}


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python search.py <filename> <method>")
        sys.exit(1)

    filename, method = sys.argv[1], sys.argv[2].upper()

    if method not in METHODS:
        print(f"Unknown method '{method}'. Available: {', '.join(METHODS)}")
        sys.exit(1)

    problem = parse_problem(filename)
    result = METHODS[method](problem)

    print(f"{filename} {method}")
    if result:
        path, cost, nodes_created = result
        print(f"{path[-1]} {nodes_created}")
        print(" -> ".join(str(n) for n in path))
    else:
        print("No path found.")
