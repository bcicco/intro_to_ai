"""
Tests for: DFS, BFS, A* (AS), CUS1 (Bidirectional BFS), CUS2 (Weighted A*).

Run with:  python -m pytest test_search.py -v
       or: python test_search.py
"""

import argparse
import os
import sys
import unittest

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import Edge, Node, Problem
from search import METHODS, astar, bfs, cus1, cus2, dfs, gbfs
from helpers import parse_problem, visualise


# ── Helpers ──────────────────────────────────────────────────────────────────


def make_problem(
    nodes: list[tuple[int, int, int]],  # (id, x, y)
    edges: list[tuple[int, int, float]],  # (start, end, cost)
    origin: int,
    destinations: list[int],
) -> Problem:
    node_map = {nid: Node(id=nid, coordinates=(x, y)) for nid, x, y in nodes}
    for start, end, cost in edges:
        node_map[start].edges.append(
            Edge(start_node_id=start, end_node_id=end, cost=cost)
        )
    return Problem(
        nodes=list(node_map.values()), origin=origin, destinations=destinations
    )


def is_valid_path(path: list[int], problem: Problem) -> bool:
    """Every consecutive pair in path must be connected by an edge."""
    edge_set = {
        (e.start_node_id, e.end_node_id) for n in problem.nodes for e in n.edges
    }
    return all((path[i], path[i + 1]) in edge_set for i in range(len(path) - 1))


def path_cost(path: list[int], problem: Problem) -> float:
    """Sum the actual edge costs along a path."""
    cost_map = {
        (e.start_node_id, e.end_node_id): e.cost for n in problem.nodes for e in n.edges
    }
    return sum(cost_map[(path[i], path[i + 1])] for i in range(len(path) - 1))


ALL = [dfs, bfs, astar, gbfs, cus1, cus2]
NAMES = ["DFS", "BFS", "A*", "GBFS", "CUS1", "CUS2"]


# ── 1. Linear chain ───────────────────────────────────────────────────────────


class TestLinearChain(unittest.TestCase):
    """
    1 → 2 → 3 → 4 → 5  (all edges cost 1, single path)
    Every algorithm must find [1,2,3,4,5] with cost 4.
    """

    def setUp(self):
        self.p = make_problem(
            nodes=[(1, 0, 0), (2, 1, 0), (3, 2, 0), (4, 3, 0), (5, 4, 0)],
            edges=[(1, 2, 1), (2, 3, 1), (3, 4, 1), (4, 5, 1)],
            origin=1,
            destinations=[5],
        )

    def test_all_find_path(self):
        for fn, name in zip(ALL, NAMES):
            with self.subTest(algorithm=name):
                result = fn(self.p)
                self.assertIsNotNone(result)
                path, cost, _ = result
                self.assertEqual(path, [1, 2, 3, 4, 5])
                self.assertAlmostEqual(cost, 4.0)


# ── 2. No path (disconnected) ─────────────────────────────────────────────────


class TestNoPath(unittest.TestCase):
    """
    Two disconnected components: {1,2,3} and {4,5}.
    No algorithm should find a path from 1 to 5.
    """

    def setUp(self):
        self.p = make_problem(
            nodes=[(1, 0, 0), (2, 1, 0), (3, 2, 0), (4, 10, 0), (5, 11, 0)],
            edges=[(1, 2, 1), (2, 3, 1), (4, 5, 1)],
            origin=1,
            destinations=[5],
        )

    def test_all_return_none(self):
        for fn, name in zip(ALL, NAMES):
            with self.subTest(algorithm=name):
                self.assertIsNone(fn(self.p))


# ── 3. Origin is destination ──────────────────────────────────────────────────


class TestOriginIsDestination(unittest.TestCase):
    """
    Origin node is in the destinations list — return [origin] with cost 0.
    """

    def setUp(self):
        self.p = make_problem(
            nodes=[(1, 0, 0), (2, 1, 0), (3, 2, 0)],
            edges=[(1, 2, 5), (2, 3, 5)],
            origin=1,
            destinations=[1],
        )

    def test_all_return_trivial_path(self):
        for fn, name in zip(ALL, NAMES):
            with self.subTest(algorithm=name):
                result = fn(self.p)
                self.assertIsNotNone(result)
                path, cost, _ = result
                self.assertEqual(path, [1])
                self.assertAlmostEqual(cost, 0.0)


# ── 4. Multiple destinations ──────────────────────────────────────────────────


class TestMultipleDestinations(unittest.TestCase):
    """
    1 → 2  (cost 1)       dest: 2  (1 hop)
    1 → 3 → 4  (cost 1+1) dest: 4  (2 hops)

    BFS and CUS1 should reach dest 2 first (fewest hops).
    A* should also prefer dest 2 (lowest cost).
    All algorithms must end at a valid destination.
    """

    def setUp(self):
        self.p = make_problem(
            nodes=[(1, 0, 0), (2, 1, 0), (3, 2, 0), (4, 3, 0)],
            edges=[(1, 2, 1), (1, 3, 1), (3, 4, 1)],
            origin=1,
            destinations=[2, 4],
        )

    def test_all_reach_a_destination(self):
        for fn, name in zip(ALL, NAMES):
            with self.subTest(algorithm=name):
                result = fn(self.p)
                self.assertIsNotNone(result)
                path, _, _ = result
                self.assertIn(path[-1], {2, 4})
                self.assertTrue(is_valid_path(path, self.p))

    def test_bfs_reaches_nearest_by_hops(self):
        path, _, _ = bfs(self.p)
        self.assertEqual(len(path) - 1, 1)

    def test_cus1_reaches_nearest_by_hops(self):
        path, _, _ = cus1(self.p)
        self.assertEqual(len(path) - 1, 1)

    def test_astar_finds_minimum_cost(self):
        _, cost, _ = astar(self.p)
        self.assertAlmostEqual(cost, 1.0)


# ── 5. Weighted shortcut ──────────────────────────────────────────────────────


class TestWeightedShortcut(unittest.TestCase):
    """
    Two paths from 1 to 4:
      1 → 2 → 4   (2 hops, cost 101)
      1 → 3 → 4   (2 hops, cost 11)

    BFS may take either (equal hops — picks lower node ID first → [1,2,4]).
    A* must find the cheaper path [1,3,4] (cost 11).
    """

    def setUp(self):
        self.p = make_problem(
            nodes=[(1, 0, 0), (2, 1, 0), (3, 5, 0), (4, 6, 0)],
            edges=[(1, 2, 1), (2, 4, 100), (1, 3, 10), (3, 4, 1)],
            origin=1,
            destinations=[4],
        )

    def test_all_reach_destination(self):
        for fn, name in zip(ALL, NAMES):
            with self.subTest(algorithm=name):
                result = fn(self.p)
                self.assertIsNotNone(result)
                path, cost, _ = result
                self.assertEqual(path[-1], 4)
                self.assertTrue(is_valid_path(path, self.p))
                self.assertAlmostEqual(cost, path_cost(path, self.p))

    def test_astar_finds_optimal(self):
        _, cost, _ = astar(self.p)
        self.assertAlmostEqual(cost, 11.0)

    def test_bfs_finds_2_hop_path(self):
        path, _, _ = bfs(self.p)
        self.assertEqual(len(path) - 1, 2)


# ── 6. Cycle handling ─────────────────────────────────────────────────────────


class TestCycleHandling(unittest.TestCase):
    """
    1 → 2 → 3 → 1  (cycle)
    2 → 4           (exit)
    Algorithms must not loop infinitely and must reach 4.
    """

    def setUp(self):
        self.p = make_problem(
            nodes=[(1, 0, 0), (2, 1, 0), (3, 0, 1), (4, 2, 0)],
            edges=[(1, 2, 1), (2, 3, 1), (3, 1, 1), (2, 4, 1)],
            origin=1,
            destinations=[4],
        )

    def test_all_find_path_without_looping(self):
        for fn, name in zip(ALL, NAMES):
            with self.subTest(algorithm=name):
                result = fn(self.p)
                self.assertIsNotNone(result)
                path, _, _ = result
                self.assertEqual(path[-1], 4)
                self.assertTrue(is_valid_path(path, self.p))


# ── 7. Directed graph ─────────────────────────────────────────────────────────


class TestDirectedGraph(unittest.TestCase):
    """
    Edges are one-way: 1→2→3→4.
    Forward (1→4): path exists.
    Backward (4→1): no path exists.
    """

    def setUp(self):
        nodes = [(1, 0, 0), (2, 1, 0), (3, 2, 0), (4, 3, 0)]
        edges = [(1, 2, 1), (2, 3, 1), (3, 4, 1)]
        self.forward = make_problem(nodes, edges, origin=1, destinations=[4])
        self.backward = make_problem(nodes, edges, origin=4, destinations=[1])

    def test_forward_path_found(self):
        for fn, name in zip(ALL, NAMES):
            with self.subTest(algorithm=name):
                result = fn(self.forward)
                self.assertIsNotNone(result)
                path, _, _ = result
                self.assertEqual(path[0], 1)
                self.assertEqual(path[-1], 4)

    def test_backward_path_not_found(self):
        for fn, name in zip(ALL, NAMES):
            with self.subTest(algorithm=name):
                self.assertIsNone(fn(self.backward))


# ── 8. Diamond graph (two equal paths) ───────────────────────────────────────


class TestDiamondGraph(unittest.TestCase):
    """
        1
       / \\
      2   3
       \\ /
        4
    All edges cost 1. Both paths have equal hops and cost.
    Every algorithm must find a valid 2-hop path.
    """

    def setUp(self):
        self.p = make_problem(
            nodes=[(1, 1, 2), (2, 0, 1), (3, 2, 1), (4, 1, 0)],
            edges=[(1, 2, 1), (1, 3, 1), (2, 4, 1), (3, 4, 1)],
            origin=1,
            destinations=[4],
        )

    def test_all_find_2_hop_path(self):
        for fn, name in zip(ALL, NAMES):
            with self.subTest(algorithm=name):
                result = fn(self.p)
                self.assertIsNotNone(result)
                path, cost, _ = result
                self.assertEqual(path[0], 1)
                self.assertEqual(path[-1], 4)
                self.assertEqual(len(path) - 1, 2)
                self.assertAlmostEqual(cost, 2.0)
                self.assertTrue(is_valid_path(path, self.p))


# ── 9. Star graph (many branches, one leads to goal) ─────────────────────────


class TestStarGraph(unittest.TestCase):
    """
    Hub node 1 connects to 6 branches; only branch 5 leads to goal 10.
    Tests that all algorithms correctly explore and backtrack/skip dead ends.

    1 → 2, 3, 4, 5, 6, 7   (dead ends except 5)
    5 → 8 → 9 → 10
    """

    def setUp(self):
        self.p = make_problem(
            nodes=[
                (1, 3, 3),
                (2, 0, 5),
                (3, 1, 5),
                (4, 2, 5),
                (5, 3, 5),
                (6, 4, 5),
                (7, 5, 5),
                (8, 3, 7),
                (9, 3, 9),
                (10, 3, 11),
            ],
            edges=[
                (1, 2, 1),
                (1, 3, 1),
                (1, 4, 1),
                (1, 5, 1),
                (1, 6, 1),
                (1, 7, 1),
                (5, 8, 1),
                (8, 9, 1),
                (9, 10, 1),
            ],
            origin=1,
            destinations=[10],
        )

    def test_all_find_path(self):
        for fn, name in zip(ALL, NAMES):
            with self.subTest(algorithm=name):
                result = fn(self.p)
                self.assertIsNotNone(result)
                path, _, _ = result
                self.assertEqual(path[0], 1)
                self.assertEqual(path[-1], 10)
                self.assertTrue(is_valid_path(path, self.p))

    def test_astar_finds_optimal(self):
        _, cost, _ = astar(self.p)
        self.assertAlmostEqual(cost, 4.0)


# ── 10. Long bidirectional chain (CUS1 vs BFS) ───────────────────────────────


class TestBidirectionalEfficiency(unittest.TestCase):
    """
    Chain of 20 nodes with edges in both directions: 1↔2↔...↔20.
    CUS1 and BFS must agree on path and cost.
    """

    def setUp(self):
        n = 20
        self.p = make_problem(
            nodes=[(i, i - 1, 0) for i in range(1, n + 1)],
            edges=[(i, i + 1, 1.0) for i in range(1, n)]
            + [(i + 1, i, 1.0) for i in range(1, n)],
            origin=1,
            destinations=[n],
        )

    def test_bfs_and_cus1_agree(self):
        bfs_result = bfs(self.p)
        cus1_result = cus1(self.p)
        self.assertIsNotNone(bfs_result)
        self.assertIsNotNone(cus1_result)
        bfs_path, bfs_cost, _ = bfs_result
        cus1_path, cus1_cost, _ = cus1_result
        self.assertEqual(bfs_path, cus1_path)
        self.assertAlmostEqual(bfs_cost, cus1_cost)


# ── 11. Nodes created count is positive ──────────────────────────────────────


class TestNodesCreatedCount(unittest.TestCase):
    """nodes_created must be >= 1 whenever a result is returned."""

    def setUp(self):
        self.p = make_problem(
            nodes=[(1, 0, 0), (2, 1, 0), (3, 2, 0)],
            edges=[(1, 2, 2), (2, 3, 3)],
            origin=1,
            destinations=[3],
        )

    def test_nodes_created_positive(self):
        for fn, name in zip(ALL, NAMES):
            with self.subTest(algorithm=name):
                result = fn(self.p)
                self.assertIsNotNone(result)
                _, _, nodes_created = result
                self.assertGreater(nodes_created, 0)

    def test_cost_matches_edge_sum(self):
        for fn, name in zip(ALL, NAMES):
            with self.subTest(algorithm=name):
                result = fn(self.p)
                self.assertIsNotNone(result)
                path, cost, _ = result
                self.assertAlmostEqual(cost, path_cost(path, self.p))


# ── 12. METHODS registry ─────────────────────────────────────────────────────


class TestMethodsRegistry(unittest.TestCase):
    """METHODS dict must expose all expected keys and map to callables."""

    def test_all_keys_present(self):
        for key in ("DFS", "BFS","GBFS", "AS", "CUS1", "CUS2"):
            self.assertIn(key, METHODS)

    def test_all_values_callable(self):
        for key, fn in METHODS.items():
            self.assertTrue(callable(fn), f"METHODS['{key}'] is not callable")


# ── 13. Integration: provided test file ──────────────────────────────────────


class TestProvidedFile(unittest.TestCase):
    """
    End-to-end test using test_data/PathFinder-test.txt.
    Origin: 2, Destinations: [5, 4].
    """

    def setUp(self):
        path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "test_data",
            "PathFinder-test.txt",
        )
        self.p = parse_problem(path)
        self.destinations = set(self.p.destinations)

    def test_all_find_valid_path(self):
        for fn, name in zip(ALL, NAMES):
            with self.subTest(algorithm=name):
                result = fn(self.p)
                self.assertIsNotNone(result)
                path, cost, nodes_created = result
                self.assertEqual(path[0], self.p.origin)
                self.assertIn(path[-1], self.destinations)
                self.assertTrue(is_valid_path(path, self.p))
                self.assertAlmostEqual(cost, path_cost(path, self.p))
                self.assertGreater(nodes_created, 0)

    def test_astar_cost_is_optimal(self):
        """A* cost must be <= cost found by every other algorithm."""
        _, astar_cost, _ = astar(self.p)
        for fn, name in zip([dfs, bfs, cus1, cus2], ["DFS", "BFS", "CUS1", "CUS2"]):
            with self.subTest(algorithm=name):
                result = fn(self.p)
                if result:
                    _, other_cost, _ = result
                    self.assertGreaterEqual(
                        other_cost + 1e-9,
                        astar_cost,
                        f"{name} found a cheaper path than A* — A* is not optimal",
                    )


# ── Visualisation ────────────────────────────────────────────────────────────


def visualize_files(pattern: str | None = None) -> None:
    """
    Show one figure per test file with a subplot for each algorithm.

    python test_search.py --visualize           # all 20 files
    python test_search.py --visualize 09        # only files matching '09'
    """
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")
    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".txt"))
    if pattern:
        files = [f for f in files if pattern in f]
    if not files:
        print(f"No test files matched pattern '{pattern}'.")
        return

    for fname in files:
        problem = parse_problem(os.path.join(data_dir, fname))
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(fname, fontsize=13, fontweight="bold")
        axes = axes.flatten()

        for i, (name, fn) in enumerate(zip(NAMES, ALL)):
            result = fn(problem)
            if result:
                path, cost, nodes_created = result
                subtitle = f"{name}   cost={cost:.0f}   nodes created={nodes_created}"
            else:
                path = None
                subtitle = f"{name}   no path found"
            visualise(problem, path=path, ax=axes[i], title=subtitle)

        plt.tight_layout()
        plt.show()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Search algorithm tests + visualisation"
    )
    parser.add_argument(
        "--visualize",
        "-V",
        nargs="?",
        const="",
        metavar="PATTERN",
        help="Visualise test files instead of running tests. "
        "Optionally pass a substring to filter filenames (e.g. '09').",
    )
    args, remaining = parser.parse_known_args()

    if args.visualize is not None:
        visualize_files(args.visualize or None)
    else:
        sys.argv = sys.argv[:1] + remaining
        unittest.main(verbosity=2)
