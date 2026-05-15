import time
import tracemalloc
from statistics import mean

from assignment1.helpers import parse_problem
from assignment1.search import METHODS


def run_once(filename: str, method: str):
    problem = parse_problem(filename)

    tracemalloc.start()
    start = time.perf_counter()

    result = METHODS[method](problem)

    end = time.perf_counter()
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    runtime_ms = (end - start) * 1000

    if result is None:
        return {
            "method": method,
            "runtime_ms": runtime_ms,
            "peak_memory_kb": peak_mem / 1024,
            "goal": None,
            "nodes_created": 0,
            "path_cost": None,
            "path_length": 0,
        }

    path, cost, nodes_created = result
    return {
        "method": method,
        "runtime_ms": runtime_ms,
        "peak_memory_kb": peak_mem / 1024,
        "goal": path[-1],
        "nodes_created": nodes_created,
        "path_cost": cost,
        "path_length": len(path) - 1,
    }


def benchmark(filename: str, repeats: int = 20):
    methods = ["DFS", "BFS", "GBFS", "AS", "CUS1", "CUS2"]
    all_results = []

    for method in methods:
        runs = [run_once(filename, method) for _ in range(repeats)]

        summary = {
            "Algorithm": method,
            "Avg Runtime (ms)": round(mean(r["runtime_ms"] for r in runs), 4),
            "Avg Peak Memory (KB)": round(mean(r["peak_memory_kb"] for r in runs), 2),
            "Goal": runs[0]["goal"],
            "Nodes Created": runs[0]["nodes_created"],
            "Path Cost": runs[0]["path_cost"],
            "Path Length": runs[0]["path_length"],
        }
        all_results.append(summary)

    return all_results


def print_results(results):
    for r in results:
        print(f"Algorithm: {r['Algorithm']}")
        print(f"Avg run time: {r['Avg Runtime (ms)']} ms")
        print(f"Avg peak memory: {r['Avg Peak Memory (KB)']} KB")
        print(f"Goal: {r['Goal']}")
        print(f"Nodes created: {r['Nodes Created']}")
        print(f"Path cost: {r['Path Cost']}")
        print(f"Path length: {r['Path Length']}")
        print("-" * 40)


if __name__ == "__main__":
    filename = "test_data/PathFinder-test.txt"
    results = benchmark(filename, repeats=20)
    print_results(results)
