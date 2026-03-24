from helpers import parse_problem, visualise
from search import dfs

problem = parse_problem("test_data/PathFinder-test.txt")

result = dfs(problem)
if result:
    path, cost, nodes_created = result
    print(f"Goal:          {path[-1]}")
    print(f"Nodes created: {nodes_created}")
    print(f"Cost:          {cost}")
    print(f"Path:          {' -> '.join(str(n) for n in path)}")
else:
    print("No path found.")

visualise(problem)
