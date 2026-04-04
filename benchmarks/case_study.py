import sys
import os
import argparse

# Add parent directory to path to allow importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark_args import add_algorithm_args, algorithm_kwargs_from_args
from utils.algorithm_adapter import run_bruteforce, run_mst, run_sim_annealing, run_qaoa

# Define the graph
graph = [
    [0, 10, 15, 20, 25],
    [10, 0, 35, 25, 30],
    [15, 35, 0, 30, 35],
    [20, 25, 30, 0, 25],
    [25, 30, 35, 25, 0]
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run all four TSP algorithms on a fixed 5-city case study graph with visualization."
    )
    add_algorithm_args(parser)
    args = parser.parse_args()
    kw = algorithm_kwargs_from_args(args)

    run_bruteforce(graph, visualize=True, **kw)
    run_mst(graph, visualize=True, **kw)
    run_sim_annealing(graph, visualize=True, **kw)
    run_qaoa(graph, visualize=True, **kw)
