import sys
import os
# Add parent directory to path to allow importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.algorithm_adapter import run_bruteforce, run_mst, run_sim_annealing, run_qaoa

# Define the graph
graph = [
    [0, 10, 15, 20, 25],
    [10, 0, 35, 25, 30],
    [15, 35, 0, 30, 35],
    [20, 25, 30, 0, 25],
    [25, 30, 35, 25, 0]
]

run_bruteforce(graph, visualize=True)
run_mst(graph, visualize=True)
run_sim_annealing(graph, visualize=True)
run_qaoa(graph, visualize=True)
