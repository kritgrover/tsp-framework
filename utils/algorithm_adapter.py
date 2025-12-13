"""
Algorithm adapters to standardize algorithm interfaces.
Wraps existing algorithms to provide consistent I/O.
"""
import time
from typing import Dict, Any
from utils.result import TSPSolution
from utils.visualizer import visualize_solution
from algorithms import bruteforce, mst, sim_annealing
from algorithms.qaoa import QAOATSPSolver


def run_bruteforce(graph: list, visualize: bool = False, **params) -> TSPSolution:
    """
    Run brute force algorithm with standardized interface.
    
    Args:
        graph: Distance matrix (2D list)
        visualize: Whether to visualize the solution
        **params: Additional parameters (update_frequency, etc.)
    
    Returns:
        TSPSolution object
    """
    t_start = time.time()
    
    # Extract parameters
    update_frequency = params.get('update_frequency', None)
    
    # Run algorithm (disable its own visualization, we'll handle it)
    result = bruteforce.tsp_solver_final(graph, visualize=False, 
                                       update_frequency=update_frequency)
    
    t_end = time.time()
    
    # Parse result: (best_cost, path_str, valid_count)
    best_cost, path_str, valid_count = result
    
    # Convert path string to list
    path = None
    if path_str:
        # Parse "0 -> 1 -> 2 -> 0" format
        path = [int(x.strip()) for x in path_str.split('->')]
        # Remove duplicate start node at end if present
        if len(path) > 1 and path[0] == path[-1]:
            path = path[:-1]
    
    metadata = {
        'time_taken': t_end - t_start,
        'valid_tours': valid_count,
        'update_frequency': update_frequency
    }
    
    solution = TSPSolution(path, best_cost, 'bruteforce', metadata)
    
    # Visualize if requested
    if visualize and path:
        visualize_solution(graph, path, best_cost, 'Brute Force')
    
    return solution


def run_mst(graph: list, visualize: bool = False, **params) -> TSPSolution:
    """
    Run MST approximation algorithm with standardized interface.
    
    Args:
        graph: Distance matrix (2D list)
        visualize: Whether to visualize the solution
        **params: Additional parameters (currently unused)
    
    Returns:
        TSPSolution object
    """
    t_start = time.time()
    
    # Run algorithm (disable its own visualization)
    result = mst.tsp_solver_final(graph, visualize=False)
    
    t_end = time.time()
    
    # Parse result: (path, total_cost)
    path, total_cost = result
    
    metadata = {
        'time_taken': t_end - t_start
    }
    
    solution = TSPSolution(path, total_cost, 'mst', metadata)
    
    # Visualize if requested
    if visualize and path:
        visualize_solution(graph, path, total_cost, 'MST Approximation')
    
    return solution


def run_sim_annealing(graph: list, visualize: bool = False, **params) -> TSPSolution:
    """
    Run simulated annealing algorithm with standardized interface.
    
    Args:
        graph: Distance matrix (2D list)
        visualize: Whether to visualize the solution
        **params: Algorithm-specific parameters:
            - sa_initial_temp: Initial temperature (default: 1000)
            - sa_cooling_rate: Cooling rate (default: 0.003)
            - sa_max_iter: Maximum iterations (default: 10000)
    
    Returns:
        TSPSolution object
    """
    t_start = time.time()
    
    # Extract parameters
    initial_temp = params.get('sa_initial_temp', 1000)
    cooling_rate = params.get('sa_cooling_rate', 0.003)
    max_iter = params.get('sa_max_iter', 10000)
    
    # Run algorithm (disable its own visualization)
    result = sim_annealing.simulated_annealing_final(
        graph,
        initial_temp=initial_temp,
        cooling_rate=cooling_rate,
        max_iter=max_iter,
        visualize=False
    )
    
    t_end = time.time()
    
    # Parse result: (best_cost, path_str)
    best_cost, path_str = result
    
    # Convert path string to list
    path = None
    if path_str:
        # Parse "0 -> 1 -> 2 -> 0" format
        path = [int(x.strip()) for x in path_str.split('->')]
        # Remove duplicate start node at end if present
        if len(path) > 1 and path[0] == path[-1]:
            path = path[:-1]
    
    metadata = {
        'time_taken': t_end - t_start,
        'initial_temp': initial_temp,
        'cooling_rate': cooling_rate,
        'max_iter': max_iter
    }
    
    solution = TSPSolution(path, best_cost, 'sim_annealing', metadata)
    
    # Visualize if requested
    if visualize and path:
        visualize_solution(graph, path, best_cost, 'Simulated Annealing')
    
    return solution


def run_qaoa(graph: list, visualize: bool = False, **params) -> TSPSolution:
    """
    Run QAOA algorithm with standardized interface.
    
    Args:
        graph: Distance matrix (2D list)
        visualize: Whether to visualize the solution
        **params: Algorithm-specific parameters:
            - qaoa_layers: Number of QAOA layers (default: 2)
            - qaoa_learning_rate: Learning rate (default: 0.01)
            - qaoa_optimization_steps: Optimization steps (default: 200)
    
    Returns:
        TSPSolution object
    """
    t_start = time.time()
    
    # Extract parameters
    num_layers = params.get('qaoa_layers', 2)
    learning_rate = params.get('qaoa_learning_rate', 0.01)
    optimization_steps = params.get('qaoa_optimization_steps', 200)
    
    # Create solver
    solver = QAOATSPSolver(
        graph,
        num_qaoa_layers=num_layers,
        learning_rate=learning_rate,
        optimization_steps=optimization_steps
    )
    
    # Run algorithm
    bitstring, prob = solver.solve()
    
    # Decode solution
    path, cost = solver.decode_solution(bitstring)
    
    t_end = time.time()
    
    metadata = {
        'time_taken': t_end - t_start,
        'num_layers': num_layers,
        'learning_rate': learning_rate,
        'optimization_steps': optimization_steps,
        'probability': prob
    }
    
    solution = TSPSolution(path, cost, 'qaoa', metadata)
    
    # Visualize if requested
    if visualize and path:
        visualize_solution(graph, path, cost, 'QAOA')
    
    return solution


# Algorithm registry
ALGORITHMS = {
    'bruteforce': run_bruteforce,
    'mst': run_mst,
    'sim_annealing': run_sim_annealing,
    'qaoa': run_qaoa
}


def get_algorithm(algorithm_name: str):
    """
    Get algorithm function by name.
    
    Args:
        algorithm_name: Name of the algorithm
    
    Returns:
        Algorithm function
    
    Raises:
        ValueError: If algorithm name is not recognized
    """
    if algorithm_name not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm: {algorithm_name}. "
                       f"Available algorithms: {list(ALGORITHMS.keys())}")
    return ALGORITHMS[algorithm_name]

