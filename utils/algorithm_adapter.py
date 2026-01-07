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
from algorithms.qaoa_ibm import QAOAIBMSolver


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
    t_start = time.perf_counter()
    
    # Extract parameters
    update_frequency = params.get('update_frequency', None)
    
    # Run algorithm
    result = bruteforce.tsp_solver_final(graph, visualize=False, 
                                       update_frequency=update_frequency)
    
    t_end = time.perf_counter()
    
    # Parse result: (best_cost, path_str, valid_count)
    best_cost, path_str, valid_count = result
    
    # Convert path string to list
    path = None
    if path_str:
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
    t_start = time.perf_counter()
    
    # Run algorithm (disable its own visualization)
    result = mst.tsp_solver_final(graph, visualize=False)
    
    t_end = time.perf_counter()
    
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
    t_start = time.perf_counter()
    
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
    
    t_end = time.perf_counter()
    
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
    t_start = time.perf_counter()
    
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
    
    t_end = time.perf_counter()
    
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


def run_qaoa_ibm(graph: list, visualize: bool = False, **params) -> TSPSolution:
    """
    Run QAOA algorithm on IBM Quantum hardware with standardized interface.
    
    Args:
        graph: Distance matrix (2D list)
        visualize: Whether to visualize the solution
        **params: Algorithm-specific parameters:
            - qaoa_layers: Number of QAOA layers (default: 2)
            - qaoa_optimization_steps: Optimization steps (default: 100)
            - ibm_backend: IBM backend name (default: 'ibm_brisbane')
            - ibm_token: IBM Quantum API token (optional)
            - ibm_channel: 'ibm_cloud' or 'ibm_quantum' (default: 'ibm_cloud')
            - ibm_instance: IBM Cloud instance CRN (optional)
            - shots: Number of measurement shots (default: 1024)
            - use_ibm_backend: If False, uses local simulation (default: True)
    
    Returns:
        TSPSolution object
    """
    t_start = time.perf_counter()
    
    # Extract parameters
    num_layers = params.get('qaoa_layers', 2)
    optimization_steps = params.get('qaoa_optimization_steps', 100)
    ibm_backend = params.get('ibm_backend', 'ibm_brisbane')
    ibm_token = params.get('ibm_token', None)
    ibm_channel = params.get('ibm_channel', 'ibm_cloud')
    ibm_instance = params.get('ibm_instance', None)
    shots = params.get('shots', 1024)
    use_ibm_backend = params.get('use_ibm_backend', True)
    
    # Create solver
    solver = QAOAIBMSolver(
        graph,
        num_qaoa_layers=num_layers,
        optimization_steps=optimization_steps,
        use_ibm_backend=use_ibm_backend,
        ibm_backend=ibm_backend,
        ibm_token=ibm_token,
        ibm_channel=ibm_channel,
        ibm_instance=ibm_instance,
        shots=shots
    )
    
    # Run algorithm
    bitstring, prob = solver.solve()
    
    # Decode solution
    path, cost = solver.decode_solution(bitstring)
    
    t_end = time.perf_counter()
    
    metadata = {
        'time_taken': t_end - t_start,
        'num_layers': num_layers,
        'optimization_steps': optimization_steps,
        'ibm_backend': ibm_backend if use_ibm_backend else 'local_simulator',
        'shots': shots,
        'probability': prob,
        'bitstring': bitstring
    }
    
    solution = TSPSolution(path, cost, 'qaoa_ibm', metadata)
    
    # Visualize if requested
    if visualize and path:
        backend_name = f"QAOA (IBM: {ibm_backend})" if use_ibm_backend else "QAOA (Local Sim)"
        visualize_solution(graph, path, cost, backend_name)
    
    return solution


# Algorithm registry
ALGORITHMS = {
    'bruteforce': run_bruteforce,
    'mst': run_mst,
    'sim_annealing': run_sim_annealing,
    'qaoa': run_qaoa,
    'qaoa_ibm': run_qaoa_ibm
}


def get_algorithm(algorithm_name: str):
    """
    Get algorithm function by name.
    """
    if algorithm_name not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm: {algorithm_name}. "
                       f"Available algorithms: {list(ALGORITHMS.keys())}")
    return ALGORITHMS[algorithm_name]

