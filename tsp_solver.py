"""
Unified TSP Solver Framework

Main interface for solving TSP problems with multiple algorithms,
graph types, and visualization options.
"""
from typing import Optional, List, Dict, Any
from utils.graph_generators import generate_graph
from utils.algorithm_adapter import get_algorithm
from utils.result import TSPSolution
from utils.visualizer import visualize_graph_only


class TSPSolver:
    """
    Unified TSP Solver that supports multiple algorithms and graph types.
    
    Example:
        >>> solver = TSPSolver(
        ...     algorithm='mst',
        ...     graph_type='circle',
        ...     graph_params={'num_nodes': 8},
        ...     visualize=True
        ... )
        >>> result = solver.solve()
        >>> print(f"Path: {result.path}, Cost: {result.cost}")
    """
    
    def __init__(self, 
                 algorithm: str,
                 graph: Optional[List[List[float]]] = None,
                 graph_type: Optional[str] = None,
                 graph_params: Optional[Dict[str, Any]] = None,
                 visualize: bool = False,
                 # Algorithm-specific parameters
                 qaoa_layers: int = 2,
                 qaoa_learning_rate: float = 0.01,
                 qaoa_optimization_steps: int = 200,
                 sa_initial_temp: float = 1000,
                 sa_cooling_rate: float = 0.003,
                 sa_max_iter: int = 10000,
                 **kwargs):
        """
        Initialize TSP Solver.
        
        Args:
            algorithm: Algorithm to use ('bruteforce', 'mst', 'sim_annealing', 'qaoa')
            graph: Optional distance matrix. If None, generates based on graph_type
            graph_type: Type of graph to generate ('fully_connected', 'partially_connected', 
                      'circle'). Required if graph is None.
            graph_params: Parameters for graph generation (e.g., {'num_nodes': 5, 'connectivity': 0.6})
            visualize: Whether to visualize the solution

            **kwargs: Additional parameters passed to algorithms
            # Algorithm-specific parameters
            qaoa_layers: Number of QAOA layers
            qaoa_learning_rate: Learning rate
            qaoa_optimization_steps: Optimization steps
            
            sa_initial_temp: Initial temperature
            sa_cooling_rate: Cooling rate
            sa_max_iter: Maximum iterations
            
            **kwargs: Additional parameters passed to algorithms
        """
        self.algorithm = algorithm.lower()
        self.visualize = visualize
        
        # Validate algorithm
        valid_algorithms = ['bruteforce', 'mst', 'sim_annealing', 'qaoa']
        if self.algorithm not in valid_algorithms:
            raise ValueError(f"Invalid algorithm: {algorithm}. "
                           f"Must be one of: {valid_algorithms}")
        
        # Handle graph input/generation
        if graph is not None:
            self.graph = graph
            if graph_type is not None:
                print("Warning: Both graph and graph_type provided. Using provided graph.")
        elif graph_type is not None:
            if graph_params is None:
                graph_params = {}
            self.graph = generate_graph(graph_type, **graph_params)
            print(f"Generated {graph_type} graph with {len(self.graph)} nodes.")
        else:
            raise ValueError("Either 'graph' or 'graph_type' must be provided.")
        
        # Store algorithm-specific parameters
        self.algorithm_params = {
            'qaoa_layers': qaoa_layers,
            'qaoa_learning_rate': qaoa_learning_rate,
            'qaoa_optimization_steps': qaoa_optimization_steps,
            'sa_initial_temp': sa_initial_temp,
            'sa_cooling_rate': sa_cooling_rate,
            'sa_max_iter': sa_max_iter,
            **kwargs
        }
        
        # Store result after solving
        self.result: Optional[TSPSolution] = None
    
    def solve(self) -> TSPSolution:
        """
        Solve the TSP problem using the selected algorithm.
        
        Returns:
            TSPSolution object containing path, cost, and metadata
        """
        # Get algorithm function
        algorithm_func = get_algorithm(self.algorithm)
        
        # Run algorithm
        self.result = algorithm_func(
            self.graph,
            visualize=self.visualize,
            **self.algorithm_params
        )
        
        return self.result
    
    def visualize_result(self):
        """
        Visualize the solution (if one has been computed).
        """
        if self.result is None:
            raise ValueError("No solution computed yet. Call solve() first.")
        
        from utils.visualizer import visualize_solution
        visualize_solution(
            self.graph,
            self.result.path,
            self.result.cost,
            self.result.algorithm
        )
    
    def visualize_graph(self):
        """
        Visualize just the graph without a solution.
        """
        visualize_graph_only(self.graph, f"Graph ({len(self.graph)} nodes)")
    
    def get_result(self) -> Optional[TSPSolution]:
        """
        Get the last computed result.
        
        Returns:
            TSPSolution object or None if not solved yet
        """
        return self.result

