from tsp_solver import TSPSolver


# ============================================================================
# BASIC INITIALIZATION
# ============================================================================

# Minimal example: Algorithm + Graph Type
solver = TSPSolver(
    algorithm='mst',                    # Required: algorithm name
    graph_type='circle',                # Required if graph not provided
    graph_params={'num_nodes': 8}       # Parameters for graph generation
)

# With custom graph instead of generating one
custom_graph = [
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
]

solver = TSPSolver(
    algorithm='mst',
    graph=custom_graph                  # Provide your own distance matrix
)

# ============================================================================
# PARAMETER REFERENCE
# ============================================================================

"""
TSPSolver Parameters:

REQUIRED:
---------
algorithm : str
    Algorithm to use. Options:
    - 'bruteforce'  : Exhaustive search (optimal, slow for large graphs)
    - 'mst'         : Minimum Spanning Tree approximation (fast, 2-approx)
    - 'sim_annealing': Simulated Annealing (heuristic, good balance)
    - 'qaoa'        : Quantum Approximate Optimization Algorithm

graph : List[List[float]], optional
    Custom distance matrix (2D list). 
    - If provided, graph_type is ignored
    - Matrix should be symmetric with zeros on diagonal
    - Example: [[0, 10, 15], [10, 0, 20], [15, 20, 0]]

graph_type : str, optional
    Type of graph to generate. Required if graph is not provided.
    Options:
    - 'fully_connected'    : Complete graph (all nodes connected)
    - 'partially_connected': Random graph with specified connectivity
    - 'circle'             : Circular graph (each node has 2 neighbors)

graph_params : dict, optional
    Parameters for graph generation. Common keys:
    - 'num_nodes'    : int, Number of nodes (default: 5)
    - 'weight_range' : tuple, (min, max) edge weights (default: (1, 100))
    - 'seed'         : int, Random seed for reproducibility
    - 'connectivity' : float, For partially_connected (0.0-1.0, default: 0.6)

OPTIONAL:
---------
visualize : bool
    Whether to visualize the solution (default: False)

ALGORITHM-SPECIFIC PARAMETERS:
-------------------------------
For QAOA:
    qaoa_layers : int
        Number of QAOA layers (default: 2)
    qaoa_learning_rate : float
        Learning rate for optimization (default: 0.01)
    qaoa_optimization_steps : int
        Number of optimization steps (default: 200)

For Simulated Annealing:
    sa_initial_temp : float
        Initial temperature (default: 1000)
    sa_cooling_rate : float
        Cooling rate per iteration (default: 0.003)
    sa_max_iter : int
        Maximum iterations (default: 10000)
"""

# ============================================================================
# EXAMPLE INITIALIZATIONS
# ============================================================================

# Example 1: MST on a circle graph
solver1 = TSPSolver(
    algorithm='mst',
    graph_type='circle',
    graph_params={'num_nodes': 8, 'weight_range': (1, 20)}
)

# Example 2: Simulated Annealing with custom parameters
solver2 = TSPSolver(
    algorithm='sim_annealing',
    graph_type='fully_connected',
    graph_params={'num_nodes': 10, 'seed': 42},
    sa_initial_temp=2000,
    sa_cooling_rate=0.005,
    sa_max_iter=5000
)

# Example 3: QAOA with custom parameters
solver3 = TSPSolver(
    algorithm='qaoa',
    graph_type='circle',
    graph_params={'num_nodes': 4},
    qaoa_layers=3,
    qaoa_learning_rate=0.02,
    qaoa_optimization_steps=300
)

# Example 4: Partially connected graph
solver4 = TSPSolver(
    algorithm='mst',
    graph_type='partially_connected',
    graph_params={
        'num_nodes': 12,
        'connectivity': 0.7,  # 70% of possible edges
        'weight_range': (5, 50),
        'seed': 123
    }
)

# Example 5: With visualization enabled
solver5 = TSPSolver(
    algorithm='mst',
    graph_type='circle',
    graph_params={'num_nodes': 6},
    visualize=True  # Will show graph and solution
)

# Example 6: Using custom graph
solver6 = TSPSolver(
    algorithm='sim_annealing',
    graph=[[0, 10, 15], [10, 0, 20], [15, 20, 0]],
    visualize=False
)

# ============================================================================
# USAGE
# ============================================================================

# After initialization, solve the problem:
result = solver2.solve()

# Access results:
print(f"Path: {result.path}")           # List of node indices
print(f"Cost: {result.cost}")           # Total tour cost
print(f"Algorithm: {result.algorithm}") # Algorithm used
print(f"Metadata: {result.metadata}")  # Additional info (time, etc.)

# Optional: Visualize separately (if visualize=False during init)
solver2.visualize_result()  # Show solution
solver2.visualize_graph()    # Show just the graph

# Get result without re-solving
last_result = solver2.get_result()

