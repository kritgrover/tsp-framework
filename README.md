# TSP Solver Framework

A unified, open-source framework for solving the Traveling Salesman Problem (TSP) using multiple classical and quantum-inspired algorithms. This project provides a comprehensive comparative implementation study of four distinct TSP-solving approaches with a focus on implementation-level trade-offs, scalability, and practical performance.

## Table of Contents

- [Introduction](#introduction)
- [Primary Objectives](#primary-objectives)
- [Algorithms Implemented](#algorithms-implemented)
- [Framework Architecture](#framework-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology and Design Choices](#methodology-and-design-choices)
- [Contributing](#contributing)

## Introduction

The Traveling Salesman Problem (TSP) is one of the most well-known and extensively studied challenges in combinatorial optimization. The problem statement is simple: given a list of cities and a starting city, what is the shortest possible route for a salesman to visit all cities exactly once and return to the starting point? Despite its simple formulation, the TSP is NP-hard, and exact solutions are computationally infeasible for large inputs due to the factorial growth of possible routes.

This research fills a critical gap in the literature by conducting a comparative implementation study and providing an open-source framework for four distinct TSP-solving approaches. While many studies emphasize either algorithmic theory or abstract numerical benchmarks, this project focuses on implementation-level trade-offs, particularly when comparing quantum and classical solvers in a unified experimental framework.

## Primary Objectives

1. **Comparative Analysis**: Evaluate four distinct TSP-solving approaches based on performance metrics such as execution time, solution quality, and scalability with varying problem sizes.

2. **Implementation Trade-offs**: Focus on the trade-offs encountered during implementation, from memory usage to algorithmic bottlenecks, and highlight the engineering decisions made to optimize or adapt each approach for practical use.

3. **Empirical Validation**: Address the ongoing debate about the suitability of quantum methods (particularly QAOA) for TSP by conducting hands-on, empirical comparisons of classical and quantum solvers, focusing on implementation-level trade-offs and scalability challenges.

4. **Open-Source Framework**: Provide a unified, extensible framework that allows researchers and practitioners to easily compare different algorithms, generate various graph types, and visualize solutions.

## Algorithms Implemented

The framework implements four distinct TSP-solving approaches:

### 1. Brute Force Search

The brute force approach represents the most straightforward method for solving the TSP, examining all possible permutations of city visits to identify the optimal route. While this guarantees finding the global optimum, for an n-city problem, the brute force method requires evaluating n! potential solutions, which quickly becomes prohibitive and limits its practical application to small problem instances (typically n ≤ 10 cities).

**Key Characteristics:**
- **Optimality**: Guarantees optimal solution
- **Time Complexity**: O(n!)
- **Use Case**: Baseline for validating correctness of heuristic and quantum approaches
- **Limitations**: Only practical for small instances (n ≤ 10)

### 2. Minimum Spanning Tree (MST) 2-Approximation

The MST 2-approximation algorithm offers a polynomial-time approach to the TSP, providing solutions guaranteed to be within a factor of 2 of the optimal solution. This method constructs a minimum spanning tree of the graph and then performs a preorder traversal to create a Hamiltonian cycle.

**Key Characteristics:**
- **Optimality**: 2-approximation guarantee (solution within 2× optimal)
- **Time Complexity**: O(E log V) for MST construction, O(V) for traversal
- **Use Case**: Fast, scalable solutions for large problem instances
- **Trade-off**: Speed vs. solution quality

### 3. Simulated Annealing (SA)

Simulated annealing emerged as a powerful metaheuristic for addressing the TSP, drawing inspiration from the physical annealing process in metallurgy. SA avoids becoming trapped in local minima through a probabilistic acceptance criterion that allows occasional uphill moves, particularly in the early stages of the optimization process.

**Key Characteristics:**
- **Optimality**: Near-optimal solutions (typically very close to optimal)
- **Time Complexity**: O(N²) per iteration, total depends on iterations
- **Use Case**: High-quality solutions with good balance of performance and quality
- **Strengths**: Excellent balance between exploration and exploitation

### 4. Quantum Approximate Optimization Algorithm (QAOA)

The QAOA represents a relatively recent approach to solving combinatorial optimization problems like the TSP, leveraging quantum computing principles to explore solution spaces more efficiently than classical methods in certain scenarios. QAOA operates by preparing a quantum state in a superposition and then applying a sequence of unitary operations generated by two distinct Hamiltonians: a problem Hamiltonian (H_C), which encodes the problem's cost function, and a mixer Hamiltonian (H_B), which drives transitions between different candidate solutions.

**Key Characteristics:**
- **Optimality**: Variable, depends on parameter tuning and problem size
- **Time Complexity**: Exponential in classical simulation
- **Use Case**: Research into quantum algorithms for combinatorial optimization
- **Challenges**: Classical precomputation bottleneck, simulation memory costs

## Framework Architecture

The framework is designed with modularity, extensibility, and ease of use in mind. The architecture consists of several key components:

### Core Components

1. **TSPSolver Class** (`tsp_solver.py`): Main interface that orchestrates algorithm selection, graph generation/input, parameter management, and result standardization.

2. **Graph Generators** (`utils/graph_generators.py`): Modular functions for generating different graph types:
   - Fully connected graphs
   - Partially connected graphs (with configurable connectivity)
   - Circle graphs (each node connected to 2 neighbors)

3. **Algorithm Adapters** (`utils/algorithm_adapter.py`): Wrapper functions that standardize algorithm interfaces, normalizing inputs and outputs across all algorithms.

4. **Unified Visualizer** (`utils/visualizer.py`): Centralized visualization module providing consistent styling and functionality across all algorithms.

5. **Result Class** (`utils/result.py`): Standardized result container (`TSPSolution`) containing path, cost, algorithm metadata, and execution information.

### Design Principles

- **Modularity**: Each algorithm is implemented independently and can be easily extended or replaced
- **Standardization**: All algorithms return consistent `TSPSolution` objects with standardized metadata
- **Flexibility**: Support for custom graphs or automatic generation of various graph types
- **Extensibility**: Easy to add new algorithms or graph generators
- **User-Friendly**: Simple, intuitive API with comprehensive parameter documentation

## Installation

### Requirements

- Python 3.8+
- NetworkX
- Matplotlib
- NumPy
- PennyLane (for QAOA algorithm)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd tsp-framework

# Install dependencies
pip install networkx matplotlib numpy pennylane
```

## Usage

### Basic Usage

```python
from tsp_solver import TSPSolver

# Create a solver with MST algorithm on a circle graph
solver = TSPSolver(
    algorithm='mst',
    graph_type='circle',
    graph_params={'num_nodes': 8, 'weight_range': (1, 20)},
    visualize=False
)

# Solve the problem
result = solver.solve()

# Access results
print(f"Path: {result.path}")
print(f"Cost: {result.cost}")
print(f"Time: {result.metadata.get('time_taken', 0):.4f}s")
```

### Using Custom Graphs

```python
# Define your own distance matrix
custom_graph = [
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
]

solver = TSPSolver(
    algorithm='sim_annealing',
    graph=custom_graph,
    visualize=True
)

result = solver.solve()
```

### Algorithm-Specific Parameters

```python
# Simulated Annealing with custom parameters
solver = TSPSolver(
    algorithm='sim_annealing',
    graph_type='fully_connected',
    graph_params={'num_nodes': 10},
    sa_initial_temp=2000,
    sa_cooling_rate=0.005,
    sa_max_iter=5000
)

# QAOA with custom parameters
solver = TSPSolver(
    algorithm='qaoa',
    graph_type='circle',
    graph_params={'num_nodes': 4},
    qaoa_layers=3,
    qaoa_learning_rate=0.02,
    qaoa_optimization_steps=300
)
```

See `main.py` for comprehensive usage examples and parameter documentation.

## Methodology and Design Choices

### Brute Force Implementation

#### Initial Design
The brute force implementation uses Python's `itertools.permutations()` to generate all possible routes through the cities. For an n-city problem, the algorithm evaluates (n-1)! permutations, as the starting city is fixed to eliminate rotationally equivalent solutions.

#### Performance Optimizations
Several performance optimizations were implemented:

1. **Graph Data Precomputation**: Precomputing all graph-related data once rather than recalculating it for each iteration. The original implementation repeatedly called expensive NetworkX functions like `spring_layout()` and `get_edge_attributes()`, which are O(n²) and O(E) operations respectively. The optimized version calculates the graph layout once with a fixed seed and creates a bidirectional edge weight lookup dictionary for O(1) access.

2. **Fast Path Cost Calculation**: Replaced the iterator-based approach using `zip()` with direct array indexing, providing approximately 40% faster path cost computation. The optimized function also implements early termination when encountering invalid edges, which can skip 50% or more of remaining edge checks in sparse graphs.

3. **Memory Management**: The optimized implementation eliminates repeated memory allocation by pre-allocating a path buffer array. For an 8-city problem, this saves approximately 40,000 list creations and reduces garbage collection pressure by 70%.

4. **Configurable Visualization Updates**: To address the performance overhead of continuous visualization, we introduced an `update_frequency` parameter allowing users to skip visualization updates. This provides linear performance scaling.

### MST 2-Approximation Implementation

#### Algorithm Design
The implementation follows the canonical MST-based 2-approximation algorithm:

1. **MST Construction**: Use Prim's algorithm to find the minimum spanning tree of the input graph.
2. **Tree Representation**: Convert the MST edge list into an adjacency list for efficient traversal.
3. **DFS Traversal**: Perform depth-first search preorder traversal starting from node 0.
4. **Cycle Formation**: Create the Hamiltonian cycle using the DFS ordering and return to the starting node.

#### Design Decisions

1. **Choice of Prim's Algorithm**: This decision was made because Prim's algorithm naturally maintains connectivity from a starting vertex, which aligns with the requirement of starting from a specific city.

2. **Adjacency List Conversion**: After MST construction, the edge list is converted to an adjacency list representation using Python's `defaultdict`. This design decision optimizes the subsequent DFS traversal by providing O(1) neighbor access rather than requiring O(n²) matrix scanning for each vertex during traversal.

3. **Bidirectional Edge Storage**: The adjacency list stores edges in both directions (u→v and v→u) since the MST is undirected. This eliminates the need for complex edge direction checking during DFS traversal.

4. **Iterative DFS Implementation**: Uses an iterative implementation to avoid recursion depth limits, with neighbors added to the stack in reverse order to maintain correct traversal order.

### Simulated Annealing Implementation

#### Basic Structure
The Simulated Annealing implementation follows the standard canonical structure:

1. **Initialization**: The algorithm uses a multi-strategy approach to ensure a valid starting path. It sequentially attempts: (1) a Greedy Nearest Neighbor search, (2) Random Valid Connections, and (3) Backtracking. If all fail (e.g., on extremely sparse graphs), it falls back to a randomized path to ensure execution continues.

2. **Neighbor Generation**: A neighbor solution is generated using a topology-aware strategy. The algorithm first attempts to swap a city with one of its 'n-hop' neighbors (cities connected within a short distance in the graph) to preserve local structure. If this fails to produce a valid tour, it falls back to a simple adjacent swap mechanism.

3. **Acceptance Criterion**: The algorithm always accepts solutions that lower the cost. Worse solutions are accepted probabilistically based on the Metropolis criterion: P(accept) = exp(-delta_C / T), where delta_C is the cost increase and T is the current temperature.

4. **Cooling Schedule**: An exponential decay schedule is used where the temperature is multiplied by a cooling rate factor (typically 0.995 or similar) at each iteration.

5. **Early Stopping**: The algorithm terminates if the temperature drops below a minimum threshold (0.1) or if no improvement is found for a specified number of consecutive iterations (default 2000).

#### Optimizations and Design Decisions

1. **Memory Management and Data Structures**: Reusable data structures were used to minimize repetitive object creation, reducing memory allocations by approximately 80%. Adjacency matrices were preprocessed into adjacency lists for O(1) neighbor lookups, dramatically accelerating neighbor generation and validation.

2. **Algorithmic Efficiency**: Path validation and distance calculation utilize early termination to avoid unnecessary computation when encountering invalid edges.

3. **Robust Initial Path Generation**: Instead of relying solely on random shuffles (which often fail on sparse graphs), a multi-strategy approach was used:
   - Greedy nearest-neighbor heuristics
   - Random valid connections
   - Backtracking and restarts
   - Pure randomization as a last resort

4. **Advanced Neighbor Generation**: Targeted swaps based on graph topology using adjacency lists and n-hop neighbors replaced simple random swaps, improving both solution quality and convergence speed. In-place swaps and reusable sets reduced garbage collection and overhead.

5. **Pre-computation and Cache Efficiency**: Adjacency lists and reusable sets were precomputed, and single-pass algorithms were favored to improve cache locality and reduce per-iteration time complexity from O(n²) to O(k), where k is the number of relevant neighbors.

6. **Visualization and User Feedback**: Visualization routines were separated from the main algorithm to avoid unnecessary overhead, with configurable update intervals to balance insight and performance.

7. **Parameterization and Flexibility**: All key parameters (initial temperature, cooling rate, maximum iterations, neighbor distance, update intervals) were exposed for user tuning, making the implementation adaptable to a wide range of TSP instances.

### QAOA Implementation

#### Core Design Choices

Our implementation of QAOA for the TSP is fundamentally guided by the methodology proposed in research literature, which introduces a resource-efficient edge-to-qubit problem mapping and a novel approach to handling constraints, as opposed to approaches that require twice the number of qubits and the additional complexity of penalty terms.

1. **Edge to Qubit Mapping**: The TSP graph is encoded by mapping each edge to a dedicated qubit. For a graph with n cities, this requires n(n-1)/2 qubits, a more efficient mapping than alternative schemes that require n² qubits. A tour is represented as a bitstring where a '1' indicates the corresponding edge is part of the tour.

2. **Problem Hamiltonian**: The cost function is encoded in the problem Hamiltonian H_C. This is constructed as a sum of Pauli-Z operators, where each Z_i operator acts on the qubit corresponding to an edge i. The coefficient of each Z_i term is the weight of that edge. Minimizing the expectation value of this Hamiltonian is equivalent to finding the tour with the minimum total distance.

3. **Constraint Handling via State Initialization**: A key design decision is to embed the TSP constraints directly into the initial state of the algorithm rather than using penalty terms in the Hamiltonian. This is achieved through a two-step process:
   - **Classical Pre-computation of Feasible Solutions**: The implementation first performs an exhaustive classical search using NetworkX to find all possible Hamiltonian cycles in the graph. Each valid tour is converted into its corresponding bit string representation.
   - **Superposition of Valid Tours**: The quantum circuit is initialized in a state that is an equal superposition of only the valid bit strings found in the previous step. This is accomplished using `qml.StatePrep`, which directly prepares the specific state vector. This technique strictly confines the quantum search to the feasible subspace of valid solutions from the outset.

4. **Constraint-Encoded Mixer Hamiltonian (H_B)**: The implementation uses a 2-opt constraint-preserving mixer that performs transitions between valid tours. The mixer, composed of terms that implement 2-opt swaps, has the property of preserving valid Hamiltonian cycles. This mixer serves as an effective proxy, encouraging exploration primarily among states that represent valid tours.

5. **Optimization and Solution Extraction**: The algorithm proceeds by applying p layers of the QAOA ansatz, alternating the evolution under H_C and H_B. An AdamOptimizer is used to classically tune the rotation angles (γ, β) for each layer to minimize the cost. After optimization, the final state's probabilities are calculated. The solution is the valid bitstring with the highest measured probability.

#### Trade-offs and Scalability

The implementation of QAOA within a classical simulation environment introduces significant trade-offs and scalability challenges:

1. **Classical Pre-computation Bottleneck**: The most severe limitation is the classical precomputation step required to identify all Hamiltonian cycles. This exhaustive search has a factorial time complexity relative to the number of cities, making the entire approach intractable for graphs larger than a few nodes. This finding empirically confirms concerns raised in recent research, which argue that many proposed quantum solutions are hampered by significant classical processing that is required either before or during the quantum algorithm, making claims of quantum advantage problematic. While this method ensures our quantum simulation operates only on valid tours, it is fundamentally bound by classical intractability from the outset.

2. **Simulation Memory Cost**: Simulating the quantum system is itself exponentially expensive. The state vector required for the simulation has a size of 2^N, where N is the number of qubits that is n(n-1)/2 for an n-city graph. This memory requirement grows rapidly, making simulations for even moderately sized TSPs infeasible on classical hardware.

3. **Approximation Quality vs. Circuit Depth**: As with any QAOA implementation, the quality of the solution depends on the number of layers, p. A larger p allows the algorithm to explore the solution space more thoroughly and can lead to better approximation ratios. However, this comes at the cost of a deeper quantum circuit, more parameters for the classical optimizer to handle, and longer simulation times.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. Areas for contribution include:

- Additional TSP algorithms
- New graph generation methods
- Performance optimizations
- Documentation improvements
- Test coverage

## Citation

If you use this framework in your research, please cite:

```bibtex
A Comparative Study on Classical and Quantum Approaches to the Traveling Salesman Problem. Krit Grover, Marcelo Ponce, 2025.
```

## Acknowledgments

This project was inspired by the ongoing debate about quantum advantage in combinatorial optimization and the need for empirical, implementation-level comparisons of classical and quantum TSP solvers. The QAOA implementation in specific is based off the work of [Ruan et. al, 2020](https://www.researchgate.net/publication/341047640_The_Quantum_Approximate_Algorithm_for_Solving_Traveling_Salesman_Problem).

