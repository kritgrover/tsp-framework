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

This research fills a critical gap in the literature by conducting a comparative implementation study and providing an open-source framework for multiple TSP-solving approaches, including both classical algorithms and quantum implementations. While many studies emphasize either algorithmic theory or abstract numerical benchmarks, this project focuses on implementation-level trade-offs, particularly when comparing quantum and classical solvers in a unified experimental framework.

## Primary Objectives

1. **Comparative Analysis**: Evaluate 5 TSP-solving approaches based on performance metrics such as execution time, solution quality, and scalability with varying problem sizes.

2. **Implementation Trade-offs**: Focus on the trade-offs encountered during implementation, from memory usage to algorithmic bottlenecks, and highlight the engineering decisions made to optimize or adapt each approach for practical use.

3. **Empirical Validation**: Address the ongoing debate about the suitability of quantum methods (particularly QAOA) for TSP by conducting hands-on, empirical comparisons of classical and quantum solvers, focusing on implementation-level trade-offs and scalability challenges.

4. **Open-Source Framework**: Provide a unified, extensible framework that allows researchers and practitioners to easily compare different algorithms, generate various graph types, and visualize solutions.

## Algorithms Implemented

The framework implements five distinct TSP-solving approaches:

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

### 4. Quantum Approximate Optimization Algorithm (QAOA) - PennyLane Simulation

The QAOA represents a relatively recent approach to solving combinatorial optimization problems like the TSP, leveraging quantum computing principles to explore solution spaces more efficiently than classical methods in certain scenarios. QAOA operates by preparing a quantum state in a superposition and then applying a sequence of unitary operations generated by two distinct Hamiltonians: a problem Hamiltonian (H_C), which encodes the problem's cost function, and a mixer Hamiltonian (H_B), which drives transitions between different candidate solutions.

**Key Characteristics:**
- **Optimality**: Variable, depends on parameter tuning and problem size
- **Time Complexity**: Exponential in classical simulation
- **Use Case**: Research into quantum algorithms for combinatorial optimization
- **Challenges**: Classical precomputation bottleneck, simulation memory costs

### 5. QAOA on IBM Quantum Hardware

Building on the theoretical QAOA framework, this implementation provides the capability to run QAOA on real IBM Quantum hardware via the Qiskit Runtime service. This enables true quantum execution rather than classical simulation, offering insights into real-world quantum computing performance and noise characteristics.

**Key Characteristics:**
- **Execution**: Real quantum hardware via IBM Quantum Network
- **Qubits**: N(N-1)/2 qubits for N-city problem (edge-based encoding)
- **Optimizer**: Nelder-Mead optimization with quantum circuit evaluation
- **Use Case**: Empirical validation of quantum algorithms on NISQ devices
- **Features**: Automatic backend selection, circuit transpilation, qubit mapping

**Hardware-Specific Considerations:**
- Queue times vary based on IBM Quantum system availability
- Circuit depth affects noise accumulation on real hardware
- Transpilation optimizes circuits for specific backend topology
- Results include both optimization trajectory and final sampling

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
- PennyLane (for QAOA simulation)
- Qiskit & IBM Runtime (for QAOA on IBM Quantum hardware)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd tsp-framework

# Install core dependencies
pip install networkx matplotlib numpy pennylane

# For IBM Quantum hardware support (optional)
pip install qiskit qiskit-algorithms qiskit-ibm-runtime scipy
```

### IBM Quantum Configuration (Optional)

To run QAOA on real IBM Quantum hardware, you need an IBM Quantum account:

1. Create a free account at [IBM Quantum](https://quantum.ibm.com/)
2. Create an instance
3. Get your API token from your account settings
4. Save your credentials:

```python
from qiskit_ibm_runtime import QiskitRuntimeService
 
QiskitRuntimeService.save_account(
    token="API_KEY",
    instance="CRN",
    set_as_default = True
    )
 
service = QiskitRuntimeService()
```

Run this script in a separate file to configure the Runtime Service once and then you can go ahead and run the `qaoa_ibm.py` file. You should be able to see the run in the [Workloads](https://quantum.cloud.ibm.com/workloads) page on the IBM Quantum Platform.

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

### Running QAOA on IBM Quantum Hardware

The IBM Quantum implementation can be used directly for research and experimentation:

```python
from algorithms.qaoa_ibm import QAOATSPSolver

# Define distance matrix
graph = [
    [0, 10, 15, 20, 25],
    [10, 0, 35, 25, 30],
    [15, 35, 0, 30, 35],
    [20, 25, 30, 0, 25],
    [25, 30, 35, 25, 0]
]

# Run on real IBM Quantum hardware
solver = QAOATSPSolver(
    distance_matrix=graph,
    num_layers=2,              # QAOA circuit depth
    optimization_steps=50,     # Max optimizer evaluations
)

best_edges, cost = solver.run()
print(f"Best tour edges: {best_edges}")
print(f"Total cost: {cost}")
```

**IBM Quantum Parameters:**
- `num_layers`: Number of QAOA layers (circuit depth). Higher values may improve solution quality but increase noise on real hardware.
- `optimization_steps`: Maximum number of Nelder-Mead optimizer evaluations. Each evaluation submits a job to IBM Quantum.
- `use_real_hardware`: When `True`, runs on IBM Quantum hardware; when `False`, uses local Qiskit simulation.

**Important Notes:**
- Each optimization iteration submits a job to IBM Quantum, which incurs queue wait times
- The solver automatically selects the least busy operational backend
- Circuit transpilation is performed at optimization level 3 for hardware compatibility
- Results include logical-to-physical qubit mapping for debugging

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

### QAOA IBM Quantum Implementation

#### Design Philosophy

The IBM Quantum implementation (`qaoa_ibm.py`) adapts the theoretical QAOA framework for execution on real NISQ (Noisy Intermediate-Scale Quantum) devices. This implementation prioritizes practical execution over simulation fidelity, incorporating hardware-aware optimizations and robust error handling.

#### Core Architecture

1. **Edge-Based Qubit Encoding**: Following the same resource-efficient mapping as the PennyLane implementation, each edge in the TSP graph maps to a dedicated qubit. For N cities, this requires N(N-1)/2 qubits. This encoding is more efficient than node-based approaches requiring N² qubits.

2. **Valid Cycle Generation**: The implementation directly generates valid Hamiltonian cycles by permuting intermediate nodes (fixing node 0 as the start). This produces exactly (N-1)!/2 unique cycles, avoiding the exponential blowup of checking all C(E,N) edge combinations.

3. **Cost Hamiltonian Construction**: The cost operator H_C is built as a sum of weighted Pauli-Z operators:
   ```
   H_C = Σ (-w_ij/2) * Z_ij
   ```
   where w_ij is the distance between cities i and j, and Z_ij acts on the qubit representing edge (i,j).

4. **2-opt Constraint-Preserving Mixer**: The mixer Hamiltonian implements 2-opt swap operations that preserve valid tour structure. Each swap between disjoint edges generates 8 Pauli string terms that implement the transition operator. This ensures the quantum evolution stays within the feasible subspace.

5. **Initial State Preparation**: A uniform superposition over all valid Hamiltonian cycles is prepared using state vector initialization. This constrains the quantum search to valid solutions from the outset.

#### Hardware Execution Pipeline

1. **Backend Selection**: The solver automatically queries IBM Quantum for the least busy operational backend, balancing queue time against system availability.

2. **Circuit Transpilation**: The QAOA ansatz is transpiled using Qiskit's preset pass manager at optimization level 3, which:
   - Maps logical qubits to physical qubits based on backend topology
   - Decomposes gates into the backend's native gate set
   - Optimizes circuit depth to minimize noise accumulation

3. **Qubit Layout Tracking**: The transpilation layout is stored to correctly interpret measurement results. Physical qubit indices may differ from logical indices after transpilation.

4. **Nelder-Mead Optimization**: Unlike gradient-based optimizers, Nelder-Mead is derivative-free and well-suited for noisy quantum objective functions. Each function evaluation:
   - Assigns current parameters to the transpiled ansatz
   - Submits a job to IBM Quantum (2048 shots)
   - Computes the cost Hamiltonian expectation value from measurement counts
   - Tracks the best solution found during optimization

5. **Final Sampling**: After optimization converges, a final sampling run with increased shots (4096) provides more accurate solution extraction.

#### Trade-offs and Considerations

1. **Queue Time vs. Iterations**: Each optimizer iteration requires a separate job submission to IBM Quantum. With typical queue times of minutes per job, optimizations with many iterations can take hours. The implementation balances this by using adaptive termination criteria.

2. **Noise vs. Circuit Depth**: Deeper circuits (more QAOA layers) can theoretically provide better solutions but accumulate more noise on real hardware. The default of 2 layers provides a practical balance.

3. **Shot Count Trade-off**: More shots per circuit evaluation provide more accurate expectation values but increase job execution time. The implementation uses 2048 shots during optimization and 4096 for final sampling.

4. **Measurement Result Processing**: Results are post-processed to:
   - Remap physical qubit measurements back to logical qubits
   - Validate that measured bitstrings represent valid Hamiltonian cycles
   - Extract the minimum-cost valid tour from measurement statistics

#### Comparison with Simulation

| Aspect | PennyLane Simulation | IBM Quantum Hardware |
|--------|---------------------|---------------------|
| Execution | Classical computer | Real quantum processor |
| Noise | None (ideal) | Hardware noise present |
| Speed | Fast iteration | Queue + execution time |
| Scalability | Memory-limited | Qubit-limited |
| Results | Deterministic | Probabilistic |
| Use Case | Algorithm development | Empirical validation |

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

This project was inspired by the ongoing debate about quantum advantage in combinatorial optimization and the need for empirical, implementation-level comparisons of classical and quantum TSP solvers. The QAOA implementation is based on the work of [Ruan et al., 2020](https://www.researchgate.net/publication/341047640_The_Quantum_Approximate_Algorithm_for_Solving_Traveling_Salesman_Problem).

The quantum implementation utilizes the [Qiskit](https://qiskit.org/) and [Pennylane](https://pennylane.ai/) frameworks and [IBM Quantum Platform](https://quantum.ibm.com/) hardware access provided by IBM Quantum.

