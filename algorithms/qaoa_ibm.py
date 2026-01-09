import numpy as np
from itertools import combinations
from typing import List

# Qiskit Core
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# Qiskit Algorithms & Optimization
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA

# IBM Runtime (no Session needed for Open Plan)
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

class QAOATSPSolver:
    def __init__(self, distance_matrix, 
                num_layers=2, optimization_steps=100,
                use_real_hardware=True):
        """
        Initialize QAOA TSP Solver.
        
        Args:
            distance_matrix: NxN distance matrix between cities
            num_layers: Number of QAOA layers (reps)
            optimization_steps: Max iterations for optimizer
            use_real_hardware: If True, runs on IBM Quantum hardware; else local simulation
        """
        self.dist = np.array(distance_matrix)
        self.num_cities = len(distance_matrix)
        self.n_layers = num_layers
        self.steps = optimization_steps
        self.use_real_hardware = use_real_hardware

        # 1. Edge Mapping (Paper Section 3.1)
        # N cities -> N(N-1)/2 qubits.
        self.edges = list(combinations(range(self.num_cities), 2))
        self.num_qubits = len(self.edges)
        
        print(f"Initializing Solver: {self.num_cities} Cities -> {self.num_qubits} Qubits")
        
        # 2. Precompute Subspace (Valid Hamiltonian Cycles)
        self.valid_bitstrings = self._get_valid_cycles()
        if not self.valid_bitstrings:
            raise ValueError("No valid tours found (check inputs).")
            
        print(f"Subspace size: {len(self.valid_bitstrings)} valid tours found.")

        # 3. Build Operators
        self.cost_op = self._build_cost_operator()
        self.mixer_op = self._build_subspace_mixer()
        self.initial_state = self._build_initial_state()

    def _get_valid_cycles(self) -> List[str]:
        """
        Generate all valid Hamiltonian cycles efficiently.
        
        Instead of checking all C(E, N) edge combinations (exponential),
        we directly generate cycles by permuting intermediate nodes.
        This produces exactly (N-1)!/2 unique cycles.
        
        Based on Algorithm 1 from Ruan et al. (2020):
        "The Quantum Approximate Algorithm for Solving Traveling Salesman Problem"
        
        """
        from itertools import permutations
        
        valid = set()  # Use set to avoid duplicates
        n = self.num_cities
        
        # Fix starting node as 0 to avoid rotational duplicates
        # Generate all permutations of remaining nodes
        other_nodes = list(range(1, n))
        
        for perm in permutations(other_nodes):
            # Build cycle: 0 -> perm[0] -> perm[1] -> ... -> perm[-1] -> 0
            cycle = [0] + list(perm)
            
            # Convert cycle to edge set
            edges = []
            for i in range(n):
                u, v = cycle[i], cycle[(i + 1) % n]
                edges.append(tuple(sorted((u, v))))
            
            # Convert to bitstring
            bs = self._edges_to_bitstring(edges)
            valid.add(bs)
        
        return list(valid)
    
    def _edges_to_bitstring(self, edges) -> str:
        """Convert a list of edges to a bitstring representation."""
        # Create edge to index lookup
        edge_to_idx = {tuple(sorted(e)): i for i, e in enumerate(self.edges)}
        
        bs = ['0'] * self.num_qubits
        for edge in edges:
            idx = edge_to_idx[tuple(sorted(edge))]
            bs[idx] = '1'
        
        # Reverse for Qiskit's standard "q_n ... q_0" representation
        return "".join(bs[::-1])
    
    def _validate_bitstring(self, bitstring: str) -> bool:
        """
        Algorithm 1 from Ruan et al. (2020): O(N²) validation.
        
        Checks if a bitstring represents a valid Hamiltonian cycle:
        1. Exactly N edges selected (N = num_cities)
        2. Each vertex has degree exactly 2
        3. Edges form a single connected cycle
        
        Args:
            bitstring: Binary string in Qiskit format (reversed)
            
        Returns:
            True if valid Hamiltonian cycle, False otherwise
        """
        n = self.num_cities
        
        # Reverse bitstring to match edge indexing
        bs = bitstring[::-1]
        
        # Count selected edges and build adjacency
        selected_count = 0
        degrees = [0] * n
        adj = [[] for _ in range(n)]
        
        for i, bit in enumerate(bs):
            if i >= len(self.edges):
                break
            if bit == '1':
                selected_count += 1
                u, v = self.edges[i]
                degrees[u] += 1
                degrees[v] += 1
                adj[u].append(v)
                adj[v].append(u)
        
        # Check 1: Must have exactly N edges
        if selected_count != n:
            return False
        
        # Check 2: Every vertex must have degree 2
        if any(d != 2 for d in degrees):
            return False
        
        # Check 3: Must form single connected cycle (DFS from node 0)
        visited = [False] * n
        stack = [0]
        visit_count = 0
        
        while stack:
            node = stack.pop()
            if visited[node]:
                continue
            visited[node] = True
            visit_count += 1
            for neighbor in adj[node]:
                if not visited[neighbor]:
                    stack.append(neighbor)
        
        return visit_count == n

    def _is_cycle(self, edges):
        """
        Check if edges form a single Hamiltonian cycle. O(N²) complexity.
        Used for result validation after quantum measurement.
        """
        n = self.num_cities
        
        # Check edge count
        if len(edges) != n:
            return False
        
        # Check degree constraint (every node degree 2)
        degrees = [0] * n
        adj = [[] for _ in range(n)]
        
        for u, v in edges:
            degrees[u] += 1
            degrees[v] += 1
            adj[u].append(v)
            adj[v].append(u)
        
        if any(d != 2 for d in degrees):
            return False
        
        # Check connectivity using DFS (O(N))
        visited = [False] * n
        stack = [0]
        visit_count = 0
        
        while stack:
            node = stack.pop()
            if visited[node]:
                continue
            visited[node] = True
            visit_count += 1
            for neighbor in adj[node]:
                if not visited[neighbor]:
                    stack.append(neighbor)
        
        return visit_count == n

    def _build_cost_operator(self) -> SparsePauliOp:
        """H_C = sum(w_ij * x_ij). Maps x_ij -> (I - Z)/2."""
        paulis = []
        coeffs = []
        
        for idx, (u, v) in enumerate(self.edges):
            weight = self.dist[u][v]
            
            # Z term
            z_str = ['I'] * self.num_qubits
            z_str[self.num_qubits - 1 - idx] = 'Z' # Reverse index
            
            paulis.append("".join(z_str))
            coeffs.append(-weight / 2.0)
            
        return SparsePauliOp(paulis, coeffs)

    def _build_subspace_mixer(self) -> SparsePauliOp:
        """
        2-opt constraint-preserving mixer with proper Pauli decomposition.
        Matches the PennyLane implementation from qaoa.py.
        
        Each 2-opt swap between disjoint edges generates 8 Pauli string terms
        that implement the transition operator between valid tour states.
        """
        paulis = []
        coeffs = []
        
        # Build edge lookup for quick index finding
        edge_to_idx = {e: i for i, e in enumerate(self.edges)}
        
        n_e = len(self.edges)
        swap_count = 0
        
        for i in range(n_e):
            for j in range(i + 1, n_e):
                e1 = self.edges[i]
                e2 = self.edges[j]
                
                # Edges must be disjoint for a valid 2-opt swap
                if set(e1).isdisjoint(set(e2)):
                    u, v = e1
                    x, y = e2
                    
                    # Two possible reconnections for 2-opt
                    reconnections = [
                        (tuple(sorted((u, x))), tuple(sorted((v, y)))),
                        (tuple(sorted((u, y))), tuple(sorted((v, x))))
                    ]
                    
                    for ne1, ne2 in reconnections:
                        # Get qubit indices for old and new edges
                        q_old1 = edge_to_idx[e1]
                        q_old2 = edge_to_idx[e2]
                        q_new1 = edge_to_idx.get(ne1)
                        q_new2 = edge_to_idx.get(ne2)
                        
                        # Skip if new edges don't exist in our edge set
                        if q_new1 is None or q_new2 is None:
                            continue
                        
                        # 8 Pauli strings per swap operation (from paper decomposition)
                        c_val = 0.125
                        term_coeffs = [c_val, -c_val, -c_val, c_val, c_val, c_val, c_val, c_val]
                        
                        # Pauli patterns: [new1, new2, old1, old2]
                        pauli_patterns = [
                            'XXXX', 'XXYY', 'YYXX', 'YYYY',
                            'XYXY', 'XYYX', 'YXXY', 'YXYX'
                        ]
                        
                        for coeff, pattern in zip(term_coeffs, pauli_patterns):
                            pauli_str = self._make_4qubit_pauli(
                                q_new1, q_new2, q_old1, q_old2, pattern
                            )
                            paulis.append(pauli_str)
                            coeffs.append(coeff)
                        
                        swap_count += 1
        
        if not paulis:
            print("Warning: Mixer has no transitions (problem too small?).")
            return SparsePauliOp(['I' * self.num_qubits], [0.0])
        
        print(f"Mixer built with {swap_count} swap operations, {len(paulis)} Pauli terms.")
        return SparsePauliOp(paulis, coeffs).simplify()
    
    def _make_4qubit_pauli(self, q1: int, q2: int, q3: int, q4: int, pattern: str) -> str:
        """
        Create a Pauli string with operators at specified qubit positions.
        
        Args:
            q1, q2, q3, q4: Qubit indices for the 4 operators
            pattern: 4-character string like 'XXXX', 'XXYY', etc.
        
        Returns:
            Full Pauli string with I's everywhere except specified positions.
            Uses Qiskit convention: rightmost character is qubit 0.
        """
        # Start with all identity
        pauli_list = ['I'] * self.num_qubits
        
        # Place operators (Qiskit uses reversed indexing: position 0 = rightmost)
        qubit_indices = [q1, q2, q3, q4]
        for q_idx, pauli_char in zip(qubit_indices, pattern):
            # Qiskit convention: qubit i is at position (n-1-i) in string
            str_pos = self.num_qubits - 1 - q_idx
            pauli_list[str_pos] = pauli_char
        
        return ''.join(pauli_list)

    def _build_initial_state(self) -> QuantumCircuit:
        """Uniform superposition of valid bitstrings."""
        qc = QuantumCircuit(self.num_qubits)
        dim = 2 ** self.num_qubits
        vector = np.zeros(dim, dtype=complex)
        
        amp = 1.0 / np.sqrt(len(self.valid_bitstrings))
        for bs in self.valid_bitstrings:
            vector[int(bs, 2)] = amp
            
        qc.initialize(vector, range(self.num_qubits))
        return qc

    def run(self):
        """Execute QAOA on IBM Runtime."""
        
        if self.use_real_hardware:
            print("\n" + "="*60)
            print("  FULL QUANTUM OPTIMIZATION MODE")
            print("  All optimization iterations will run on IBM Quantum hardware")
            print("  WARNING: This will be SLOW due to queue times!")
            print("="*60)
            
            # 1. Connect to IBM Quantum
            print("\n1. Connecting to IBM Quantum...")
            service = QiskitRuntimeService()
            
            print("   Selecting least busy operational backend...")
            backend = service.least_busy(simulator=False, operational=True)
            print(f"   Selected backend: {backend.name}")
            
            # 2. Build the QAOA ansatz
            print("\n2. Building QAOA ansatz...")
            ansatz = QAOAAnsatz(self.cost_op, reps=self.n_layers,
                               initial_state=self.initial_state, mixer_operator=self.mixer_op)
            print(f"   Ansatz has {ansatz.num_parameters} parameters")
            
            # 3. Transpile the parametrized circuit for the backend
            print("\n3. Transpiling circuit for hardware...")
            pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
            isa_ansatz = pm.run(ansatz)
            print(f"   Transpiled circuit: {isa_ansatz.num_qubits} physical qubits, depth {isa_ansatz.depth()}")
            
            # Store the final layout mapping (logical qubit -> physical qubit)
            if isa_ansatz.layout is not None:
                self._final_layout = isa_ansatz.layout.final_index_layout()
                print(f"   Qubit mapping (logical->physical): {self._final_layout[:self.num_qubits]}")
            else:
                self._final_layout = None
            
            # 4. Create sampler for quantum hardware
            sampler = SamplerV2(mode=backend)
            
            # 5. Nelder-Mead optimization on quantum hardware
            print(f"\n4. Starting Nelder-Mead optimization on quantum hardware (max {self.steps} evaluations)...")
            print("   Each function evaluation submits a job to IBM Quantum.")
            
            # Initialize parameters with small random values
            num_params = ansatz.num_parameters
            initial_params = np.random.uniform(0, 0.5, size=num_params)
            
            # Track best solution and evaluation count
            self._eval_count = 0
            self._best_cost = float('inf')
            self._best_params = initial_params.copy()
            self._best_counts = None
            
            def cost_function(params):
                """Objective function evaluated on quantum hardware."""
                self._eval_count += 1
                
                # Build circuit with current parameters
                circuit = isa_ansatz.assign_parameters(params)
                circuit.measure_all()
                
                # Submit job to quantum hardware
                print(f"\n   Evaluation {self._eval_count}: Submitting job...")
                job = sampler.run([circuit], shots=2048)
                print(f"   Job ID: {job.job_id()}")
                print("   Waiting for results...")
                
                result = job.result()
                counts = self._extract_counts(result[0])
                cost = self._compute_expectation_from_counts(counts)
                
                print(f"   Cost: {cost:.4f}")
                
                # Track best solution
                if cost < self._best_cost:
                    self._best_cost = cost
                    self._best_params = params.copy()
                    self._best_counts = counts
                
                return cost
            
            # Run Nelder-Mead optimization
            from scipy.optimize import minimize
            
            result = minimize(
                cost_function,
                initial_params,
                method='Nelder-Mead',
                options={
                    'maxfev': self.steps,  # Maximum function evaluations
                    'xatol': 0.01,         # Parameter tolerance
                    'fatol': 0.01,         # Function value tolerance
                    'adaptive': True       # Adapt algorithm parameters to dimensionality
                }
            )
            
            print(f"\n   Optimization finished after {self._eval_count} evaluations")
            print(f"   Final cost from optimizer: {result.fun:.4f}")
            print(f"   Best cost found: {self._best_cost:.4f}")
            
            # Use best parameters found during optimization
            best_params = self._best_params
            best_cost = self._best_cost
            
            # 5. Final sampling with best parameters (increased shots for accuracy)
            print(f"\n5. Final sampling with best parameters (cost={best_cost:.4f})...")
            final_circuit = isa_ansatz.assign_parameters(best_params)
            final_circuit.measure_all()
            
            # Increased shots for more accurate final sampling
            job = sampler.run([final_circuit], shots=4096)
            print(f"   Job ID: {job.job_id()}")
            print("   Waiting for final results...")
            
            final_result = job.result()
            final_counts = self._extract_counts(final_result[0])
            
            print(f"   Got {len(final_counts)} unique measurement outcomes")
            return self._process_counts(final_counts)
                
        else:
            # Local Simulation fallback using qiskit's built-in sampler
            from qiskit.primitives import StatevectorSampler
            
            print("\n--- RUNNING LOCAL SIMULATION ---")
            self._final_layout = None  # No qubit remapping for local simulation
            local_sampler = StatevectorSampler()
            optimizer = COBYLA(maxiter=self.steps)
            
            qaoa = QAOA(sampler=local_sampler, optimizer=optimizer, reps=self.n_layers, 
                        initial_state=self.initial_state, mixer=self.mixer_op)
            result = qaoa.compute_minimum_eigenvalue(self.cost_op)
            
            # Extract best measurement from SamplingMinimumEigensolverResult
            if hasattr(result, 'best_measurement') and result.best_measurement:
                best_bs = result.best_measurement.get('bitstring', '')
                return self._decode_solution(best_bs)
            
            # Fallback: use eigenstate if available
            if hasattr(result, 'eigenstate') and result.eigenstate is not None:
                # eigenstate is a dict of {bitstring: probability}
                best_bitstring = max(result.eigenstate, key=result.eigenstate.get)
                return self._decode_solution(best_bitstring)
            
            return None, None
    
    def _extract_counts(self, pub_result):
        """Extract measurement counts from a PubResult object."""
        try:
            return pub_result.data.meas.get_counts()
        except AttributeError:
            # Try alternate accessor names
            data_keys = list(pub_result.data.keys())
            if data_keys:
                return getattr(pub_result.data, data_keys[0]).get_counts()
            return {}
    
    def _compute_expectation_from_counts(self, counts):
        """
        Compute the cost Hamiltonian expectation value from measurement counts.
        H_C = sum(-w_ij/2 * Z_ij) where Z_ij = 1 if qubit is |0⟩, -1 if |1⟩
        """
        total_shots = sum(counts.values())
        expectation = 0.0
        
        for bitstring, count in counts.items():
            # Compute cost for this bitstring
            cost = 0.0
            bs_len = len(bitstring)
            
            for logical_idx, (u, v) in enumerate(self.edges):
                weight = self.dist[u][v]
                
                # Map logical qubit to physical qubit using transpilation layout
                if hasattr(self, '_final_layout') and self._final_layout is not None:
                    physical_idx = self._final_layout[logical_idx]
                else:
                    physical_idx = logical_idx  # No remapping for local simulation
                
                # In Qiskit bitstring, physical qubit i is at position (len-1-i)
                bit_pos = bs_len - 1 - physical_idx
                if 0 <= bit_pos < bs_len:
                    bit_val = int(bitstring[bit_pos])
                else:
                    bit_val = 0  # Default if out of range
                    
                # Z eigenvalue: |0⟩ -> +1, |1⟩ -> -1
                z_val = 1 - 2 * bit_val
                cost += (-weight / 2.0) * z_val
            
            expectation += cost * count
        
        return expectation / total_shots

    def _process_counts(self, counts):
        """Find most frequent valid bitstring in counts."""
        # Sort counts by frequency
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        
        for bs, count in sorted_counts:
            # Remap physical qubits back to logical qubits using transpilation layout
            bs_len = len(bs)
            edges = []
            
            for logical_idx in range(self.num_qubits):
                # Map logical qubit to physical qubit
                if hasattr(self, '_final_layout') and self._final_layout is not None:
                    physical_idx = self._final_layout[logical_idx]
                else:
                    physical_idx = logical_idx  # No remapping for local simulation
                
                # physical qubit i is at position (len-1-i)
                bit_pos = bs_len - 1 - physical_idx
                if 0 <= bit_pos < bs_len and bs[bit_pos] == '1':
                    edges.append(self.edges[logical_idx])
            
            if self._is_cycle(edges):
                cost = self._calculate_cost(edges)
                return edges, cost
                
        print("No valid tour found in top results.")
        return None, float('inf')

    def _calculate_cost(self, edges):
        c = 0
        for u, v in edges:
            c += self.dist[u][v]
        return c

    def _decode_solution(self, bs):
        """Decode a bitstring to edges using layout mapping if available."""
        bs_len = len(bs)
        edges = []
        
        for logical_idx in range(self.num_qubits):
            # Map logical qubit to physical qubit
            if hasattr(self, '_final_layout') and self._final_layout is not None:
                physical_idx = self._final_layout[logical_idx]
            else:
                physical_idx = logical_idx  # No remapping for local simulation
            
            # physical qubit i is at position (len-1-i)
            bit_pos = bs_len - 1 - physical_idx
            if 0 <= bit_pos < bs_len and bs[bit_pos] == '1':
                edges.append(self.edges[logical_idx])
                
        return edges, self._calculate_cost(edges)

if __name__ == "__main__":
    # Example 5-city Matrix
    graph = [
        [0, 10, 15, 20, 25],
        [10, 0, 35, 25, 30],
        [15, 35, 0, 30, 35],
        [20, 25, 30, 0, 25],
        [25, 30, 35, 25, 0]
    ]

    try:
        print(f"Running solver with IBM...")
        
        # Initialize the solver
        # Note: Requires QiskitRuntimeService.save_account() to have been run previously
        # with your IBM Quantum API token
        solver = QAOATSPSolver(
            distance_matrix=graph, 
            num_layers=2, 
            optimization_steps=50
        )
        
        # Run the solver
        best_edges, cost = solver.run()
        
        print("\n" + "="*30)
        print("FINAL RESULTS")
        print("="*30)
        
        if best_edges:
            print(f"Edges selected: {best_edges}")
            print(f"Total Cost: {cost}")
            print("Tour found successfully!")
        else:
            print("No valid tour found.")
            print(f"Cost: {cost}")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\nError encountered: {e}")

