import numpy as np
import networkx as nx
from itertools import combinations
from typing import List, Tuple

# Qiskit Core
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# Qiskit Algorithms & Optimization
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import SPSA, COBYLA

# IBM Runtime (no Session needed for Open Plan)
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

class QAOATSPSolver:
    def __init__(self, distance_matrix, num_layers=2, optimization_steps=100, 
                 backend_name=None, use_real_hardware=False, use_least_busy=True):
        """
        Initialize QAOA TSP Solver.
        
        Args:
            distance_matrix: NxN distance matrix between cities
            num_layers: Number of QAOA layers (reps)
            optimization_steps: Max iterations for optimizer
            backend_name: Specific IBM backend name (e.g., 'ibm_brisbane'). 
                         If None and use_least_busy=True, selects least busy backend.
            use_real_hardware: If True, runs on IBM Quantum hardware; else local simulation
            use_least_busy: If True and backend_name is None, automatically selects 
                           the least busy operational backend
        """
        self.dist = np.array(distance_matrix)
        self.num_cities = len(distance_matrix)
        self.n_layers = num_layers
        self.steps = optimization_steps
        self.backend_name = backend_name
        self.use_real_hardware = use_real_hardware
        self.use_least_busy = use_least_busy
        
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
        """Find all valid Hamiltonian cycles (brute-force for small N)."""
        valid = []
        # A tour must have exactly N edges
        for edge_indices in combinations(range(self.num_qubits), self.num_cities):
            chosen_edges = [self.edges[i] for i in edge_indices]
            if self._is_cycle(chosen_edges):
                # Create bitstring (little-endian: q0 is rightmost)
                # We map index i -> qubit i.
                bs = ['0'] * self.num_qubits
                for idx in edge_indices:
                    bs[idx] = '1'
                # Reverse for Qiskit's standard "q_n ... q_0" representation
                valid.append("".join(bs[::-1]))
        return valid

    def _is_cycle(self, edges):
        """Check if edges form a single Hamiltonian cycle."""
        # Check degree constraint (every node degree 2)
        degrees = {i: 0 for i in range(self.num_cities)}
        for u, v in edges:
            degrees[u] += 1
            degrees[v] += 1
        if any(d != 2 for d in degrees.values()):
            return False
        
        # Check connectivity (single loop)
        g = nx.Graph()
        g.add_edges_from(edges)
        return nx.is_connected(g) and g.number_of_nodes() == self.num_cities

    def _build_cost_operator(self) -> SparsePauliOp:
        """H_C = sum(w_ij * x_ij). Maps x_ij -> (I - Z)/2."""
        paulis = []
        coeffs = []
        
        for idx, (u, v) in enumerate(self.edges):
            weight = self.dist[u][v]
            # Term: weight * (I - Z_i)/2
            # We ignore the constant part (weight/2 * I) for optimization
            
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
                        # Coefficient 1/8 = 0.125 for each term
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
            # Hybrid approach: optimize locally (fast), then sample on real hardware
            # This avoids the qubit mismatch issue with QAOA + IBM hardware transpilation
            from qiskit.primitives import StatevectorSampler
            
            print("\n--- HYBRID MODE: Local Optimization + IBM Hardware Sampling ---")
            
            # 1. Optimize parameters using fast local simulation
            print("1. Optimizing parameters locally (fast)...")
            local_sampler = StatevectorSampler()
            optimizer = COBYLA(maxiter=self.steps)  # COBYLA works well for simulation
            
            qaoa = QAOA(sampler=local_sampler, optimizer=optimizer, reps=self.n_layers,
                        initial_state=self.initial_state, mixer=self.mixer_op)
            
            local_result = qaoa.compute_minimum_eigenvalue(self.cost_op)
            optimal_params = local_result.optimal_point
            print(f"   Local optimization done. Eigenvalue: {local_result.eigenvalue}")
            
            # 2. Connect to IBM Quantum for final sampling
            print("2. Connecting to IBM Quantum...")
            service = QiskitRuntimeService()
            
            if self.backend_name:
                print(f"   Using specified backend: {self.backend_name}")
                backend = service.backend(self.backend_name)
            elif self.use_least_busy:
                print("   Selecting least busy operational backend...")
                backend = service.least_busy(simulator=False, operational=True)
                print(f"   Selected backend: {backend.name}")
            else:
                raise ValueError("Either backend_name must be specified or use_least_busy must be True")
            
            # 3. Build and transpile the final circuit with optimal parameters
            print("3. Transpiling circuit for hardware...")
            ansatz = QAOAAnsatz(self.cost_op, reps=self.n_layers,
                               initial_state=self.initial_state, mixer_operator=self.mixer_op)
            
            # Bind optimal parameters to circuit
            bound_circuit = ansatz.assign_parameters(optimal_params)
            
            # Add measurement to all qubits
            bound_circuit.measure_all()
            
            # Transpile for target backend
            pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
            isa_circuit = pm.run(bound_circuit)
            print(f"   Circuit transpiled: {isa_circuit.num_qubits} physical qubits")
            
            # 4. Sample on real hardware
            print("4. Sampling on IBM Quantum hardware...")
            sampler = SamplerV2(mode=backend)
            
            job = sampler.run([isa_circuit])
            print(f"   Job submitted: {job.job_id()}")
            print("   Waiting for results (this may take a while in queue)...")
            
            pub_result = job.result()[0]
            
            # Extract counts - measurement register name varies
            try:
                counts = pub_result.data.meas.get_counts()
            except AttributeError:
                # Try alternate accessor names
                data_keys = list(pub_result.data.keys())
                if data_keys:
                    counts = getattr(pub_result.data, data_keys[0]).get_counts()
                else:
                    print("Warning: Could not extract counts from result")
                    return None, float('inf')
            
            print(f"   Got {len(counts)} unique measurement outcomes")
            return self._process_counts(counts)
                
        else:
            # Local Simulation fallback using qiskit's built-in sampler
            from qiskit.primitives import StatevectorSampler
            
            print("\n--- RUNNING LOCAL SIMULATION ---")
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

    def _process_counts(self, counts):
        """Find most frequent valid bitstring in counts."""
        # Sort counts by frequency
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        
        for bs, count in sorted_counts:
            # Convert back to edges
            rev_bs = bs[::-1]
            edges = []
            for i, bit in enumerate(rev_bs):
                if bit == '1': edges.append(self.edges[i])
            
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
        rev_bs = bs[::-1]
        edges = []
        for i, bit in enumerate(rev_bs):
            if bit == '1': edges.append(self.edges[i])
        return edges, self._calculate_cost(edges)

if __name__ == "__main__":
    # Example 4-city Matrix (Distance between cities)
    graph = [
        [0, 10, 15, 20, 25],
        [10, 0, 35, 25, 30],
        [15, 35, 0, 30, 35],
        [20, 25, 30, 0, 25],
        [25, 30, 35, 25, 0]
    ]

    # --- CONFIGURATION START ---
    USE_IBM = True 
    # Option 1: Set to None to auto-select least busy backend
    # Option 2: Specify a backend name like 'ibm_brisbane', 'ibm_kyoto', etc.
    BACKEND_NAME = None  # Will use least_busy() to find best available backend
    USE_LEAST_BUSY = True  # Only used when BACKEND_NAME is None
    # --- CONFIGURATION END ---

    try:
        print(f"Running solver with USE_IBM={USE_IBM}...")
        
        # Initialize the solver
        # Note: Requires QiskitRuntimeService.save_account() to have been run previously
        # with your IBM Quantum API token
        solver = QAOATSPSolver(
            distance_matrix=graph, 
            num_layers=1, 
            optimization_steps=20, 
            backend_name=BACKEND_NAME,
            use_real_hardware=USE_IBM,
            use_least_busy=USE_LEAST_BUSY
        )
        
        # Run the solver
        best_edges, cost = solver.run()
        
        print("\n" + "="*30)
        print("FINAL RESULTS")
        print("="*30)
        
        if best_edges:
            print(f"Edges selected: {best_edges}")
            print(f"Total Cost: {cost}")
            # Optional: format path for readability
            print("Tour found successfully!")
        else:
            print("No valid tour found.")
            print(f"Cost: {cost}")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\nError encountered: {e}")

