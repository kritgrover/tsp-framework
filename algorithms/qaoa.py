import pennylane as qml
from pennylane import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time

class QAOATSPSolver:
    """
    Quantum Approximate Optimization Algorithm solver for Traveling Salesman Problem.
    Based on Ruan et al. (2020) with enhanced 2-opt constraint-preserving mixer.
    """

    def __init__(self, distance_matrix, num_qaoa_layers=2, learning_rate=0.01, optimization_steps=200):
        self.distance_matrix = distance_matrix
        self.num_cities = len(distance_matrix)
        self.num_layers = num_qaoa_layers
        self.learning_rate = learning_rate
        self.steps = optimization_steps
        
        # For N cities, we have N*(N-1)/2 edges in a complete graph
        self.num_edges = self.num_cities * (self.num_cities - 1) // 2
        self.wires = range(self.num_edges)
        
        self._setup_graph()
        self.dev = qml.device('default.qubit', wires=self.num_edges)
        
        # Valid bitstrings (Hamiltonian cycles)
        self.valid_bitstrings = self._find_hamiltonian_cycles()
        
        # Hamiltonians
        self.cost_h = self._build_cost_hamiltonian()
        self.mixer_h = self._build_mixer_hamiltonian()
        
        self.optimal_params = None

    def _setup_graph(self):
        """Map graph edges to qubits."""
        self.edge_to_qubit = {}
        self.qubit_to_edge = {}
        self.edge_weights = {}
        
        count = 0
        for i in range(self.num_cities):
            for j in range(i + 1, self.num_cities):
                edge = tuple(sorted((i, j)))
                self.edge_to_qubit[edge] = count
                self.qubit_to_edge[count] = edge
                self.edge_weights[edge] = self.distance_matrix[i][j]
                count += 1
                
        # Create NetworkX graph for visualization/helpers
        self.G = nx.Graph()
        for i in range(self.num_cities):
            self.G.add_node(i)
        for e, w in self.edge_weights.items():
            self.G.add_edge(e[0], e[1], weight=w)

    def _find_hamiltonian_cycles(self):
        """Find all valid tours to define the feasible subspace."""
        # Finding all cycles is NP-hard, but for small N it's fine.
        directed_G = self.G.to_directed()
        simple_cycles = list(nx.simple_cycles(directed_G))
        valid_cycles = [c for c in simple_cycles if len(c) == self.num_cities]
        
        bitstrings = set()
        for cycle in valid_cycles:
            # Convert cycle path to edge bitstring
            bs = ['0'] * self.num_edges
            for i in range(self.num_cities):
                u, v = cycle[i], cycle[(i + 1) % self.num_cities]
                edge = tuple(sorted((u, v)))
                q_idx = self.edge_to_qubit[edge]
                bs[q_idx] = '1'
            bitstrings.add("".join(bs))
            
        return list(bitstrings)

    def _build_cost_hamiltonian(self):
        """
        Cost Hamiltonian H_C = sum(-0.5 * w_ij * Z_ij).
        Minimizing this minimizes the total path weight.
        """
        coeffs = []
        obs = []
        for q in range(self.num_edges):
            edge = self.qubit_to_edge[q]
            w = self.edge_weights[edge]
            coeffs.append(-0.5 * w)
            obs.append(qml.PauliZ(q))
            
        return qml.Hamiltonian(coeffs, obs)

    def _build_mixer_hamiltonian(self):
        """
        2-opt constraint-preserving mixer.
        Transitions between valid tours by swapping two edges.
        """
        coeffs = []
        obs = []
        
        edges = list(self.edge_to_qubit.keys())
        n_e = len(edges)
        
        for i in range(n_e):
            for j in range(i+1, n_e):
                e1 = edges[i]
                e2 = edges[j]
                
                # Edges must be disjoint for a valid 2-opt
                if set(e1).isdisjoint(set(e2)):
                    u, v = e1
                    x, y = e2
                    
                    # Two possible reconnections
                    reconnections = [
                        (tuple(sorted((u,x))), tuple(sorted((v,y)))),
                        (tuple(sorted((u,y))), tuple(sorted((v,x))))
                    ]
                    
                    for ne1, ne2 in reconnections:
                        q_old1 = self.edge_to_qubit[e1]
                        q_old2 = self.edge_to_qubit[e2]
                        q_new1 = self.edge_to_qubit[ne1]
                        q_new2 = self.edge_to_qubit[ne2]
                        
                        # Add term 1/8 * (ReRe terms + ImIm terms)
                        c_val = 0.125
                        
                        # 8 Pauli strings per swap operation
                        term_coeffs = [c_val, -c_val, -c_val, c_val, c_val, c_val, c_val, c_val]
                        
                        term_obs = [
                            qml.PauliX(q_new1) @ qml.PauliX(q_new2) @ qml.PauliX(q_old1) @ qml.PauliX(q_old2),
                            qml.PauliX(q_new1) @ qml.PauliX(q_new2) @ qml.PauliY(q_old1) @ qml.PauliY(q_old2),
                            qml.PauliY(q_new1) @ qml.PauliY(q_new2) @ qml.PauliX(q_old1) @ qml.PauliX(q_old2),
                            qml.PauliY(q_new1) @ qml.PauliY(q_new2) @ qml.PauliY(q_old1) @ qml.PauliY(q_old2),
                            qml.PauliX(q_new1) @ qml.PauliY(q_new2) @ qml.PauliX(q_old1) @ qml.PauliY(q_old2),
                            qml.PauliX(q_new1) @ qml.PauliY(q_new2) @ qml.PauliY(q_old1) @ qml.PauliX(q_old2),
                            qml.PauliY(q_new1) @ qml.PauliX(q_new2) @ qml.PauliX(q_old1) @ qml.PauliY(q_old2),
                            qml.PauliY(q_new1) @ qml.PauliX(q_new2) @ qml.PauliY(q_old1) @ qml.PauliX(q_old2),
                        ]
                        
                        coeffs.extend(term_coeffs)
                        obs.extend(term_obs)

        if not obs:
            return qml.Hamiltonian([0], [qml.Identity(0)])
            
        return qml.Hamiltonian(coeffs, obs)

    def _circuit(self, params):
        """QAOA Circuit Definition"""
        
        # 1. Initial State: Equal Superposition of Valid Tours
        dim = 2**self.num_edges
        state = np.zeros(dim, dtype=np.complex128)
        
        valid_indices = [int(bs, 2) for bs in self.valid_bitstrings]
        amp = 1.0 / np.sqrt(len(valid_indices))
        
        for idx in valid_indices:
            state[idx] = amp
            
        qml.StatePrep(state, wires=self.wires)
        
        # 2. QAOA Layers
        for i in range(self.num_layers):
            gamma = params[i, 0]
            beta = params[i, 1]
            qml.evolve(self.cost_h, gamma)
            qml.evolve(self.mixer_h, beta)

    def solve(self):
        """Run the full optimization loop."""
        print(f"QAOA Solver started for {self.num_cities} cities.")
        print(f"Search space size: {len(self.valid_bitstrings)} valid tours.")
        
        @qml.qnode(self.dev)
        def cost_fn(params):
            self._circuit(params)
            return qml.expval(self.cost_h)

        # Initialize parameters
        params = np.random.uniform(0, np.pi/2, size=(self.num_layers, 2), requires_grad=True)
        opt = qml.AdamOptimizer(stepsize=self.learning_rate)
        
        print(f"Optimizing for {self.steps} steps...")
        t0 = time.time()
        
        for i in range(self.steps):
            params, cost = opt.step_and_cost(cost_fn, params)
            if (i+1) % 50 == 0:
                print(f"  Step {i+1}: Cost = {cost:.4f}")
                
        self.optimal_params = params
        t1 = time.time()
        print(f"Optimization finished in {t1-t0:.2f}s.")
        
        # Get final result
        return self._get_result()

    def _get_result(self):
        """Extract best bitstring from final state."""
        @qml.qnode(self.dev)
        def probs_fn(params):
            self._circuit(params)
            return qml.probs(wires=self.wires)
            
        probs = probs_fn(self.optimal_params)
        
        best_prob = -1
        best_bs = None
        
        # Scan only valid subspace
        for bs in self.valid_bitstrings:
            idx = int(bs, 2)
            p = probs[idx]
            if p > best_prob:
                best_prob = p
                best_bs = bs
                
        return best_bs, best_prob

    def decode_solution(self, bitstring):
        """Convert bitstring to readable path."""
        if not bitstring: return None, float('inf')
        
        selected_edges = []
        for i, bit in enumerate(bitstring):
            if bit == '1':
                selected_edges.append(self.qubit_to_edge[i])
                
        # Reconstruct path
        adj = {i: [] for i in range(self.num_cities)}
        total_dist = 0
        for u, v in selected_edges:
            adj[u].append(v)
            adj[v].append(u)
            total_dist += self.edge_weights[(u,v)]
            
        # Walk path
        curr = 0
        path = [0]
        visited = {0}
        
        while len(path) < self.num_cities:
            found = False
            for nbr in adj[curr]:
                if nbr not in visited:
                    visited.add(nbr)
                    path.append(nbr)
                    curr = nbr
                    found = True
                    break
            if not found: break
            
        return path, total_dist

if __name__ == "__main__":
    # Test on 4 cities
    dist = [
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ]
    
    solver = QAOATSPSolver(dist, num_qaoa_layers=2, optimization_steps=100)
    bs, prob = solver.solve()
    path, cost = solver.decode_solution(bs)
    
    print("\n" + "="*40)
    print("QAOA RESULTS (Finalized)")
    print("="*40)
    print(f"Best Bitstring: {bs}")
    print(f"Probability:    {prob:.4f}")
    print(f"Path:           {' -> '.join(map(str, path))}")
    print(f"Cost:           {cost}")
    print("="*40)

