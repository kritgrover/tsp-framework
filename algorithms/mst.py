import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
import time
from utils.shared import calculate_path_cost

def find_minimum_spanning_tree(graph):
    """
    Prim's algorithm to find the Minimum Spanning Tree of a graph.
    Optimized for dense graphs using adjacency matrix.
    """
    n = len(graph)
    selected = [False] * n
    selected[0] = True
    mst_edges = []
    
    # Pre-calculate valid edges to avoid repeated checks
    min_weights = [float('inf')] * n
    parent = [-1] * n
    
    # Initialize from start node 0
    for v in range(1, n):
        if graph[0][v] > 0:
            min_weights[v] = graph[0][v]
            parent[v] = 0
            
    for _ in range(n - 1):
        # Find closest unselected node
        min_val = float('inf')
        u = -1
        
        for i in range(1, n):
            if not selected[i] and min_weights[i] < min_val:
                min_val = min_weights[i]
                u = i
                
        if u == -1:
            break # Graph disconnected
            
        selected[u] = True
        mst_edges.append((parent[u], u, min_val))
        
        # Update neighbors
        for v in range(n):
            if not selected[v] and graph[u][v] > 0 and graph[u][v] < min_weights[v]:
                min_weights[v] = graph[u][v]
                parent[v] = u
                
    return mst_edges

def build_adjacency_list_from_mst(mst_edges):
    """
    Build adjacency list representation of the MST for DFS traversal.
    """
    adj_list = defaultdict(list)
    temp_adj = defaultdict(list)
    for u, v, weight in mst_edges:
        temp_adj[u].append(v)
        temp_adj[v].append(u)
        
    # Sort neighbors to ensure consistent path order for same inputs
    for node in temp_adj:
        temp_adj[node].sort()
        
    return temp_adj

def dfs_preorder_traversal(adj_list, start_node=0):
    """
    Perform DFS preorder traversal on the MST.
    Iterative implementation to avoid recursion depth limits.
    """
    visited = set()
    path = []
    stack = [start_node]
    
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            path.append(node)
            
            # Add neighbors to stack in reverse order so they are popped in correct order
            neighbors = adj_list[node]
            for neighbor in reversed(neighbors):
                if neighbor not in visited:
                    stack.append(neighbor)
                    
    return path

def tsp_solver_final(graph, visualize=True):
    """
    Optimized MST-based 2-Approximation TSP Solver.
    """
    t_start = time.perf_counter()
    n = len(graph)
    
    # 1. Compute MST (Prim's)
    mst_edges = find_minimum_spanning_tree(graph)
    
    if len(mst_edges) != n - 1:
        print("Error: Graph is not connected.")
        return None, float('inf')
        
    # 2. Build MST Adjacency
    adj_list = build_adjacency_list_from_mst(mst_edges)
    
    # 3. DFS Preorder Walk (The TSP Path)
    path = dfs_preorder_traversal(adj_list)
    
    # 4. Calculate Cost
    total_cost = calculate_path_cost(path, graph)
    
    t_end = time.perf_counter()
    
    
    path_str = " -> ".join(map(str, path)) + f" -> {path[0]}"
    
    print("\n" + "="*40)
    print("MST APPROXIMATION RESULTS (Finalized)")
    print("="*40)
    print(f"MST Edges:    {len(mst_edges)}")
    print(f"TSP Cost:     {total_cost if (total_cost != float('inf')) else 'Inf'}")
    print(f"Path:         {path_str}")
    print(f"Time Taken:   {t_end - t_start:.4f}s")
    print("="*40)
    
    if visualize:
        G = nx.Graph()
        # Add all edges for background
        for i in range(n):
            for j in range(i+1, n):
                if graph[i][j] > 0:
                    G.add_edge(i, j, weight=graph[i][j])
        
        pos = nx.spring_layout(G, seed=1)
        
        plt.figure(figsize=(10, 8))
        # Draw background
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500)
        nx.draw_networkx_labels(G, pos)
        nx.draw_networkx_edges(G, pos, edge_color='lightgray', alpha=0.5)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'weight'))
        
        # Draw MST edges (dashed orange)
        mst_pairs = [(u, v) for u, v, w in mst_edges]
        nx.draw_networkx_edges(G, pos, edgelist=mst_pairs, edge_color='orange', width=2, style='dashed', alpha=0.6)
        
        # Draw TSP Path (solid green)
        tsp_pairs = [(path[i], path[i+1]) for i in range(n-1)]
        tsp_pairs.append((path[-1], path[0]))
        nx.draw_networkx_edges(G, pos, edgelist=tsp_pairs, edge_color='green', width=2)
        
        plt.title(f"MST Approximation | Cost: {total_cost}", fontsize=14, weight='bold')
        # Legend hack
        plt.plot([], [], color='orange', linestyle='dashed', label='MST Structure')
        plt.plot([], [], color='green', linewidth=2, label='TSP Tour')
        plt.legend()
        
        plt.show()
        
    return path, total_cost

if __name__ == "__main__":
    # Test
    sample = [
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ]
    tsp_solver_final(sample, visualize=True)
