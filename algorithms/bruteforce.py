import itertools
import matplotlib.pyplot as plt
import networkx as nx
import math
import time
from utils.shared import calculate_path_cost

def precompute_graph_data(matrix):
    """
    Pre-compute all graph-related data once for visualization.
    This is done to avoid recalculating the graph data every iteration.
    """
    G = nx.Graph()
    n = len(matrix)
    edge_weights = {}
    
    # Create the graph
    for i in range(n):
        G.add_node(i)
        for j in range(i + 1, n):
            weight = matrix[i][j]
            if weight > 0:
                G.add_edge(i, j, weight=weight)
                edge_weights[(i, j)] = weight
                edge_weights[(j, i)] = weight

    pos = nx.spring_layout(G, seed=1)
    return G, pos, edge_weights

def tsp_solver_final(matrix, visualize=True, update_frequency=None):
    """
    Highly optimized Brute Force TSP solver.
    """
    n = len(matrix)
    
    # Visualization Setup
    if visualize:
        G, pos, edge_weights = precompute_graph_data(matrix)
        plt.ion()
        fig, ax = plt.subplots(figsize=(10, 8))
        if update_frequency is None:
            total_perms = math.factorial(n - 1)
            update_frequency = max(1, total_perms // 100) # Update ~100 times total
    else:
        G, pos, edge_weights, fig, ax = None, None, None, None, None

    # Algorithm Initialization
    start_city = 0
    other_cities = list(range(1, n))
    
    best_cost = float('inf')
    best_path = None
    valid_count = 0
    
    # Path buffer pre-allocation (avoiding list creation in loop)
    # Format: [start, c1, c2, ..., cn-1, start]
    current_path = [0] * (n + 1)
    current_path[0] = start_city
    current_path[-1] = start_city
    
    total_perms = math.factorial(n - 1)
    print(f"Starting optimized brute force search on {n} cities.")
    print(f"Checking {total_perms:,} permutations...")
    
    t_start = time.time()
    
    # Main Loop
    for idx, perm in enumerate(itertools.permutations(other_cities), 1):
        
        # Fill path buffer
        for i, city in enumerate(perm):
            current_path[i+1] = city
            
        # Calculate Cost
        cost = calculate_path_cost(current_path[:-1], matrix)
        
        if cost != float('inf'):
            valid_count += 1
            if cost < best_cost:
                best_cost = cost
                best_path = list(current_path) # Copy only when new best found

        # Visualization Update
        if visualize and idx % update_frequency == 0:
            ax.clear()
            nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightblue', node_size=500)
            nx.draw_networkx_labels(G, pos, ax=ax)
            nx.draw_networkx_edges(G, pos, ax=ax, edge_color='lightgray')
            
            # Draw current path
            path_edges = [(current_path[i], current_path[i+1]) for i in range(n)]
            color = 'orange' if valid else 'red'
            nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color=color, width=2, ax=ax)
            
            ax.set_title(f"Iter {idx}/{total_perms} | Cost: {cost if valid else 'Inf'}", fontsize=12)
            plt.pause(0.001)

    t_end = time.time()
    
    # Final Result Display
    if best_path:
        path_str = ' -> '.join(map(str, best_path))
        print("\n" + "="*40)
        print("BRUTE FORCE RESULTS (Finalized)")
        print("="*40)
        print(f"Optimal Cost: {best_cost}")
        print(f"Optimal Path: {path_str}")
        print(f"Time Taken:   {t_end - t_start:.4f}s")
        print(f"Valid Tours:  {valid_count}")
        print("="*40)
        
        if visualize:
            ax.clear()
            nx.draw(G, pos, ax=ax, with_labels=True, node_color='lightgreen')
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_weights, ax=ax)
            final_edges = [(best_path[i], best_path[i+1]) for i in range(n)]
            nx.draw_networkx_edges(G, pos, edgelist=final_edges, edge_color='green', width=3, ax=ax)
            ax.set_title(f"OPTIMAL SOLUTION | Cost: {best_cost}", fontsize=14, weight='bold')
            plt.ioff()
            plt.show()
            
        return best_cost, path_str, valid_count
    else:
        print("\nNo valid tour found.")
        return float('inf'), None, 0

if __name__ == "__main__":
    # Self-test if run directly
    sample_matrix = [
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ]
    tsp_solver_final(sample_matrix, visualize=True)

