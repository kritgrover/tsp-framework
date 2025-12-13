import math
import random
import matplotlib.pyplot as plt
import networkx as nx
import time
from utils.shared import calculate_path_cost


def generate_initial_path(graph):
    """
    Generate a valid initial random path.
    Tries 100 times to find a valid random permutation.
    """
    n = len(graph)
    cities = list(range(1, n)) # Start at 0 fixed
    
    for _ in range(100):
        random.shuffle(cities)
        path = [0] + cities
        if calculate_path_cost(path, graph) != float('inf'):
            return path
            
    return [0] + list(range(1, n))

def simulated_annealing_final(graph, 
                            initial_temp=1000, 
                            cooling_rate=0.003, 
                            max_iter=10000,
                            visualize=True):
    """
    Optimized Simulated Annealing Solver for TSP.
    """
    t_start = time.time()
    n = len(graph)
    
    # 1. Initialization
    current_path = generate_initial_path(graph)
    current_cost = calculate_path_cost(current_path, graph)
    
    best_path = list(current_path)
    best_cost = current_cost
    
    temp = initial_temp
    
    # Visualization Setup
    if visualize:
        plt.ion()
        fig, ax = plt.subplots(figsize=(10, 8))
        G = nx.Graph()
        for i in range(n):
            G.add_node(i)
            for j in range(i+1, n):
                if graph[i][j] > 0:
                    G.add_edge(i, j, weight=graph[i][j])
        pos = nx.spring_layout(G, seed=1)
        
        # Update ~50 times total
        update_freq = max(1, max_iter // 50)
    
    # 2. Annealing Loop
    print(f"Starting Simulated Annealing (Max Iter: {max_iter})...")
    
    no_improv_count = 0
    
    for i in range(max_iter):
        if temp < 0.1: break
        
        # 3. Neighbor Generation (Swap 2 Cities)
        idx1, idx2 = random.sample(range(1, n), 2)
        
        # Create new path by swapping
        new_path = list(current_path)
        new_path[idx1], new_path[idx2] = new_path[idx2], new_path[idx1]
        
        new_cost = calculate_path_cost(new_path, graph)
        
        if new_cost == float('inf'):
            continue # Skip invalid paths
            
        # 4. Acceptance Criteria (Metropolis)
        delta = new_cost - current_cost
        
        if delta < 0 or math.exp(-delta / temp) > random.random():
            # Accept
            current_path = new_path
            current_cost = new_cost
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_path = list(current_path)
                no_improv_count = 0
            else:
                no_improv_count += 1
        else:
            no_improv_count += 1
            
        # 5. Cooling
        temp *= (1 - cooling_rate)
        
        # Early exit if stuck
        if no_improv_count > 2000:
            print("  Stopping early: No improvement for 2000 iterations.")
            break
            
        # Visualization
        if visualize and i % update_freq == 0:
            ax.clear()
            nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightblue', node_size=500)
            nx.draw_networkx_labels(G, pos, ax=ax)
            nx.draw_networkx_edges(G, pos, ax=ax, edge_color='lightgray', alpha=0.5)
            
            # Draw current best path
            path_edges = [(current_path[j], current_path[j+1]) for j in range(n-1)]
            path_edges.append((current_path[-1], current_path[0]))
            
            nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='orange', width=2, ax=ax)
            
            ax.set_title(f"Iter {i} | Temp {temp:.1f} | Cost {current_cost}", fontsize=12)
            plt.pause(0.001)

    t_end = time.time()
    path_str = " -> ".join(map(str, best_path)) + f" -> {best_path[0]}"
    
    print("\n" + "="*40)
    print("SIMULATED ANNEALING RESULTS (Finalized)")
    print("="*40)
    print(f"Iterations:   {i+1}")
    print(f"Final Temp:   {temp:.4f}")
    print(f"Best Cost:    {best_cost}")
    print(f"Best Path:    {path_str}")
    print(f"Time Taken:   {t_end - t_start:.4f}s")
    print("="*40)
    
    if visualize:
        ax.clear()
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightgreen', node_size=600)
        nx.draw_networkx_labels(G, pos, ax=ax)
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='lightgray', alpha=0.5)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'weight'))
        
        final_edges = [(best_path[j], best_path[j+1]) for j in range(n-1)]
        final_edges.append((best_path[-1], best_path[0]))
        
        nx.draw_networkx_edges(G, pos, edgelist=final_edges, edge_color='green', width=3, ax=ax)
        ax.set_title(f"SA OPTIMAL FOUND | Cost: {best_cost}", fontsize=14, weight='bold')
        plt.ioff()
        plt.show()
        
    return best_cost, path_str

if __name__ == "__main__":
    # Test
    sample = [
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ]
    simulated_annealing_final(sample, visualize=True)
