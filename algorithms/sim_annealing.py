import math
import random
import matplotlib.pyplot as plt
import networkx as nx
import time
from utils.shared import calculate_path_cost


def build_adjacency_lists(graph):
    """
    Convert adjacency matrix to adjacency lists during initialization.
    This pre-computation creates O(1) neighbor access.
    """
    n = len(graph)
    adj_lists = [[] for _ in range(n)]
    
    # Single pass through matrix to build all adjacency lists
    for i in range(n):
        for j in range(n):
            if i != j and graph[i][j] > 0:
                adj_lists[i].append(j)
    
    return adj_lists

def get_nth_neighbors_optimized(city, adj_lists, n_hops, visited_set):
    """
    Efficient nth-neighbor discovery using BFS with adjacency lists.
    """
    if n_hops == 0:
        return [city] if city not in visited_set else []
    if n_hops == 1:
        return [c for c in adj_lists[city] if c not in visited_set]
    
    current_level = {city}
    
    for hop in range(n_hops):
        if not current_level:
            return []
        
        next_level = set() 
        for current_city in current_level:
            for neighbor in adj_lists[current_city]:
                if neighbor not in visited_set:
                    next_level.add(neighbor)
        
        current_level = next_level
        if hop < n_hops - 1:
            visited_set.update(current_level)
    
    return list(current_level)

def is_valid_path_fast(path, graph):
    """
    Fast path validation with early termination.
    """
    n = len(path)
    if path[0] != 0:
        return False
    
    for i in range(n - 1):
        if graph[path[i]][path[i + 1]] <= 0:
            return False
    
    return graph[path[-1]][path[0]] > 0

def generate_valid_initial_path_robust(graph, adj_lists):
    """
    Robust initial path generation with multiple fallback strategies.
    """
    n = len(graph)
    max_attempts = 100
    
    for attempt in range(max_attempts):
        path = [0]
        unvisited = set(range(1, n))
        current = 0
        
        # STRATEGY 1: Greedy nearest neighbor
        while unvisited:
            best_next = None
            best_distance = float('inf')

            for neighbor in adj_lists[current]:
                if neighbor in unvisited:
                    w = graph[current][neighbor]
                    if w < best_distance:
                        best_distance = w
                        best_next = neighbor
            
            if best_next is None:
                # STRATEGY 2 (attempts 0-29): Random valid connection
                if attempt < 30:
                    candidates = list(unvisited)
                    random.shuffle(candidates)
                    for candidate in candidates:
                        if graph[current][candidate] > 0:
                            best_next = candidate
                            break
                
                # STRATEGY 3 (attempts 30-59): Backtracking
                if best_next is None and attempt < 60:
                    if len(path) > 1:
                        last_city = path.pop()
                        unvisited.add(last_city)
                        current = path[-1]
                        continue
                
                # STRATEGY 4 (attempts 60+): Complete restart
                if best_next is None:
                    break
            
            if best_next is not None:
                path.append(best_next)
                unvisited.remove(best_next)
                current = best_next
        
        # Check if we found a complete path and it's valid
        if len(path) == n and is_valid_path_fast(path, graph):
            return path
        
        if attempt >= 60:
            # Fallback to random sampling if intelligent construction fails
            candidates = list(range(1, n))
            if len(candidates) >= n-1: # Safety check
                path = [0] + random.sample(candidates, n-1)
                if is_valid_path_fast(path, graph):
                    return path
    
    # If all attempts failed
    return [0] + list(range(1, n))

def generate_neighbor_path_optimized(current_path, graph, adj_lists, neighbor_distance, path_set, temp_visited):
    """
    Advanced neighbor generation with memory reuse and targeted swaps.
    Returns tuple (idx1, idx2) if a swap was made, or None if failed.
    """
    n = len(current_path)
    
    # Reuse temp_visited set instead of creating new ones
    temp_visited.clear()
    temp_visited.update(current_path)
    
    for _ in range(20):
        city_idx = random.randint(1, n - 1)
        city = current_path[city_idx]
        
        # Get nth neighbors efficiently
        temp_visited_copy = temp_visited.copy()
        if city in temp_visited_copy:
            temp_visited_copy.remove(city)
        
        nth_neighbors = get_nth_neighbors_optimized(city, adj_lists, neighbor_distance, temp_visited_copy)
        
        # Filter for cities in current path (excluding city 0)
        valid_candidates = [c for c in nth_neighbors if c in path_set and c != 0]
        
        if not valid_candidates:
            continue
        
        swap_candidate = random.choice(valid_candidates)
        try:
            swap_idx = current_path.index(swap_candidate)
        except ValueError:
            continue
            
        current_path[city_idx], current_path[swap_idx] = current_path[swap_idx], current_path[city_idx]
        
        # Quick validation
        if is_valid_path_fast(current_path, graph):
            return (city_idx, swap_idx)
                
        # Revert swap if invalid
        current_path[city_idx], current_path[swap_idx] = current_path[swap_idx], current_path[city_idx]
    
    # Fallback: simple adjacent swap
    for _ in range(10):
        if n <= 2:
            break
        i = random.randint(1, n - 2)
        
        current_path[i], current_path[i + 1] = current_path[i + 1], current_path[i]
        
        if is_valid_path_fast(current_path, graph):
            return (i, i + 1)
        
        # Revert
        current_path[i], current_path[i + 1] = current_path[i + 1], current_path[i]
    
    return None

def simulated_annealing_final(graph, 
                            initial_temp=1000, 
                            cooling_rate=0.003, 
                            max_iter=10000,
                            visualize=True):
    """
    Optimized Simulated Annealing Solver for TSP.
    Matches the interface required by the framework.
    """
    t_start = time.time()
    n = len(graph)
    
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
        
        # Dynamic update frequency
        update_freq = max(1, max_iter // 50)
    else:
        G, pos, ax = None, None, None
        update_freq = 0

    # 1. Pre-compute adjacency lists
    adj_lists = build_adjacency_lists(graph)
    
    # 2. Initialization
    current_path = generate_valid_initial_path_robust(graph, adj_lists)
    current_cost = calculate_path_cost(current_path, graph)
    
    best_path = list(current_path)
    best_cost = current_cost
    
    # Pre-allocate reusable data structures
    path_set = set(current_path)
    temp_visited = set()
    neighbor_distance = 2
    
    temp = initial_temp
    no_improv_count = 0
    no_improv_limit = 2000 # Default limit
    
    print(f"Starting Optimized Simulated Annealing (Max Iter: {max_iter})...")
    
    # 3. Annealing Loop
    for i in range(max_iter):
        if temp < 0.1: break
        
        # Generate neighbor in-place (returns indices if swap made and valid)
        swap_indices = generate_neighbor_path_optimized(current_path, graph, adj_lists, 
                                                   neighbor_distance, path_set, temp_visited)
        
        if swap_indices:
            idx1, idx2 = swap_indices
            new_cost = calculate_path_cost(current_path, graph)
            
            # 4. Acceptance Criteria (Metropolis)
            delta = new_cost - current_cost
            
            if delta < 0 or math.exp(-delta / temp) > random.random():
                current_cost = new_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_path = list(current_path)
                    no_improv_count = 0
                else:
                    no_improv_count += 1
            else:
                # Revert if rejected
                current_path[idx1], current_path[idx2] = current_path[idx2], current_path[idx1]
                no_improv_count += 1
        else:
            no_improv_count += 1

        # Early exit if stuck
        if no_improv_count > no_improv_limit:
            print(f"  Stopping early: No improvement for {no_improv_limit} iterations.")
            break
            
        # Cooling
        temp *= (1 - cooling_rate)
        
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
        
        final_edges = [(best_path[j], best_path[j+1]) for j in range(n-1)]
        final_edges.append((best_path[-1], best_path[0]))
        
        nx.draw_networkx_edges(G, pos, edgelist=final_edges, edge_color='green', width=3, ax=ax)
        ax.set_title(f"SA OPTIMAL FOUND | Cost: {best_cost}", fontsize=14, weight='bold')
        plt.ioff()
        plt.show()
        
    return best_cost, path_str

if __name__ == "__main__":
    sample = [
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ]
    simulated_annealing_final(sample, visualize=True)
