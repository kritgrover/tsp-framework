def calculate_path_cost(path, graph):
    """
    Calculates the total cost of a path given a distance matrix.
    """
    total_dist = 0
    n = len(path)
    
    for i in range(n - 1):
        w = graph[path[i]][path[i + 1]]
        if w <= 0: return float('inf')
        total_dist += w
        
    w_return = graph[path[-1]][path[0]]
    if w_return <= 0: return float('inf')

    return total_dist + w_return