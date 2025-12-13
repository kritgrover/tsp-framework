"""
Graph generation utilities for TSP framework.
Generates different types of graphs as distance matrices.
"""
import random
import math


def generate_fully_connected(n, weight_range=(1, 20), seed=None):
    """
    Generate a fully connected graph (complete graph) with random weights.
    """
    if seed is not None:
        random.seed(seed)
    
    matrix = [[0] * n for _ in range(n)]
    min_weight, max_weight = weight_range
    
    for i in range(n):
        for j in range(i + 1, n):
            weight = random.randint(min_weight, max_weight)
            matrix[i][j] = weight
            matrix[j][i] = weight
    
    return matrix


def generate_partially_connected(n, connectivity=0.6, weight_range=(1, 100), seed=None):
    """
    Generate a partially connected graph with random edges.
    """
    if seed is not None:
        random.seed(seed)
    
    matrix = [[0] * n for _ in range(n)]
    min_weight, max_weight = weight_range
    
    # Ensure graph is connected by creating a spanning tree first
    # Start with node 0 and connect all nodes
    for i in range(1, n):
        parent = random.randint(0, i - 1)
        weight = random.randint(min_weight, max_weight)
        matrix[parent][i] = weight
        matrix[i][parent] = weight
    
    # Add additional edges based on connectivity probability
    for i in range(n):
        for j in range(i + 1, n):
            if matrix[i][j] == 0 and random.random() < connectivity:
                weight = random.randint(min_weight, max_weight)
                matrix[i][j] = weight
                matrix[j][i] = weight
    
    return matrix


def generate_circle(n, weight_range=(1, 100), seed=None):
    """
    Generate a circular graph where each node is connected to exactly 2 neighbors.
    """
    if seed is not None:
        random.seed(seed)
    
    matrix = [[0] * n for _ in range(n)]
    min_weight, max_weight = weight_range
    
    # Connect each node to its next neighbor (circular)
    for i in range(n):
        next_node = (i + 1) % n
        weight = random.randint(min_weight, max_weight)
        matrix[i][next_node] = weight
        matrix[next_node][i] = weight
    
    return matrix


def generate_graph(graph_type, **params):
    """
    Unified function to generate graphs based on type.
    
    Args:
        graph_type: One of 'fully_connected', 'partially_connected', 'circle'
        **params: Parameters specific to graph type
    
    Returns:
        2D list representing distance matrix
    """
    if graph_type == 'fully_connected':
        n = params.get('num_nodes', params.get('n', 5))
        weight_range = params.get('weight_range', (1, 100))
        seed = params.get('seed', None)
        return generate_fully_connected(n, weight_range, seed)
    
    elif graph_type == 'partially_connected':
        n = params.get('num_nodes', params.get('n', 5))
        connectivity = params.get('connectivity', 0.6)
        weight_range = params.get('weight_range', (1, 100))
        seed = params.get('seed', None)
        return generate_partially_connected(n, connectivity, weight_range, seed)
    
    elif graph_type == 'circle':
        n = params.get('num_nodes', params.get('n', 5))
        weight_range = params.get('weight_range', (1, 100))
        seed = params.get('seed', None)
        return generate_circle(n, weight_range, seed)
    
    else:
        raise ValueError(f"Unknown graph type: {graph_type}. "
                        f"Must be one of: 'fully_connected', 'partially_connected', 'circle'")

