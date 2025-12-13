"""
Unified visualization module for TSP solutions.
"""
import matplotlib.pyplot as plt
import networkx as nx
from typing import List, Optional


def visualize_solution(graph: List[List[float]], path: List[int], cost: float,
                      algorithm_name: str, title: Optional[str] = None,
                      show_edge_labels: bool = True):
    """
    Visualize a TSP solution on a graph.
    
    Args:
        graph: Distance matrix (2D list)
        path: List of node indices representing the tour
        cost: Total cost of the tour
        algorithm_name: Name of the algorithm used
        title: Optional custom title for the plot
        show_edge_labels: Whether to show edge weight labels
    """
    n = len(graph)
    
    # Create NetworkX graph
    G = nx.Graph()
    for i in range(n):
        G.add_node(i)
        for j in range(i + 1, n):
            if graph[i][j] > 0:
                G.add_edge(i, j, weight=graph[i][j])
    
    pos = nx.spring_layout(G, seed=1, k=1, iterations=50)
    plt.figure(figsize=(12, 8))
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                          node_size=600, alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
    # Draw all edges (background, light gray)
    nx.draw_networkx_edges(G, pos, edge_color='lightgray', 
                          alpha=0.3, width=1)
    
    if show_edge_labels:
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)
    
    # Draw TSP tour path (highlighted)
    if path and len(path) > 0:
        tour_edges = []
        for i in range(len(path) - 1):
            tour_edges.append((path[i], path[i + 1]))
        if len(path) > 1:
            tour_edges.append((path[-1], path[0]))
        
        nx.draw_networkx_edges(G, pos, edgelist=tour_edges, 
                              edge_color='green', width=3, 
                              alpha=0.8, style='solid')
    
    # Set title
    if title is None:
        title = f"{algorithm_name.upper()} Solution | Cost: {cost:.2f}"
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def visualize_graph_only(graph: List[List[float]], title: Optional[str] = None):
    """
    Visualize just the graph without a solution path.
    """
    n = len(graph)
    
    # Create NetworkX graph
    G = nx.Graph()
    for i in range(n):
        G.add_node(i)
        for j in range(i + 1, n):
            if graph[i][j] > 0:
                G.add_edge(i, j, weight=graph[i][j])
    
    pos = nx.spring_layout(G, seed=1, k=1, iterations=50)
    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                          node_size=600, alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color='gray', 
                          alpha=0.6, width=2)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)
    
    # Set title
    if title is None:
        title = f"Graph ({n} nodes)"
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()

