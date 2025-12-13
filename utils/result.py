"""
Standardized result class for TSP solutions.
"""
import math
from typing import List, Dict, Optional


class TSPSolution:
    """
    Standardized result container for TSP solutions.
    
    Attributes:
        path: List of node indices representing the tour
        cost: Total cost of the tour
        algorithm: Name of the algorithm used
        metadata: Dictionary with additional information (time, iterations, etc.)
    """
    
    def __init__(self, path: List[int], cost: float, algorithm: str, 
                 metadata: Optional[Dict] = None):
        """
        Initialize a TSP solution.
        
        Args:
            path: List of node indices in tour order (should form a cycle)
            cost: Total cost/distance of the tour
            algorithm: Name of the algorithm that produced this solution
            metadata: Optional dictionary with additional info (time, iterations, etc.)
        """
        self.path = path
        self.cost = cost
        self.algorithm = algorithm
        self.metadata = metadata if metadata is not None else {}
    
    def __repr__(self):
        """String representation of the solution."""
        path_str = " -> ".join(map(str, self.path))
        if len(self.path) > 0:
            path_str += f" -> {self.path[0]}"
        return (f"TSPSolution(algorithm={self.algorithm}, cost={self.cost:.2f}, "
                f"path={path_str})")
    
    def __str__(self):
        """Human-readable string representation."""
        path_str = " -> ".join(map(str, self.path))
        if len(self.path) > 0:
            path_str += f" -> {self.path[0]}"
        
        result = f"\n{'='*50}\n"
        result += f"TSP SOLUTION ({self.algorithm.upper()})\n"
        result += f"{'='*50}\n"
        result += f"Cost: {self.cost:.4f}\n"
        result += f"Path: {path_str}\n"
        
        if self.metadata:
            result += "\nMetadata:\n"
            for key, value in self.metadata.items():
                result += f"  {key}: {value}\n"
        
        result += f"{'='*50}\n"
        return result
    
    def is_valid(self) -> bool:
        """
        Check if the solution is valid (has a path and finite cost).
        
        Returns:
            True if solution is valid, False otherwise
        """
        return (self.path is not None and 
                len(self.path) > 0 and 
                self.cost != float('inf') and 
                not math.isnan(self.cost))

