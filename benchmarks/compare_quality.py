import sys
import os
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path to allow importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.graph_generators import generate_graph
from utils.algorithm_adapter import run_bruteforce, run_mst, run_sim_annealing, run_qaoa

def run_benchmark():
    # Define problem sizes (number of cities): 3 to 10
    sizes = range(3, 11)
    
    algorithms = {
        'Bruteforce': run_bruteforce,
        'MST': run_mst,
        'Simulated Annealing': run_sim_annealing,
        'QAOA': run_qaoa
    }
    
    # Store approximation ratios instead of costs
    ratios = {algo: [] for algo in algorithms}
    
    print("Running Quality Benchmark...")
    
    for n in sizes:
        print(f"Processing size {n}...")
        # Generate a fully connected graph (complete graph) for all algorithms to solve
        # This ensures every pair of cities is connected, guaranteeing valid TSP solutions
        graph = generate_graph('fully_connected', num_nodes=n, weight_range=(1, 100), seed=42)
        
        # Verify graph is fully connected (all non-diagonal entries should be > 0)
        is_fully_connected = True
        for i in range(n):
            for j in range(n):
                if i != j and graph[i][j] <= 0:
                    is_fully_connected = False
                    break
            if not is_fully_connected:
                break
        
        if not is_fully_connected:
            print(f"  Warning: Graph for size {n} is not fully connected! Regenerating...")
            # Regenerate with explicit parameters
            graph = generate_graph('fully_connected', num_nodes=n, weight_range=(1, 100), seed=42+n)
        
        # First, run bruteforce to get baseline cost
        bruteforce_cost = None
        try:
            print(f"  Running Bruteforce (baseline)...")
            bruteforce_solution = run_bruteforce(graph)
            bruteforce_cost = bruteforce_solution.cost
            
            # Validate bruteforce solution
            if bruteforce_cost is None or bruteforce_cost == float('inf') or bruteforce_cost <= 0:
                print(f"  Warning: Bruteforce could not find a valid solution (cost: {bruteforce_cost})")
                print(f"  Skipping size {n} - if bruteforce can't solve it, others likely can't either")
                # If bruteforce fails, skip this size for all algorithms
                for name in algorithms:
                    ratios[name].append(None)
                continue
            
            print(f"  Baseline cost: {bruteforce_cost}")
        except Exception as e:
            print(f"  Error running Bruteforce: {e}")
            # If bruteforce fails, skip this size
            for name in algorithms:
                ratios[name].append(None)
            continue
        
        # Now run other algorithms and calculate approximation ratios
        for name, func in algorithms.items():
            # Apply limits for computationally expensive algorithms
            if name == 'QAOA' and n > 5:
                ratios[name].append(None)
                continue
            
            if name == 'Bruteforce':
                # Bruteforce is always 1.0 (baseline)
                ratios[name].append(1.0)
            else:
                try:
                    print(f"  Running {name}...")
                    solution = func(graph)
                    # Validate solution cost
                    if solution.cost is None or solution.cost == float('inf') or solution.cost <= 0:
                        print(f"  Warning: {name} returned invalid cost: {solution.cost}")
                        ratios[name].append(None)
                    else:
                        ratio = solution.cost / bruteforce_cost
                        ratios[name].append(ratio)
                        print(f"  {name} cost: {solution.cost}, ratio: {ratio:.3f}")
                except Exception as e:
                    print(f"  Error running {name}: {e}")
                    ratios[name].append(None)

    # Plotting
    plt.figure(figsize=(10, 6))
    
    for name, ratio_list in ratios.items():
        # Filter out None values
        valid_sizes = [s for s, r in zip(sizes, ratio_list) if r is not None]
        valid_ratios = [r for r in ratio_list if r is not None]
        plt.plot(valid_sizes, valid_ratios, marker='o', label=name, linewidth=2, markersize=6)
    
    plt.xlabel('Number of Cities')
    plt.ylabel('Approximation Ratio')
    plt.title('Solution Quality Comparison (Approximation Ratios)')
    plt.xticks(sizes)  # Ensure x-axis shows integer values for cities
    plt.ylim(0.95, 3.0)  # Set y-axis range from 0.95 to 3.0 (space below 1.0 for visual appeal)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path = os.path.join(os.path.dirname(__file__), 'plots', 'quality_comparison.png')
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    run_benchmark()

