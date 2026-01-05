import sys
import os
import matplotlib.pyplot as plt
import tracemalloc

# Add parent directory to path to allow importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.graph_generators import generate_graph
from utils.algorithm_adapter import run_bruteforce, run_mst, run_sim_annealing, run_qaoa

def run_benchmark():
    # Define problem sizes
    sizes = range(3, 21)
    
    algorithms = {
        'Bruteforce': run_bruteforce,
        'MST': run_mst,
        'Simulated Annealing': run_sim_annealing,
        'QAOA': run_qaoa
    }
    
    results = {algo: [] for algo in algorithms}
    
    print("Running Memory Benchmark...")
    
    for n in sizes:
        print(f"Processing size {n}...")
        graph = generate_graph('fully_connected', num_nodes=n, seed=42)
        
        for name, func in algorithms.items():
            # Apply limits for computationally expensive algorithms
            if name == 'QAOA' and n > 5:
                results[name].append(None)
                continue
            if name == 'Bruteforce' and n > 15:
                results[name].append(None)
                continue

            try:
                # Start tracing
                tracemalloc.start()
                
                # Run algorithm
                solution = func(graph)
                
                # Get peak memory usage
                current, peak = tracemalloc.get_traced_memory()
                
                tracemalloc.stop()
                
                # Convert bytes to MB
                peak_mb = peak / (1024 * 1024)
                results[name].append(peak_mb)
                
            except Exception as e:
                print(f"Error running {name} on size {n}: {e}")
                results[name].append(None)

    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Collect all valid memory values to determine appropriate y-axis range
    all_mems = []
    for name, mems in results.items():
        valid_mems = [m for m in mems if m is not None and m > 0]
        all_mems.extend(valid_mems)
    
    for name, mems in results.items():
        valid_sizes = [s for s, m in zip(sizes, mems) if m is not None]
        valid_mems = [m for m in mems if m is not None]
        plt.plot(valid_sizes, valid_mems, marker='o', label=name, linewidth=2, markersize=6)
    
    plt.xlabel('Number of Cities')
    plt.ylabel('Peak Memory Usage (MB)')
    plt.title('Memory Usage vs Number of Cities')
    plt.xticks(sizes)
    plt.legend()
    plt.yscale('log')
    
    # Set y-axis limits to ensure all algorithms are visible
    if all_mems:
        min_mem = min(all_mems)
        max_mem = max(all_mems)
        plt.ylim(bottom=min_mem * 0.5, top=max_mem * 2)
    
    plt.grid(True, which='both', alpha=0.3)
    
    output_path = os.path.join(os.path.dirname(__file__), 'plots', 'memory_comparison.png')
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    run_benchmark()

