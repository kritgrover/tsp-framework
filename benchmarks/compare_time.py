import sys
import os
import matplotlib.pyplot as plt
import time

# Add parent directory to path to allow importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.graph_generators import generate_graph
from utils.algorithm_adapter import run_bruteforce, run_mst, run_sim_annealing, run_qaoa

def run_benchmark():
    # Define problem sizes
    sizes = range(3, 31)
    
    algorithms = {
        'Bruteforce': run_bruteforce,
        'MST': run_mst,
        'Simulated Annealing': run_sim_annealing,
        'QAOA': run_qaoa
    }
    
    results = {algo: [] for algo in algorithms}
    
    print("Running Time Benchmark...")
    
    for n in sizes:
        print(f"Processing size {n}...")
        graph = generate_graph('fully_connected', num_nodes=n, seed=42)
        
        for name, func in algorithms.items():
            # Apply limits
            if name == 'QAOA' and n > 5:
                results[name].append(None)
                continue
            if name == 'Bruteforce' and n > 10:
                results[name].append(None)
                continue

            try:
                solution = func(graph)
                time_taken = solution.metadata.get('time_taken', 0)
                results[name].append(time_taken)
            except Exception as e:
                print(f"Error running {name} on size {n}: {e}")
                results[name].append(None)

    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Collect all valid times to determine appropriate y-axis range
    all_times = []
    for name, times in results.items():
        valid_times = [t for t in times if t is not None and t > 0]
        all_times.extend(valid_times)
    
    for name, times in results.items():
        valid_sizes = [s for s, t in zip(sizes, times) if t is not None]
        valid_times = [t for t in times if t is not None]
        plt.plot(valid_sizes, valid_times, marker='o', label=name, linewidth=2, markersize=6)
    
    plt.xlabel('Number of Cities')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Execution Time vs Number of Cities')
    plt.xticks(sizes)
    plt.legend()
    plt.yscale('log')
    
    # Set y-axis limits to ensure all algorithms are visible
    if all_times:
        min_time = min(all_times)
        max_time = max(all_times)
        plt.ylim(bottom=1e-4, top=max_time * 2)
    
    plt.grid(True, which='both', alpha=0.3)
    
    output_path = os.path.join(os.path.dirname(__file__), 'plots', 'time_comparison.png')
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    run_benchmark()

