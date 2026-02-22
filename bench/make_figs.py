#!/usr/bin/env python3
"""
Generate paper-quality figures from benchmark results.

Usage:
    python3 bench/make_figs.py results.json
    python3 bench/make_figs.py results.json --output figs/
"""

import json
import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed, skipping figure generation")


def load_results(results_path):
    """Load benchmark results from JSON."""
    with open(results_path) as f:
        return json.load(f)


def plot_prefetch_depth_sweep(results, output_path=None):
    """Plot prefetch depth vs speedup."""
    if not HAS_MATPLOTLIB:
        print("Skipping: matplotlib not available")
        return
    
    # Extract data
    depths = sorted(results.keys(), key=lambda x: int(x))
    times = [results[d]["total_time_s"] for d in depths]
    io_times = [results[d]["io_time_s"] for d in depths]
    compute_times = [results[d]["compute_time_s"] for d in depths]
    
    # Calculate speedup (relative to depth=1)
    baseline = times[0]
    speedups = [baseline / t for t in times]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Plot 1: Time breakdown
    x = list(range(len(depths)))
    ax1.bar(x, io_times, label='I/O', alpha=0.7, color='#2ecc71')
    ax1.bar(x, compute_times, bottom=io_times, label='Compute', alpha=0.7, color='#3498db')
    ax1.set_xticks(x)
    ax1.set_xticklabels(depths)
    ax1.set_xlabel('Prefetch Depth')
    ax1.set_ylabel('Time (s)')
    ax1.set_title('Pipeline Time Breakdown')
    ax1.legend()
    
    # Plot 2: Speedup
    ax2.plot(depths, speedups, 'o-', color='#e74c3c', linewidth=2, markersize=8)
    ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Prefetch Depth')
    ax2.set_ylabel('Speedup (vs depth=1)')
    ax2.set_title('Prefetch Speedup')
    ax2.grid(True, alpha=0.3)
    
    # Add annotation for best
    best_idx = speedups.index(max(speedups))
    best_depth = depths[best_idx]
    best_speedup = speedups[best_idx]
    ax2.annotate(f'Best: {best_depth}x', 
                 xy=(best_depth, best_speedup),
                 xytext=(best_depth + 0.5, best_speedup * 0.9),
                 arrowprops=dict(arrowstyle='->', color='gray'))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    plt.close()


def plot_io_comparison(io_results, output_path=None):
    """Plot I/O comparison (sequential vs random)."""
    if not HAS_MATPLOTLIB:
        return
    
    methods = []
    throughputs = []
    
    for key, val in io_results.items():
        if 'throughput_gbps' in val:
            methods.append(val.get('method', key))
            throughputs.append(val['throughput_gbps'])
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    bars = ax.bar(methods, throughputs, color=['#2ecc71', '#e74c3c', '#3498db', '#9b59b6'])
    
    ax.set_ylabel('Throughput (GB/s)')
    ax.set_title('I/O Throughput: Sequential vs Random Access')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, throughputs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    plt.close()


def generate_all_figures(results_dir=".", output_dir="figs"):
    """Generate all figures."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Look for results files
    results_files = list(Path(results_dir).glob("*.json"))
    
    if not results_files:
        print(f"No results.json found in {results_dir}")
        print("Run benchmark first:")
        print("  python3 bench/run_pipeline_cpu.py dummy_model/packed --sweep")
        return
    
    for rf in results_files:
        print(f"\nProcessing: {rf.name}")
        
        try:
            data = json.loads(rf.read_text())
            
            # Determine what kind of data it is
            if "1" in data and isinstance(data["1"], dict):
                # Prefetch depth sweep results
                plot_prefetch_depth_sweep(
                    data, 
                    output_path / f"prefetch_sweep_{rf.stem}.png"
                )
            elif "sequential" in data:
                # I/O benchmark results
                plot_io_comparison(
                    data,
                    output_path / f"io_comparison_{rf.stem}.png"
                )
        except Exception as e:
            print(f"  Error: {e}")
    
    print(f"\n✅ Figures saved to: {output_path}/")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate figures from results")
    parser.add_argument("results", nargs="?", default="results.json",
                       help="Results JSON file")
    parser.add_argument("--output", "-o", default="paper/figs",
                       help="Output directory for figures")
    parser.add_argument("--run-bench", action="store_true",
                       help="Run benchmark first")
    
    args = parser.parse_args()
    
    if args.run_bench:
        print("Running benchmark first...")
        import subprocess
        subprocess.run([
            sys.executable, "bench/run_pipeline_cpu.py",
            "dummy_model/packed", "--sweep", "--output", args.results
        ], cwd=Path(__file__).parent.parent)
    
    # Check if file exists
    results_path = Path(args.results)
    if not results_path.exists():
        # Look in common locations
        for candidate in ["results.json", "dummy_model/packed/results.json"]:
            if Path(candidate).exists():
                results_path = Path(candidate)
                break
    
    if results_path.exists():
        generate_all_figures(results_path.parent, args.output)
    else:
        print(f"Results file not found: {args.results}")
        print("\nUsage:")
        print("  # Run benchmark first")
        print("  python3 bench/run_pipeline_cpu.py dummy_model/packed --sweep")
        print("  ")
        print("  # Then generate figures")
        print("  python3 bench/make_figs.py results.json")


if __name__ == "__main__":
    main()
