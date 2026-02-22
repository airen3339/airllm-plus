#!/usr/bin/env python3
"""
Bench: Run CPU Pipeline

Run the full inference pipeline on CPU (no GPU needed).
Tests different prefetch depths to find optimal configuration.

This validates the design before we have GPU access.
"""

import sys
import json
import time
import argparse
from pathlib import Path

# Add runtime to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from runtime.executor import Executor, PipelineConfig, PipelineStats


def run_pipeline(
    root_dir: str,
    prefetch_depth: int = 2,
    mock_compute_ms: float = 30.0,
    verbose: bool = True,
) -> PipelineStats:
    """Run pipeline with given configuration."""
    config = PipelineConfig(
        prefetch_depth=prefetch_depth,
        mock_compute_ms=mock_compute_ms,
        use_buffered_store=True,
        use_prefetch=(prefetch_depth > 1),
    )
    
    executor = Executor(root_dir, config)
    stats = executor.run()
    
    if verbose:
        print(f"\n=== Prefetch Depth {prefetch_depth} ===")
        print(f"Total time: {stats.total_time_s:.2f}s")
        print(f"I/O time: {stats.io_time_s:.2f}s")
        print(f"Compute time: {stats.compute_time_s:.2f}s")
        print(f"Prefetch time: {stats.prefetch_time_s:.2f}s")
        print(f"Cache hits: {stats.cache_hits}, misses: {stats.cache_misses}")
    
    return stats


def sweep_prefetch_depth(
    root_dir: str,
    depths: list = None,
    mock_compute_ms: float = 30.0,
) -> dict:
    """Sweep prefetch depth and compare results."""
    if depths is None:
        depths = [1, 2, 4, 8]
    
    results = {}
    
    for depth in depths:
        print(f"\n{'='*50}")
        print(f"Testing prefetch depth = {depth}")
        print(f"{'='*50}")
        
        stats = run_pipeline(
            root_dir=root_dir,
            prefetch_depth=depth,
            mock_compute_ms=mock_compute_ms,
            verbose=True,
        )
        
        results[depth] = {
            "total_time_s": stats.total_time_s,
            "io_time_s": stats.io_time_s,
            "compute_time_s": stats.compute_time_s,
            "prefetch_time_s": stats.prefetch_time_s,
            "cache_hits": stats.cache_hits,
            "cache_misses": stats.cache_misses,
        }
    
    # Summary
    print(f"\n{'='*50}")
    print("SUMMARY: Prefetch Depth Sweep")
    print(f"{'='*50}")
    print(f"{'Depth':<8} {'Total':<10} {'I/O':<10} {'Compute':<10} {'Speedup':<10}")
    print("-" * 50)
    
    baseline = results[1]["total_time_s"]
    for depth, r in results.items():
        speedup = baseline / r["total_time_s"]
        print(f"{depth:<8} {r['total_time_s']:<10.2f} "
              f"{r['io_time_s']:<10.2f} {r['compute_time_s']:<10.2f} "
              f"{speedup:<10.2f}x")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run CPU Pipeline Benchmark")
    parser.add_argument(
        "root_dir", 
        nargs="?",
        default="dummy_model/packed",
        help="Directory with packed blocks (default: dummy_model/packed)"
    )
    parser.add_argument("--depth", type=int, help="Single prefetch depth to test")
    parser.add_argument("--sweep", action="store_true", help="Sweep all depths")
    parser.add_argument("--mock_compute_ms", type=float, default=30.0,
                       help="Mock compute time per block (ms)")
    parser.add_argument("--output", help="Output JSON file for results")
    
    args = parser.parse_args()
    
    root_dir = args.root_dir
    
    # Check if directory exists
    if not Path(root_dir).exists():
        print(f"Error: {root_dir} does not exist")
        print("\nFirst generate dummy data:")
        print(f"  python pack/pack_blocks.py --generate_dummy")
        print(f"\nThen run:")
        print(f"  python bench/run_pipeline_cpu.py {root_dir}")
        sys.exit(1)
    
    # Run benchmark
    if args.sweep:
        results = sweep_prefetch_depth(
            root_dir=root_dir,
            mock_compute_ms=args.mock_compute_ms,
        )
    else:
        depth = args.depth or 2
        stats = run_pipeline(
            root_dir=root_dir,
            prefetch_depth=depth,
            mock_compute_ms=args.mock_compute_ms,
        )
        results = {depth: stats.to_dict()}
    
    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
