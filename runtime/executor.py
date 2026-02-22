"""
Runtime: Executor Module

Orchestrates the full inference pipeline:
- Plan: block execution order
- Store: block I/O
- Prefetch: background loading
- Cache: GPU memory management

v0.1: CPU-only with mock compute
v0.2: Real GPU execution with pinned memory
"""

import time
import threading
from pathlib import Path
from typing import Optional, Callable, List, Dict, Any
from dataclasses import dataclass

from .plan import load_plan, create_plan
from .store import BlockStore, BufferedBlockStore
from .prefetch import Prefetcher
from .cache import BlockCache


@dataclass
class PipelineConfig:
    """Configuration for the inference pipeline."""
    prefetch_depth: int = 2
    max_cache_mb: int = 2048
    max_cache_blocks: int = 4
    mock_compute_ms: float = 30.0  # Simulate GPU compute time
    use_buffered_store: bool = True
    use_prefetch: bool = True


@dataclass
class PipelineStats:
    """Runtime statistics."""
    total_blocks: int = 0
    total_time_s: float = 0.0
    io_time_s: float = 0.0
    compute_time_s: float = 0.0
    prefetch_time_s: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_blocks": self.total_blocks,
            "total_time_s": self.total_time_s,
            "io_time_s": self.io_time_s,
            "compute_time_s": self.compute_time_s,
            "prefetch_time_s": self.prefetch_time_s,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "io_throughput_gbps": (
                0 if self.total_time_s == 0 
                else (self.io_time_s / self.total_time_s)
            ),
        }


class Executor:
    """
    Main inference pipeline executor.
    
    Coordinates:
    1. Prefetch thread: loads next blocks in background
    2. Main thread: processes current block (compute)
    3. Cache: keeps recently used blocks in memory
    """
    
    def __init__(
        self, 
        root_dir: str,
        config: Optional[PipelineConfig] = None,
    ):
        self.root_dir = Path(root_dir)
        self.config = config or PipelineConfig()
        
        # Load plan
        index_path = self.root_dir / "index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"Index not found: {index_path}")
        
        self.plan = create_plan(str(index_path), self.config.prefetch_depth)
        self.blocks = self.plan["blocks"]
        
        # Initialize components
        if self.config.use_buffered_store:
            self.store = BufferedBlockStore(str(self.root_dir))
        else:
            self.store = BlockStore(str(self.root_dir))
        
        self.cache = BlockCache(
            max_size_mb=self.config.max_cache_mb,
            max_blocks=self.config.max_cache_blocks,
        )
        
        self.prefetcher: Optional[Prefetcher] = None
        if self.config.use_prefetch:
            self.prefetcher = Prefetcher(self.store, depth=self.config.prefetch_depth)
        
        # Stats
        self.stats = PipelineStats()
    
    def run(self, compute_fn: Optional[Callable] = None) -> PipelineStats:
        """
        Run the full inference pipeline.
        
        Args:
            compute_fn: Optional function to call for each block.
                       If None, uses mock compute (time.sleep).
        
        Returns:
            PipelineStats with timing information
        """
        self.stats = PipelineStats()
        total_blocks = len(self.blocks)
        
        # Start prefetcher
        if self.prefetcher:
            self.prefetcher.start()
        
        t_start = time.perf_counter()
        
        # Main loop
        for i, block in enumerate(self.blocks):
            t_block_start = time.perf_counter()
            
            # Try cache first
            cached_data = self.cache.get(block.block_id)
            if cached_data is not None:
                self.stats.cache_hits += 1
                block_data = cached_data
            else:
                self.stats.cache_misses += 1
                
                # Use prefetched data if available
                if self.prefetcher and i >= self.config.prefetch_depth:
                    # Wait for prefetched block
                    t_prefetch_start = time.perf_counter()
                    prefetched = self.prefetcher.get(timeout=30)
                    self.stats.prefetch_time_s += time.perf_counter() - t_prefetch_start
                    block_data = prefetched.data
                else:
                    # Direct read
                    t_io_start = time.perf_counter()
                    block_data = self.store.read_whole_block(block.file)
                    self.stats.io_time_s += time.perf_counter() - t_io_start
                
                # Cache it
                self.cache.put(block.block_id, block_data, block.bytes)
            
            # Compute
            t_compute_start = time.perf_counter()
            if compute_fn:
                compute_fn(block, block_data)
            else:
                # Mock compute
                time.sleep(self.config.mock_compute_ms / 1000.0)
            
            self.stats.compute_time_s += time.perf_counter() - t_compute_start
            
            # Submit next block for prefetching
            if self.prefetcher:
                next_idx = i + self.config.prefetch_depth
                if next_idx < total_blocks:
                    self.prefetcher.submit(self.blocks[next_idx])
            
            # Progress
            if (i + 1) % 4 == 0:
                elapsed = time.perf_counter() - t_start
                eta = elapsed / (i + 1) * (total_blocks - i - 1)
                print(f"Progress: {i+1}/{total_blocks} blocks, "
                      f"elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s")
        
        self.stats.total_time_s = time.perf_counter() - t_start
        self.stats.total_blocks = total_blocks
        
        # Stop prefetcher
        if self.prefetcher:
            self.prefetcher.stop()
        
        return self.stats
    
    def run_with_depth_sweep(
        self,
        depths: List[int] = [1, 2, 4, 8],
    ) -> Dict[int, PipelineStats]:
        """
        Run pipeline with different prefetch depths.
        
        Useful for benchmarking to find optimal depth.
        """
        results = {}
        
        for depth in depths:
            print(f"\n=== Testing prefetch depth = {depth} ===")
            
            # Reconfigure
            self.config.prefetch_depth = depth
            self.prefetcher = Prefetcher(self.store, depth=depth)
            
            # Clear cache for fair comparison
            self.cache.clear()
            
            # Run
            stats = self.run()
            results[depth] = stats
            
            print(f"Depth {depth}: {stats.total_time_s:.2f}s total, "
                  f"{stats.io_time_s:.2f}s I/O, "
                  f"{stats.compute_time_s:.2f}s compute")
        
        return results


def run_cpu_pipeline(
    root_dir: str,
    prefetch_depth: int = 2,
    mock_compute_ms: float = 30.0,
) -> PipelineStats:
    """
    Convenience function to run CPU-only pipeline.
    
    This is what you run now (without GPU) to validate the design.
    """
    config = PipelineConfig(
        prefetch_depth=prefetch_depth,
        mock_compute_ms=mock_compute_ms,
    )
    
    executor = Executor(root_dir, config)
    return executor.run()


if __name__ == "__main__":
    import sys
    import json
    
    if len(sys.argv) < 2:
        print("Usage: python executor.py <packed_model_dir> [prefetch_depth]")
        sys.exit(1)
    
    root_dir = sys.argv[1]
    depth = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    
    print(f"Running pipeline on: {root_dir}")
    print(f"Prefetch depth: {depth}")
    
    stats = run_cpu_pipeline(root_dir, prefetch_depth=depth)
    
    print(f"\n=== Results ===")
    print(f"Total time: {stats.total_time_s:.2f}s")
    print(f"I/O time: {stats.io_time_s:.2f}s")
    print(f"Compute time: {stats.compute_time_s:.2f}s")
    print(f"Cache hits: {stats.cache_hits}")
    print(f"Cache misses: {stats.cache_misses}")
