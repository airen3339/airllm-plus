"""
Bench: I/O Benchmark

Measures I/O throughput for different access patterns:
1. Sequential large reads (what we want)
2. Random small reads (what AirLLM does)
3. Effect of prefetch depth
"""

import os
import time
import random
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from typing import List, Tuple, Dict, Any
import argparse


def create_test_files(
    output_dir: str,
    num_files: int = 100,
    file_size_mb: int = 10,
) -> List[Path]:
    """Create test files for benchmarking."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    files = []
    chunk = b"x" * (1024 * 1024)  # 1MB chunk
    
    print(f"Creating {num_files} files of {file_size_mb}MB each...")
    
    for i in range(num_files):
        fpath = out_path / f"file_{i:04d}.bin"
        with open(fpath, "wb") as f:
            for _ in range(file_size_mb):
                f.write(chunk)
        files.append(fpath)
    
    print(f"Created {len(files)} files")
    return files


def bench_sequential_read(files: List[Path]) -> Dict[str, Any]:
    """
    Benchmark sequential reading of all files.
    
    This is what our block format achieves.
    """
    total_bytes = 0
    t_start = time.perf_counter()
    
    for fpath in files:
        with open(fpath, "rb") as f:
            data = f.read()
            total_bytes += len(data)
    
    t_end = time.perf_counter()
    elapsed = t_end - t_start
    
    return {
        "method": "sequential",
        "files": len(files),
        "total_mb": total_bytes / 1e6,
        "time_s": elapsed,
        "throughput_mbps": (total_bytes / 1e6) / elapsed,
        "throughput_gbps": (total_bytes / 1e9) / elapsed,
    }


def bench_random_read(files: List[Path], num_reads: int = 100) -> Dict[str, Any]:
    """
    Benchmark random small reads (like AirLLM's layer-wise loading).
    """
    # Read random small chunks
    chunk_size = 1024 * 1024  # 1MB chunks
    total_bytes = 0
    t_start = time.perf_counter()
    
    for _ in range(num_reads):
        fpath = random.choice(files)
        with open(fpath, "rb") as f:
            offset = random.randint(0, os.path.getsize(fpath) - chunk_size)
            f.seek(offset)
            data = f.read(chunk_size)
            total_bytes += len(data)
    
    t_end = time.perf_counter()
    elapsed = t_end - t_start
    
    return {
        "method": "random_small",
        "reads": num_reads,
        "total_mb": total_bytes / 1e6,
        "time_s": elapsed,
        "throughput_mbps": (total_bytes / 1e6) / elapsed,
        "throughput_gbps": (total_bytes / 1e9) / elapsed,
    }


def bench_concatenated_read(
    files: List[Path],
    concat_count: int = 4,
) -> Dict[str, Any]:
    """
    Benchmark reading concatenated blocks (like our block format).
    """
    # Create temp concatenated file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as tmp:
        tmp_path = tmp.name
    
    try:
        # Concatenate
        print(f"Creating concatenated file with {concat_count} files...")
        with open(tmp_path, "wb") as out_f:
            for fpath in files[:concat_count]:
                with open(fpath, "rb") as in_f:
                    out_f.write(in_f.read())
        
        file_size = os.path.getsize(tmp_path)
        
        # Read multiple times
        num_iterations = 10
        t_start = time.perf_counter()
        
        for _ in range(num_iterations):
            with open(tmp_path, "rb") as f:
                data = f.read()
        
        t_end = time.perf_counter()
        elapsed = t_end - t_start
        total_bytes = file_size * num_iterations
        
        return {
            "method": "concatenated",
            "files_per_block": concat_count,
            "block_size_mb": file_size / 1e6,
            "iterations": num_iterations,
            "total_mb": total_bytes / 1e6,
            "time_s": elapsed,
            "throughput_mbps": (total_bytes / 1e6) / elapsed,
            "throughput_gbps": (total_bytes / 1e9) / elapsed,
        }
    finally:
        os.unlink(tmp_path)


def run_io_benchmarks(output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Run all I/O benchmarks and return results.
    """
    # Create temp directory
    if output_dir:
        test_dir = Path(output_dir)
    else:
        test_dir = Path(tempfile.mkdtemp(prefix="airllm_bench_"))
    
    print(f"Benchmark directory: {test_dir}")
    
    # Create test files
    files = create_test_files(str(test_dir), num_files=50, file_size_mb=10)
    
    results = {}
    
    # 1. Sequential read
    print("\n=== Sequential Read (1 file at a time) ===")
    results["sequential"] = bench_sequential_read(files)
    print(f"Throughput: {results['sequential']['throughput_gbps']:.2f} GB/s")
    
    # 2. Random small reads
    print("\n=== Random Small Reads ===")
    results["random_small"] = bench_random_read(files, num_reads=50)
    print(f"Throughput: {results['random_small']['throughput_mbps']:.1f} MB/s")
    
    # 3. Concatenated block reads
    for concat_count in [2, 4, 8]:
        print(f"\n=== Concatenated Block Read (k={concat_count}) ===")
        key = f"concat_{concat_count}"
        results[key] = bench_concatenated_read(files, concat_count)
        print(f"Throughput: {results[key]['throughput_gbps']:.2f} GB/s")
    
    # Cleanup
    if not output_dir:
        shutil.rmtree(test_dir)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="I/O Benchmark")
    parser.add_argument("--output_dir", help="Output directory (creates temp if not set)")
    parser.add_argument("--num_files", type=int, default=50)
    parser.add_argument("--file_size_mb", type=int, default=10)
    args = parser.parse_args()
    
    if args.output_dir:
        create_test_files(args.output_dir, args.num_files, args.file_size_mb)
    
    results = run_io_benchmarks(args.output_dir)
    
    print("\n" + "="*50)
    print("BENCHMARK RESULTS")
    print("="*50)
    
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"  Throughput: {result.get('throughput_mbps', result.get('throughput_gbps') * 1000):.1f} MB/s")
        print(f"  Time: {result['time_s']:.2f}s")
