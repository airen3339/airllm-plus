# AirLLM++

**Faster than AirLLM: Super-block residency with deep prefetching**

## What is AirLLM++?

AirLLM++ is an optimization of AirLLM's layer-swapping approach. Instead of loading one layer at a time (layer-wise), we:

1. **Group layers into super-blocks** (e.g., 4 layers per block)
2. **Use large sequential reads** instead of many small random reads
3. **Deep prefetching** (depth ≥ 2) to overlap I/O with compute

## Quick Start (One Command)

```bash
cd airllm-plus
python3 run.py
```

This will:
1. Generate dummy model data (if needed)
2. Run the full pipeline benchmark
3. Output results to `results.json`

## Project Structure

```
airllm-plus/
├── pack/
│   └── pack_blocks.py       layer shards to block files # Convert
├── runtime/
│   ├── __init__.py
│   ├── plan.py             # Load execution plan from index
│   ├── store.py            # Block I/O (sequential reads)
│   ├── prefetch.py         # Background prefetching
│   ├── cache.py            # LRU cache for blocks
│   └── executor.py         # Main pipeline orchestrator
├── bench/
│   ├── io_bench.py         # I/O throughput benchmarks
│   ├── run_pipeline_cpu.py # Full pipeline benchmark
│   └── make_figs.py        # Generate paper figures
├── paper/figs/             # Output figures
├── run.py                  # Single command runner
└── README.md
```

## Usage

### Generate Dummy Data

```bash
python3 pack/pack_blocks.py \
    --generate_dummy \
    --dummy_layers 32 \
    --dummy_size_mb 50
```

### Run Pipeline Benchmark

```bash
# Single depth
python3 bench/run_pipeline_cpu.py dummy_model/packed --depth 2

# Full sweep
python3 bench/run_pipeline_cpu.py dummy_model/packed --sweep

# Custom parameters
python3 bench/run_pipeline_cpu.py dummy_model/packed \
    --depth 4 \
    --mock_compute_ms 50 \
    --output results.json
```

### Run I/O Benchmark

```bash
python3 bench/io_bench.py
```

### Generate Figures

```bash
# Requires matplotlib
pip install matplotlib
python3 bench/make_figs.py results.json
```

## Key Concepts

### Super-block Residency

Instead of:
```
Load Layer 0 → Compute → Unload → Load Layer 1 → Compute → ...
```

We do:
```
Load Block 0 (4 layers) → Compute all 4 → Load Block 1 → Compute all 4 → ...
```

### Prefetch Depth

| Depth | Behavior | Best For |
|-------|----------|----------|
| 1 | No prefetch (baseline) | Debugging |
| 2 | 1 block ahead | Low latency |
| 4 | 3 blocks ahead | Typical SSD |
| 8 | 7 blocks ahead | Slow HDD / Network |

## Results (Dummy Data)

| Prefetch Depth | Total Time | I/O Time | Speedup |
|----------------|------------|----------|---------|
| 1 | 0.95s | 0.69s | 1.00x |
| **2** | **0.39s** | **0.07s** | **2.44x** |
| 4 | 0.46s | 0.18s | 2.09x |
| 8 | 0.63s | 0.36s | 1.52x |

**Best: 2.44x speedup with depth=2**

## v0.1 vs v0.2

| Feature | v0.1 (current) | v0.2 |
|---------|-----------------|------|
| Block format | Opaque binary | Tensor-level index |
| I/O | Whole block | Range reads |
| Cache | Block-level | Tensor-level |
| GPU support | Mock compute | Real forward pass |
| Prefetch | Thread-based | AsyncIO |

## Credits

Based on ideas from:
- [AirLLM](https://github.com/lyogavin/airllm)
- [Linear Attention](https://arxiv.org/abs/2006.16236)
- [Mamba SSMs](https://arxiv.org/abs/2312.00752)

## License

Apache 2.0
