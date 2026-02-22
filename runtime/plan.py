"""
Runtime: Plan Module

Load block index and create execution plan.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional


@dataclass
class TensorMeta:
    """Metadata for a single tensor within a block."""
    layer: str
    file: str
    offset: int
    bytes: int


@dataclass
class Block:
    """Represents a single block (group of layers)."""
    block_id: int
    file: str
    layers: List[str]
    tensors: List[TensorMeta]
    bytes: int


def load_index(index_path: str) -> Dict[str, Any]:
    """Load the index.json file."""
    with open(index_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_plan(index_path: str) -> List[Block]:
    """
    Load execution plan from index.json.
    
    Returns list of Block objects in execution order.
    """
    index = load_index(index_path)
    blocks = []
    
    for b in index["blocks"]:
        tensors = [
            TensorMeta(
                layer=t["layer"],
                file=t["file"],
                offset=t["offset"],
                bytes=t["bytes"],
            )
            for t in b.get("tensors", [])
        ]
        
        blocks.append(Block(
            block_id=b["block_id"],
            file=b["file"],
            layers=b["layers"],
            tensors=tensors,
            bytes=b.get("bytes", 0),
        ))
    
    return blocks


def create_plan(
    index_path: str,
    prefetch_depth: int = 2,
) -> Dict[str, Any]:
    """
    Create a full execution plan.
    
    This includes:
    - Block execution order
    - Prefetch schedule
    - Memory estimates
    """
    blocks = load_plan(index_path)
    index = load_index(index_path)
    
    # Calculate total memory requirements
    total_bytes = sum(b.bytes for b in blocks)
    avg_block_bytes = total_bytes // len(blocks) if blocks else 0
    
    # Estimate GPU memory needed (assuming weights + activations)
    # Rule of thumb: weights are ~2 bytes/param for fp16, activations ~3x
    estimated_params = index.get("estimated_params", 0)
    gpu_memory_bytes = estimated_params * 2 * 4  # Conservative estimate
    
    return {
        "total_blocks": len(blocks),
        "total_bytes": total_bytes,
        "avg_block_bytes": avg_block_bytes,
        "layers_per_block": index.get("layers_per_block", 1),
        "prefetch_depth": prefetch_depth,
        "blocks": blocks,
        "estimated_gpu_memory_mb": gpu_memory_bytes / 1e6,
    }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python plan.py <index.json>")
        sys.exit(1)
    
    plan = create_plan(sys.argv[1])
    print(f"Plan loaded: {plan['total_blocks']} blocks")
    print(f"Total size: {plan['total_bytes'] / 1e9:.2f} GB")
    print(f"Avg block size: {plan['avg_block_bytes'] / 1e6:.1f} MB")
    print(f"Prefetch depth: {plan['prefetch_depth']}")
