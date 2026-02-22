#!/usr/bin/env python3
"""
AirLLM++ Block Packer

Converts layer-wise model shards into large sequential block files.
This reduces I/O overhead from many small random reads to few large sequential reads.

v0.1: Simple concatenation of layer files into blocks
v0.2: Will support tensor-level indexing (QKV/MLP fusion)
"""

import os
import json
import struct
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import argparse


def discover_layers(layer_dir: Path) -> List[Path]:
    """
    Discover layer directories by naming convention.
    Supports: layer_000, layer_001, ... or model.00, model.01, ...
    """
    # Look for directories starting with "layer_" or numeric prefixes
    candidates = []
    for item in sorted(layer_dir.iterdir()):
        if item.is_dir():
            # Match layer_XXX or model.XXX patterns
            name = item.name
            if name.startswith("layer_") or name.startswith("model."):
                candidates.append(item)
            # Also match numeric-only directories (e.g., "0", "1", "2")
            elif name.isdigit():
                candidates.append(item)
    
    # Sort by numeric suffix if possible
    def sort_key(p: Path) -> int:
        name = p.name
        if name.isdigit():
            return int(name)
        # Extract number from layer_000 or model.00
        for sep in ("layer_", "model."):
            if name.startswith(sep):
                try:
                    return int(name[len(sep):])
                except ValueError:
                    pass
        return 0
    
    return sorted(candidates, key=sort_key)


def list_tensors_in_layer(layer_path: Path) -> List[Path]:
    """
    List all tensor files in a layer directory.
    Supports: .safetensors, .bin, .pt, .onnx
    """
    patterns = ["*.safetensors", "*.bin", "*.pt", "*.onnx", "*.msgpack"]
    files = []
    for pattern in patterns:
        files.extend(layer_path.glob(pattern))
    return sorted(files)


def pack_blocks(
    input_root: str,
    output_root: str,
    layers_per_block: int = 4,
    target_block_bytes: Optional[int] = None,
    format: str = "bin",
) -> Dict[str, Any]:
    """
    Main packing function.
    
    Args:
        input_root: Directory containing layer_* subdirectories
        output_root: Where to write blocks/ and index.json
        layers_per_block: Number of layers to group per block
        target_block_bytes: Target size for each block (optional)
        format: Output format - "bin" or "safetensors"
    
    Returns:
        Statistics dictionary
    """
    in_root = Path(input_root)
    out_root = Path(output_root)
    blocks_dir = out_root / "blocks"
    blocks_dir.mkdir(parents=True, exist_ok=True)
    
    # Discover layers
    layer_dirs = discover_layers(in_root)
    if not layer_dirs:
        raise RuntimeError(
            f"No layer directories found under {in_root}. "
            f"Expected directories named layer_000, layer_001, etc."
        )
    
    print(f"Discovered {len(layer_dirs)} layers")
    
    # Build index
    index: Dict[str, Any] = {
        "format": "airllmpp.block.v0.1",
        "version": "0.1.0",
        "layers_per_block": layers_per_block,
        "total_layers": len(layer_dirs),
        "source": str(in_root),
        "blocks": [],
    }
    
    # Group layers into blocks
    block_id = 0
    current_block_layers: List[Path] = []
    current_block_bytes = 0
    
    stats = {
        "total_blocks": 0,
        "total_bytes": 0,
        "layers_processed": 0,
    }
    
    for layer_dir in layer_dirs:
        current_block_layers.append(layer_dir)
        
        # Estimate layer size (quick scan)
        layer_size = sum(f.stat().st_size for f in list_tensors_in_layer(layer_dir))
        current_block_bytes += layer_size
        
        # Flush if we've accumulated enough layers
        if len(current_block_layers) >= layers_per_block:
            block_info = _write_block(
                block_id=block_id,
                layers=current_block_layers,
                blocks_dir=blocks_dir,
                format=format,
            )
            index["blocks"].append(block_info)
            
            stats["total_blocks"] += 1
            stats["total_bytes"] += block_info["bytes"]
            stats["layers_processed"] += len(current_block_layers)
            
            block_id += 1
            current_block_layers = []
            current_block_bytes = 0
    
    # Flush remaining layers
    if current_block_layers:
        block_info = _write_block(
            block_id=block_id,
            layers=current_block_layers,
            blocks_dir=blocks_dir,
            format=format,
        )
        index["blocks"].append(block_info)
        
        stats["total_blocks"] += 1
        stats["total_bytes"] += block_info["bytes"]
        stats["layers_processed"] += len(current_block_layers)
    
    # Write index
    index_path = out_root / "index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    
    print(f"\n=== Packing Complete ===")
    print(f"Blocks created: {stats['total_blocks']}")
    print(f"Total size: {stats['total_bytes'] / 1e9:.2f} GB")
    print(f"Layers processed: {stats['layers_processed']}")
    print(f"Index written to: {index_path}")
    
    return stats


def _write_block(
    block_id: int,
    layers: List[Path],
    blocks_dir: Path,
    format: str,
) -> Dict[str, Any]:
    """Write a single block file."""
    block_name = f"block_{block_id:03d}.{format}"
    block_path = blocks_dir / block_name
    
    meta = {
        "block_id": block_id,
        "file": f"blocks/{block_name}",
        "layers": [ld.name for ld in layers],
        "tensors": [],
    }
    
    offset = 0
    with open(block_path, "wb") as wf:
        for layer_dir in layers:
            tensor_files = list_tensors_in_layer(layer_dir)
            for tf in tensor_files:
                data = tf.read_bytes()
                size = len(data)
                
                wf.write(data)
                
                meta["tensors"].append({
                    "layer": layer_dir.name,
                    "file": tf.name,
                    "offset": offset,
                    "bytes": size,
                })
                offset += size
    
    meta["bytes"] = offset
    return meta


def generate_dummy_layers(
    output_dir: str | Path,
    num_layers: int = 32,
    layer_size_mb: int = 500,
) -> None:
    """
    Generate dummy layer files for testing.
    
    This creates synthetic "layer shards" so we can test the packer
    without needing actual model weights.
    """
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {num_layers} dummy layers ({layer_size_mb}MB each)...")
    
    for i in range(num_layers):
        layer_dir = out_root / f"layer_{i:03d}"
        layer_dir.mkdir(exist_ok=True)
        
        # Create dummy tensor files (simulate QKV, MLP, etc.)
        tensor_names = [
            "attn.qkv.weight",
            "attn.o.weight",
            "mlp.gate.weight",
            "mlp.up.weight",
            "mlp.down.weight",
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
        ]
        
        # Distribute layer size across tensors
        tensor_size = (layer_size_mb * 1024 * 1024) // len(tensor_names)
        
        for tname in tensor_names:
            tensor_path = layer_dir / f"{tname}.bin"
            # Write random-ish data (reproducible)
            data = bytes([(i * 17 + ord(c)) % 256 for c in tname]) * (tensor_size // len(tname))
            tensor_path.write_bytes(data)
    
    print(f"Generated dummy layers in: {out_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AirLLM++ Block Packer")
    parser.add_argument("--input_root", help="Input directory with layer_* subdirs")
    parser.add_argument("--output_root", help="Output directory for blocks/")
    parser.add_argument("--layers_per_block", type=int, default=4, 
                        help="Layers per block (default: 4)")
    parser.add_argument("--generate_dummy", action="store_true",
                        help="Generate dummy test data")
    parser.add_argument("--dummy_layers", type=int, default=32,
                        help="Number of dummy layers to generate")
    parser.add_argument("--dummy_size_mb", type=int, default=100,
                        help="Size per dummy layer in MB")
    
    args = parser.parse_args()
    
    if args.generate_dummy:
        # Generate test data in current directory
        dummy_dir = Path("dummy_model")
        generate_dummy_layers(dummy_dir, args.dummy_layers, args.dummy_size_mb)
        args.input_root = str(dummy_dir)
        args.output_root = str(dummy_dir / "packed")
    
    if not args.input_root or not args.output_root:
        parser.print_help()
        print("\nExample usage:")
        print("  # Generate dummy test data")
        print("  python pack_blocks.py --generate_dummy --dummy_layers 32 --dummy_size_mb 100")
        print("\n  # Pack real model layers")
        print("  python pack_blocks.py --input_root /path/to/layers --output_root /path/to/output --layers_per_block 4")
        exit(1)
    
    pack_blocks(
        input_root=args.input_root,
        output_root=args.output_root,
        layers_per_block=args.layers_per_block,
    )
