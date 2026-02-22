#!/usr/bin/env python3
"""
AirLLM++ Single Command Runner

Does everything in one command:
1. Generate dummy data (if needed)
2. Run full pipeline benchmark
3. Generate paper figures

Usage:
    python run.py              # Default run
    python run.py --sweep      # Full sweep
    python run.py --figs-only  # Just generate figures
"""

import subprocess
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DUMMY_DIR = PROJECT_ROOT / "dummy_model"
PACKED_DIR = DUMMY_DIR / "packed"


def cmd(c, cwd=None):
    """Run command and return success."""
    print(f"\n{'='*60}")
    print(f"> {c}")
    print(f"{'='*60}")
    result = subprocess.run(c, shell=True, cwd=cwd or PROJECT_ROOT)
    return result.returncode == 0


def main():
    # Parse args
    sweep = "--sweep" in sys.argv
    figs_only = "--figs-only" in sys.argv
    clean = "--clean" in sys.argv
    
    # Clean if requested
    if clean:
        print("🧹 Cleaning up...")
        cmd(f"rm -rf {DUMMY_DIR}")
    
    # Step 1: Generate data if needed
    if not figs_only:
        if not PACKED_DIR.exists():
            print("\n📦 Generating dummy model data...")
            if not cmd("python3 pack/pack_blocks.py --generate_dummy --dummy_layers 32 --dummy_size_mb 50"):
                print("ERROR: Failed to generate data")
                sys.exit(1)
        else:
            print(f"\n✅ Using existing data: {PACKED_DIR}")
        
        # Step 2: Run benchmark
        output_file = "results.json"
        print("\n🚀 Running pipeline benchmark...")
        
        if sweep:
            if not cmd(f"python3 bench/run_pipeline_cpu.py {PACKED_DIR} --sweep --output {output_file}"):
                print("ERROR: Benchmark failed")
                sys.exit(1)
        else:
            if not cmd(f"python3 bench/run_pipeline_cpu.py {PACKED_DIR} --depth 2 --output {output_file}"):
                print("ERROR: Benchmark failed")
                sys.exit(1)
    
    # Step 3: Generate figures
    print("\n📊 Generating figures...")
    if not cmd("python3 bench/make_figs.py results.json"):
        print("Warning: Figure generation failed (may need matplotlib)")
    
    # Show results
    print("\n" + "="*60)
    print("🎉 DONE! AirLLM++ pipeline complete.")
    print("="*60)
    
    print("\n📁 Output files:")
    print(f"  - Results: {PROJECT_ROOT}/results.json")
    print(f"  - Figures: {PROJECT_ROOT}/paper/figs/")
    
    print("\n📈 To view figures:")
    print(f"  open {PROJECT_ROOT}/paper/figs/")


if __name__ == "__main__":
    main()
