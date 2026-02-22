"""
Runtime: Store Module

Handles block I/O - reading blocks from disk.
v0.1: Whole block reads
v0.2: Will support range reads + sharding
"""

import os
import mmap
from pathlib import Path
from typing import Optional
import threading


class BlockStore:
    """
    Manages block file I/O.
    
    Optimizations:
    - Memory-mapped files for large blocks
    - Buffered reads
    - Optional prefetching
    """
    
    def __init__(self, root_dir: str, use_mmap: bool = True):
        self.root_dir = Path(root_dir)
        self.use_mmap = use_mmap
        self._mmap_cache: dict = {}
        self._lock = threading.Lock()
    
    def read_whole_block(self, rel_path: str) -> bytes:
        """
        Read an entire block file into memory.
        
        This is the v0.1 approach - simple but works well with sequential access.
        """
        full_path = self.root_dir / rel_path
        
        with open(full_path, "rb") as f:
            return f.read()
    
    def read_block_range(
        self, 
        rel_path: str, 
        offset: int, 
        size: int,
    ) -> bytes:
        """
        Read a specific range from a block file.
        
        Used for tensor-level access in v0.2+.
        """
        full_path = self.root_dir / rel_path
        
        with open(full_path, "rb") as f:
            f.seek(offset)
            return f.read(size)
    
    def get_block_path(self, rel_path: str) -> Path:
        """Get full path to a block file."""
        return self.root_dir / rel_path
    
    def get_block_size(self, rel_path: str) -> int:
        """Get the size of a block file in bytes."""
        full_path = self.root_dir / rel_path
        return full_path.stat().st_size
    
    def prefetch_block(self, rel_path: str) -> bytes:
        """
        Prefetch a block into memory.
        
        On Linux this can use readahead(); on macOS the OS handles caching.
        """
        full_path = self.root_dir / rel_path
        size = full_path.stat().st_size
        
        # Open and read - OS will cache this
        with open(full_path, "rb") as f:
            return f.read()
    
    def exists(self, rel_path: str) -> bool:
        """Check if a block file exists."""
        return (self.root_dir / rel_path).exists()


class BufferedBlockStore(BlockStore):
    """
    Block store with read-ahead buffering.
    
    Useful when accessing blocks sequentially.
    """
    
    def __init__(self, root_dir: str, buffer_size: int = 4 * 1024 * 1024):
        super().__init__(root_dir)
        self.buffer_size = buffer_size
        self._buffer: Optional[bytes] = None
        self._buffer_path: Optional[str] = None
    
    def read_whole_block(self, rel_path: str) -> bytes:
        """Read with buffering for sequential access."""
        # Check buffer first
        if self._buffer_path == rel_path and self._buffer is not None:
            return self._buffer
        
        # Read and cache
        data = super().read_whole_block(rel_path)
        
        # Only cache if small enough
        if len(data) <= self.buffer_size * 2:
            self._buffer = data
            self._buffer_path = rel_path
        
        return data
    
    def clear_buffer(self):
        """Clear the read buffer."""
        self._buffer = None
        self._buffer_path = None


if __name__ == "__main__":
    # Quick test
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python store.py <root_dir> <block_rel_path>")
        sys.exit(1)
    
    store = BlockStore(sys.argv[1])
    block_path = sys.argv[2]
    
    print(f"Block: {block_path}")
    print(f"Size: {store.get_block_size(block_path) / 1e6:.1f} MB")
    
    import time
    t0 = time.perf_counter()
    data = store.read_whole_block(block_path)
    t1 = time.perf_counter()
    
    print(f"Read time: {(t1-t0)*1000:.1f} ms")
    print(f"Throughput: {len(data) / (t1-t0) / 1e9:.2f} GB/s")
