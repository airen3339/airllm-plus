"""
Runtime: Cache Module

GPU-side caching for loaded blocks.
v0.1: Simple LRU cache for blocks
v0.2: Will support tensor-level LRU + shard-level caching
"""

import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, Any, Dict
import threading


@dataclass
class CacheEntry:
    """An entry in the block cache."""
    block_id: int
    data: Any  # Could be bytes (CPU) or tensor (GPU)
    size_bytes: int
    timestamp: float
    access_count: int = 1


class BlockCache:
    """
    LRU cache for blocks.
    
    In v0.1, this is a simple CPU-side cache.
    In v0.2, this will manage GPU memory with proper eviction.
    """
    
    def __init__(
        self, 
        max_size_mb: float = 2048,  # 2GB default
        max_blocks: int = 4,
    ):
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.max_blocks = max_blocks
        self._cache: OrderedDict[int, CacheEntry] = OrderedDict()
        self._lock = threading.Lock()
        
        # Stats
        self._hits = 0
        self._misses = 0
        self._evictions = 0
    
    def get(self, block_id: int) -> Optional[Any]:
        """Get a block from cache."""
        with self._lock:
            if block_id in self._cache:
                # Move to end (most recently used)
                entry = self._cache.pop(block_id)
                entry.access_count += 1
                self._cache[block_id] = entry
                self._hits += 1
                return entry.data
            else:
                self._misses += 1
                return None
    
    def put(self, block_id: int, data: Any, size_bytes: int) -> None:
        """
        Put a block into cache.
        
        May trigger eviction if cache is full.
        """
        with self._lock:
            # Don't cache if too large
            if size_bytes > self.max_size_bytes:
                return
            
            # Remove existing entry if present
            if block_id in self._cache:
                del self._cache[block_id]
            
            # Evict if necessary
            while (len(self._cache) >= self.max_blocks or 
                   self._current_size() + size_bytes > self.max_size_bytes):
                if not self._cache:
                    break
                self._evict_oldest()
            
            # Add new entry
            entry = CacheEntry(
                block_id=block_id,
                data=data,
                size_bytes=size_bytes,
                timestamp=time.time(),
            )
            self._cache[block_id] = entry
    
    def _current_size(self) -> int:
        """Calculate current cache size in bytes."""
        return sum(e.size_bytes for e in self._cache.values())
    
    def _evict_oldest(self) -> None:
        """Evict the oldest (least recently used) entry."""
        if self._cache:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            self._evictions += 1
    
    def clear(self) -> None:
        """Clear all cached blocks."""
        with self._lock:
            self._cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0
            
            return {
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "evictions": self._evictions,
                "current_blocks": len(self._cache),
                "current_size_mb": self._current_size() / 1e6,
                "max_size_mb": self.max_size_bytes / 1e6,
            }


class GPUBlockCache(BlockCache):
    """
    GPU-aware block cache.
    
    v0.2 feature - requires PyTorch/CUDA.
    """
    
    def __init__(
        self,
        max_size_mb: float = 4096,  # 4GB GPU memory
        max_blocks: int = 4,
    ):
        super().__init__(max_size_mb, max_blocks)
        self._gpu_cache: Dict[int, Any] = {}  # block_id -> GPU tensor
        # TODO: Integrate with PyTorch CUDA memory management
    
    def put_to_gpu(self, block_id: int, data: Any, size_bytes: int) -> None:
        """
        Put a block directly onto GPU.
        
        v0.2: Will handle pinned memory + cudaMemcpyAsync
        """
        raise NotImplementedError("GPUBlockCache v0.2 feature")


class ShardCache:
    """
    Cache for individual tensor shards within a block.
    
    Enables finer-grained caching - only keep frequently accessed tensors.
    v0.2 feature.
    """
    
    def __init__(self, max_shards: int = 16):
        self.max_shards = max_shards
        self._cache: OrderedDict = OrderedDict()
        # TODO: Implement shard-level LRU
    
    def get(self, tensor_name: str) -> Optional[Any]:
        """Get a specific tensor from cache."""
        raise NotImplementedError("ShardCache v0.2 feature")
    
    def put(self, tensor_name: str, data: Any) -> None:
        """Cache a specific tensor."""
        raise NotImplementedError("ShardCache v0.2 feature")


if __name__ == "__main__":
    # Test the cache
    cache = BlockCache(max_size_mb=100, max_blocks=3)
    
    # Put some blocks
    for i in range(5):
        data = b"x" * (10 * 1024 * 1024)  # 10MB each
        cache.put(i, data, len(data))
        print(f"Put block {i}, stats: {cache.get_stats()['current_blocks']} blocks")
    
    # Get some blocks
    for i in [0, 2, 4]:
        result = cache.get(i)
        if result:
            print(f"Hit block {i}")
        else:
            print(f"Miss block {i}")
    
    print(f"\nFinal stats: {cache.get_stats()}")
