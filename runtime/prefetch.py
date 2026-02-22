"""
Runtime: Prefetch Module

Background prefetching to overlap I/O with computation.

Key insight: While the main thread is computing on block N,
the prefetch thread should be loading block N+1, N+2, ...
"""

import queue
import threading
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Callable, Any
from .store import BlockStore
from .plan import Block


@dataclass
class PrefetchedBlock:
    """A block that has been prefetched into memory."""
    block: Block
    data: bytes
    timestamp: float  # When it was fetched


class Prefetcher:
    """
    Background prefetch worker.
    
    Uses a simple producer-consumer pattern:
    - Main thread: submits block IDs to prefetch queue
    - Background thread: reads blocks and puts in output queue
    - Main thread: gets completed blocks from output queue
    """
    
    def __init__(
        self, 
        store: BlockStore, 
        depth: int = 2,
        priority: bool = False,
    ):
        self.store = store
        self.depth = depth
        self.priority = priority  # If True, newest blocks first
        
        # Queues
        self._submit_queue: queue.Queue = queue.Queue(maxsize=depth)
        self._result_queue: queue.Queue = queue.Queue(maxsize=depth)
        
        # Control
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
        self._started = False
        
        # Stats
        self._prefetch_count = 0
        self._prefetch_time = 0.0
        self._miss_count = 0
    
    def start(self):
        """Start the prefetch worker thread."""
        if self._started:
            return
        
        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="AirLLM-Prefetch"
        )
        self._worker_thread.start()
        self._started = True
    
    def stop(self, timeout: float = 2.0):
        """Stop the prefetch worker."""
        if not self._started:
            return
        
        self._stop_event.set()
        if self._worker_thread:
            self._worker_thread.join(timeout=timeout)
        self._started = False
    
    def submit(self, block: Block):
        """
        Submit a block for prefetching.
        
        Non-blocking - returns immediately if queue is full.
        """
        try:
            self._submit_queue.put_nowait(block)
        except queue.Full:
            # Queue full - will need to wait
            self._submit_queue.put(block)
    
    def get(self, timeout: Optional[float] = None) -> PrefetchedBlock:
        """
        Get a prefetched block.
        
        Blocks until a block is available.
        """
        return self._result_queue.get(timeout=timeout)
    
    def _worker_loop(self):
        """Main prefetch worker loop."""
        pending_blocks: dict = {}  # block_id -> block
        
        while not self._stop_event.is_set():
            # Try to submit pending blocks first
            while not self._submit_queue.empty():
                try:
                    block = self._submit_queue.get_nowait()
                    pending_blocks[block.block_id] = block
                except queue.Empty:
                    break
            
            # Process pending blocks
            if pending_blocks:
                # Get next block to prefetch
                if self.priority:
                    # Newest first
                    next_id = max(pending_blocks.keys())
                else:
                    # Oldest first (sequential)
                    next_id = min(pending_blocks.keys())
                
                block = pending_blocks.pop(next_id)
                
                # Actually read the block
                t0 = time.perf_counter()
                try:
                    data = self.store.read_whole_block(block.file)
                    t1 = time.perf_counter()
                    
                    self._prefetch_count += 1
                    self._prefetch_time += (t1 - t0)
                    
                    prefetched = PrefetchedBlock(
                        block=block,
                        data=data,
                        timestamp=t1,
                    )
                    
                    # Try to put result (non-blocking)
                    try:
                        self._result_queue.put_nowait(prefetched)
                    except queue.Full:
                        # Result queue full - main thread not keeping up
                        self._miss_count += 1
                        
                except Exception as e:
                    print(f"Prefetch error for block {block.block_id}: {e}")
            else:
                # Nothing to do - sleep briefly
                time.sleep(0.01)
        
        # Drain queues on shutdown
        while not self._submit_queue.empty():
            try:
                self._submit_queue.get_nowait()
            except queue.Empty:
                break
    
    def get_stats(self) -> dict:
        """Get prefetch statistics."""
        return {
            "prefetch_count": self._prefetch_count,
            "prefetch_time_s": self._prefetch_time,
            "avg_prefetch_ms": (self._prefetch_time / self._prefetch_count * 1000 
                                if self._prefetch_count > 0 else 0),
            "miss_count": self._miss_count,
            "queue_depth": self.depth,
        }


class AsyncPrefetcher:
    """
    AsyncIO-based prefetcher for better concurrency.
    
    v0.2 feature - not implemented yet.
    """
    
    def __init__(self, store: BlockStore, depth: int = 2):
        self.store = store
        self.depth = depth
        # TODO: Implement with asyncio
        raise NotImplementedError("AsyncPrefetcher v0.2 feature")


if __name__ == "__main__":
    # Simple test
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python prefetch.py <index.json>")
        sys.exit(1)
    
    from .plan import load_plan
    
    index_path = sys.argv[1]
    blocks = load_plan(index_path)
    
    store = BlockStore(str(Path(index_path).parent))
    pf = Prefetcher(store, depth=2)
    pf.start()
    
    # Submit first few blocks
    for i in range(min(3, len(blocks))):
        pf.submit(blocks[i])
    
    # Get them back
    for i in range(min(3, len(blocks))):
        prefetched = pf.get(timeout=5)
        print(f"Got block {prefetched.block.block_id}: {len(prefetched.data) / 1e6:.1f} MB")
    
    pf.stop()
    
    stats = pf.get_stats()
    print(f"\nStats: {stats}")
