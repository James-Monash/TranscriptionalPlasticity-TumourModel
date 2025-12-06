"""
Quick test to verify RNG initialization in multiprocessing workers.
"""
import numpy as np
from multiprocessing import Pool
import os

_rng = None

def _get_rng():
    global _rng
    if _rng is None:
        _rng = np.random.default_rng()
    return _rng

def _init_worker():
    """Initialize worker process with fresh RNG using process-safe seeding."""
    global _rng
    import os
    pid = os.getpid()
    _rng = np.random.default_rng(np.random.SeedSequence(entropy=pid))
    print(f"Worker PID {pid} initialized with RNG")

def test_random(task_id):
    """Generate a random number in worker."""
    rng = _get_rng()
    rand_val = rng.random()
    pid = os.getpid()
    print(f"Task {task_id} in PID {pid}: {rand_val}")
    return (task_id, pid, rand_val)

if __name__ == "__main__":
    print("Testing multiprocessing RNG initialization:")
    print("="*60)
    
    with Pool(processes=4, initializer=_init_worker) as pool:
        results = pool.map(test_random, range(10))
    
    print("="*60)
    print("\nResults:")
    for task_id, pid, rand_val in results:
        print(f"Task {task_id}: PID {pid}, Random: {rand_val:.6f}")
    
    # Check for duplicates
    rand_vals = [r[2] for r in results]
    if len(rand_vals) != len(set(rand_vals)):
        print("\n⚠️  WARNING: Duplicate random values detected!")
    else:
        print("\n✓ All random values are unique")
