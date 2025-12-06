"""
Test script to compare sequential vs multiprocessing execution.
This helps diagnose the multiprocessing bug causing incorrect extinction rates.
"""

import time
from tumour_simulation import run_simulation_from_config

if __name__ == "__main__":
    # Path to your test configuration
    config_path = "example_configs/test.json"  # Update this to your actual test config path

    print("="*80)
    print("TESTING SEQUENTIAL EXECUTION")
    print("="*80)
    start_time = time.time()
    run_simulation_from_config(config_path, use_multiprocessing=False)
    sequential_time = time.time() - start_time
    print(f"\nSequential execution time: {sequential_time:.2f} seconds\n")

    print("\n" + "="*80)
    print("TESTING MULTIPROCESSING EXECUTION (FIXED)")
    print("="*80)
    start_time = time.time()
    run_simulation_from_config(config_path, use_multiprocessing=True)
    parallel_time = time.time() - start_time
    print(f"\nParallel execution time: {parallel_time:.2f} seconds")
    print(f"Speedup: {sequential_time/parallel_time:.2f}x\n")

