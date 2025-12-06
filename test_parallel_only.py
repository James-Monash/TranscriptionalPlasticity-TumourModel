"""
Test just the multiprocessing version with debug output.
"""

from tumour_simulation import run_simulation_from_config

if __name__ == "__main__":
    config_path = "example_configs/test.json"
    print("Running multiprocessing version with debug output:")
    print("="*80)
    run_simulation_from_config(config_path, use_multiprocessing=True)
