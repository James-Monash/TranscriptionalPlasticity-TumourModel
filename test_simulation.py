"""
Test script to demonstrate and validate the tumor simulation framework.

This script runs simple test cases to verify the simulation works correctly.
"""

import json
import numpy as np
import random
from pathlib import Path

from tumour_simulation import TumourSimulation


def test_basic_simulation():
    """Test a basic simulation with no treatment."""
    print("=" * 60)
    print("TEST 1: Basic simulation (no treatment)")
    print("=" * 60)
    
    # Set seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Create a minimal config
    config = {
        "simulation": {
            "generations": 100,
            "initial_size": 100,
            "output_dir": "./test_output",
            "output_prefix": "test_basic",
            "track_history": False
        },
        "biological_parameters": {
            "s": 0.05,
            "u": 0.000001,
            "pq": 0.000001,
            "pr": 0.000001,
            "ps": 0.000001,
            "idle": 0.1
        },
        "treatment": {
            "schedule_type": "off",
            "drug_type": "abs",
            "treat_amt": 0.8,
            "pen_amt": 4.0,
            "dose_duration": 24,
            "penalty": False,
            "secondary_therapy": False
        }
    }
    
    # Save config
    config_path = "./test_config_basic.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Run simulation
    simulation = TumourSimulation(config_path)
    state, results = simulation.run()
    
    # Print results
    summary = simulation.get_summary()
    print(f"\nResults:")
    print(f"  Final state: {state}")
    print(f"  Total: {summary['total_cells']}")
    print(f"  Sensitive: {summary['sensitive_cells']}")
    print(f"  Transient: {summary['transient_cells']}")
    print(f"  Resistant: {summary['resistant_cells']}")
    print(f"  Number of clones: {summary['n_clones']}")
    
    # Cleanup
    Path(config_path).unlink()
    
    return state, results


def test_treatment_simulation():
    """Test simulation with MTD treatment."""
    print("\n" + "=" * 60)
    print("TEST 2: MTD treatment simulation")
    print("=" * 60)
    
    # Set seeds for reproducibility
    random.seed(123)
    np.random.seed(123)
    
    config = {
        "simulation": {
            "generations": 500,
            "initial_size": 1,
            "output_dir": "./test_output",
            "output_prefix": "test_mtd",
            "track_history": True
        },
        "biological_parameters": {
            "s": 0.01,
            "u": 0.0000005,
            "pq": 0.0000005,
            "pr": 0.0000005,
            "ps": 0.0000005,
            "idle": 0.1
        },
        "treatment": {
            "schedule_type": "mtd",
            "drug_type": "abs",
            "treat_amt": 0.8,
            "pen_amt": 4.0,
            "dose_duration": 24,
            "treatment_start_size": 100000,
            "relapse_size": 1000000,
            "penalty": False,
            "secondary_therapy": False
        }
    }
    
    # Save config
    config_path = "./test_config_mtd.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Run simulation
    simulation = TumourSimulation(config_path)
    state, results = simulation.run()
    
    # Print results
    summary = simulation.get_summary()
    treatment_state = simulation.treatment.get_state_summary()
    
    print(f"\nResults:")
    print(f"  Final state: {state}")
    print(f"  Total cells: {summary['total_cells']}")
    print(f"  Sensitive: {summary['sensitive_cells']}")
    print(f"  Quasi: {summary['quasi_cells']}")
    print(f"  Resistant: {summary['resistant_cells']}")
    print(f"  Number of clones: {summary['n_clones']}")
    print(f"  Doses given: {treatment_state['doses_given']}")
    print(f"  Treatment active: {treatment_state['treatment_active']}")
    
    # Cleanup
    Path(config_path).unlink()
    
    return state, results


def test_adaptive_therapy():
    """Test simulation with adaptive therapy."""
    print("\n" + "=" * 60)
    print("TEST 3: Adaptive therapy simulation")
    print("=" * 60)
    
    # Set seeds for reproducibility
    random.seed(456)
    np.random.seed(456)
    
    config = {
        "simulation": {
            "generations": 500,
            "initial_size": 1,
            "output_dir": "./test_output",
            "output_prefix": "test_adaptive",
            "track_history": True
        },
        "biological_parameters": {
            "s": 0.01,
            "u": 0.0000005,
            "pq": 0.0000005,
            "pr": 0.0000005,
            "ps": 0.0000005,
            "idle": 0.1
        },
        "treatment": {
            "schedule_type": "adaptive",
            "drug_type": "abs",
            "treat_amt": 0.8,
            "pen_amt": 4.0,
            "dose_duration": 24,
            "treatment_start_size": 100000,
            "treatment_stop_size": 50000,
            "relapse_size": 1000000,
            "penalty": False,
            "secondary_therapy": False
        }
    }
    
    # Save config
    config_path = "./test_config_adaptive.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Run simulation
    simulation = TumourSimulation(config_path)
    state, results = simulation.run()
    
    # Print results
    summary = simulation.get_summary()
    treatment_state = simulation.treatment.get_state_summary()
    
    print(f"\nResults:")
    print(f"  Final state: {state}")
    print(f"  Total cells: {summary['total_cells']}")
    print(f"  Sensitive: {summary['sensitive_cells']}")
    print(f"  Quasi: {summary['quasi_cells']}")
    print(f"  Resistant: {summary['resistant_cells']}")
    print(f"  Number of clones: {summary['n_clones']}")
    print(f"  Doses given: {treatment_state['doses_given']}")
    print(f"  Treatment active: {treatment_state['treatment_active']}")
    
    # Cleanup
    Path(config_path).unlink()
    
    return state, results


def test_clone_class():
    """Test Clone class functionality."""
    print("\n" + "=" * 60)
    print("TEST 4: Clone class")
    print("=" * 60)
    
    from clone import Clone, CloneCollection
    
    # Create a clone
    clone = Clone(
        clone_id=0,
        parent_id=-1,
        parent_type="Sens",
        generation=0,
        n_sensitive=100,
        n_transient=10,
        n_resistant=5,
    )
    
    print(f"Initial clone: {clone}")
    print(f"  Total cells: {clone.total_cells}")
    print(f"  Is extinct: {clone.is_extinct()}")
    
    # Update counts
    clone.update_counts(delta_sensitive=50, delta_transient=-5, delta_resistant=10)
    print(f"\nAfter update: {clone}")
    print(f"  Total cells: {clone.total_cells}")
    
    # Test CloneCollection
    collection = CloneCollection()
    c1 = collection.add_clone(-1, "Sens", 0, n_sensitive=100)
    c2 = collection.add_clone(0, "S", 5, n_sensitive=50, n_transient=10)
    
    print(f"\nClone collection: {collection}")
    print(f"  Total cells: {collection.get_total_cells()}")
    print(f"  By type: {collection.get_total_by_type()}")
    
    # Remove extinct
    c1.n_sensitive = 0
    collection.remove_extinct_clones()
    print(f"\nAfter removing extinct: {collection}")


def test_treatment_class():
    """Test Treatment class functionality."""
    print("\n" + "=" * 60)
    print("TEST 5: Treatment class")
    print("=" * 60)
    
    from treatment import Treatment
    
    config = {
        "schedule_type": "mtd",
        "drug_type": "abs",
        "treat_amt": 0.8,
        "pen_amt": 4.0,
        "dose_duration": 24,
        "treatment_start_size": 1000,
        "relapse_size": 10000,
        "penalty": False,
        "secondary_therapy": True,
        "secondary_therapy_type": "plast"
    }
    
    base_params = {
        's': 0.01,
        'u': 0.0000005,
        'pq': 0.0000005,
        'pr': 0.0000005,
        'ps': 0.0000005,
        'idle': 0.1
    }
    
    treatment = Treatment(config, base_params)
    
    # Test probability calculation
    probs_s = treatment.calculate_probabilities(k=1, cell_type="S")
    print(f"Sensitive cell probabilities (no treatment):")
    print(f"  prob_idle: {probs_s['prob_idle']:.6f}")
    print(f"  prob_birth: {probs_s['prob_birth']:.6f}")
    print(f"  prob_death: {probs_s['prob_death']:.6f}")
    
    # Activate treatment
    treatment.update_treatment_state(tumor_size=5000, generation=10)
    probs_s_treated = treatment.calculate_probabilities(k=1, cell_type="S")
    print(f"\nSensitive cell probabilities (with treatment):")
    print(f"  prob_idle: {probs_s_treated['prob_idle']:.6f}")
    print(f"  prob_birth: {probs_s_treated['prob_birth']:.6f}")
    print(f"  prob_death: {probs_s_treated['prob_death']:.6f}")
    
    # Test concentration
    conc1, conc2 = treatment.get_drug_concentration()
    print(f"\nDrug concentrations: primary={conc1:.3f}, secondary={conc2:.3f}")
    
    # Advance a few iterations
    for i in range(24):
        treatment.update_treatment_state(tumor_size=5000, generation=11+i)
    
    conc1, conc2 = treatment.get_drug_concentration()
    print(f"After 24 iterations: primary={conc1:.3f}, secondary={conc2:.3f}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TUMOR SIMULATION TEST SUITE")
    print("=" * 60)
    
    # Create output directory
    Path("./test_output").mkdir(exist_ok=True)
    
    # Run tests
    try:
        test_clone_class()
        test_treatment_class()
        test_basic_simulation()
        test_treatment_simulation()
        test_adaptive_therapy()
        
        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("\nCheck ./test_output/ for generated files")
        
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"TEST FAILED: {e}")
        print(f"{'='*60}")
        import traceback
        traceback.print_exc()
