"""
Debug script to trace simulation execution step by step.

This script runs a simulation with detailed logging to help identify
why the tumor might be going extinct unexpectedly.
"""

import json
import numpy as np
import random
from pathlib import Path

from tumour_simulation import TumourSimulation
from treatment import Treatment


def run_debug_simulation(config_path: str, verbose_generations: int = 20):
    """
    Run simulation with detailed debugging output.
    
    Parameters:
    -----------
    config_path : str
        Path to config JSON file
    verbose_generations : int
        Number of generations to print detailed info for
    """
    # Set seeds for reproducibility during debugging
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    print(f"Random seed set to: {seed}")
    
    # Load config to inspect it
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print("\n" + "=" * 60)
    print("CONFIGURATION")
    print("=" * 60)
    print(json.dumps(config, indent=2))
    
    # Create simulation
    sim = TumourSimulation(config_path)
    
    print("\n" + "=" * 60)
    print("INITIAL STATE")
    print("=" * 60)
    print(f"Initial tumor size: {sim.clones.get_total_cells()}")
    print(f"Number of clones: {len(sim.clones)}")
    for clone in sim.clones:
        print(f"  Clone {clone.clone_id}: S={clone.n_sensitive}, Q={clone.n_quasi}, R={clone.n_resistant}")
    
    # Check base probabilities
    print("\n" + "=" * 60)
    print("BASE PROBABILITIES (k=0, no treatment)")
    print("=" * 60)
    probs = sim.treatment.get_base_probabilities(k=0)
    for key, value in probs.items():
        print(f"  {key}: {value:.6f}")
    
    print(f"\nProbability sum check: {sum(probs.values()):.6f}")
    
    # Calculate expected growth rate
    prob_birth = probs['prob_birth']
    prob_death = probs['prob_death']
    expected_growth = prob_birth - prob_death
    print(f"\nExpected net growth per cell per generation:")
    print(f"  prob_birth - prob_death = {prob_birth:.6f} - {prob_death:.6f} = {expected_growth:.6f}")
    
    if expected_growth < 0:
        print("  WARNING: Negative expected growth! Tumor will likely shrink on average.")
    elif expected_growth < 0.01:
        print("  WARNING: Very low expected growth. With small initial size, extinction is likely.")
    
    print("\n" + "=" * 60)
    print(f"RUNNING SIMULATION (detailed logging for first {verbose_generations} generations)")
    print("=" * 60)
    
    # Run simulation manually with logging
    for generation in range(min(sim.generations, verbose_generations)):
        sim.current_generation = generation
        
        # Get state before simulation step
        counts_before = sim.clones.get_total_by_type()
        
        # Remove extinct clones
        sim.clones.remove_extinct_clones()
        
        # Check if tumor went extinct
        tumor_size = sim.clones.get_total_cells()
        if tumor_size == 0:
            sim.state = 1
            print(f"\n*** TUMOR EXTINCT at generation {generation} ***")
            break
        
        # Update treatment state
        sim.treatment.update_treatment_state(tumor_size, generation)
        
        # Log before simulation step
        print(f"\nGeneration {generation}:")
        print(f"  Before: Total={counts_before['total']}, S={counts_before['sensitive']}, "
              f"Q={counts_before['quasi']}, R={counts_before['resistant']}")
        
        # Get probabilities being used
        probs_s = sim.treatment.calculate_probabilities(0, "S")
        print(f"  Probabilities (S cells): birth={probs_s['prob_birth']:.4f}, "
              f"death={probs_s['prob_death']:.4f}, idle={probs_s['prob_idle']:.4f}")
        
        # Simulate one generation
        sim._simulate_generation()
        
        # Log after simulation step
        counts_after = sim.clones.get_total_by_type()
        print(f"  After:  Total={counts_after['total']}, S={counts_after['sensitive']}, "
              f"Q={counts_after['quasi']}, R={counts_after['resistant']}")
        
        # Check for extinction
        if counts_after['total'] == 0:
            print(f"\n*** TUMOR WENT EXTINCT during generation {generation} ***")
            sim.state = 1
            break
    
    # If still running after verbose generations, continue silently
    if sim.state == 0 and sim.current_generation < sim.generations - 1:
        print(f"\n... continuing simulation (logging every 100 generations) ...")
        
        for generation in range(verbose_generations, sim.generations):
            sim.current_generation = generation
            sim.clones.remove_extinct_clones()
            
            tumor_size = sim.clones.get_total_cells()
            if tumor_size == 0:
                sim.state = 1
                print(f"\n*** TUMOR EXTINCT at generation {generation} ***")
                break
            
            sim.treatment.update_treatment_state(tumor_size, generation)
            sim._simulate_generation()
            
            if generation % 100 == 0:
                counts = sim.clones.get_total_by_type()
                print(f"Gen {generation}: Total={counts['total']:.2e}")
    
    print("\n" + "=" * 60)
    print("FINAL STATE")
    print("=" * 60)
    counts = sim.clones.get_total_by_type()
    print(f"Final generation: {sim.current_generation}")
    print(f"Final state: {sim.state} (0=timeout, 1=extinct, 2=relapsed)")
    print(f"Final tumor size: {counts['total']}")
    print(f"  Sensitive: {counts['sensitive']}")
    print(f"  Quasi-resistant: {counts['quasi']}")
    print(f"  Resistant: {counts['resistant']}")
    print(f"Number of clones: {len(sim.clones)}")
    
    return sim


def analyze_extinction_probability(config_path: str, n_runs: int = 100):
    """
    Run multiple simulations to estimate extinction probability.
    
    Parameters:
    -----------
    config_path : str
        Path to config JSON file
    n_runs : int
        Number of simulations to run
    """
    print("\n" + "=" * 60)
    print(f"EXTINCTION ANALYSIS ({n_runs} runs)")
    print("=" * 60)
    
    extinctions = 0
    final_sizes = []
    extinction_generations = []
    
    for i in range(n_runs):
        # Different seed each run
        random.seed(i)
        np.random.seed(i)
        
        sim = TumourSimulation(config_path)
        
        # Run until extinction or 1000 generations
        max_gens = min(sim.generations, 1000)
        for generation in range(max_gens):
            sim.current_generation = generation
            sim.clones.remove_extinct_clones()
            
            tumor_size = sim.clones.get_total_cells()
            if tumor_size == 0:
                extinctions += 1
                extinction_generations.append(generation)
                break
            
            sim.treatment.update_treatment_state(tumor_size, generation)
            sim._simulate_generation()
        
        final_sizes.append(sim.clones.get_total_cells())
        
        if (i + 1) % 10 == 0:
            print(f"  Completed {i + 1}/{n_runs} runs...")
    
    print(f"\nResults:")
    print(f"  Extinction rate: {extinctions}/{n_runs} ({100*extinctions/n_runs:.1f}%)")
    if extinction_generations:
        print(f"  Mean extinction generation: {np.mean(extinction_generations):.1f}")
        print(f"  Median extinction generation: {np.median(extinction_generations):.1f}")
    
    surviving_sizes = [s for s in final_sizes if s > 0]
    if surviving_sizes:
        print(f"  Mean final size (survivors): {np.mean(surviving_sizes):.2e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        config_path = "./example_configs/no_treatment.json"
        print(f"No config specified, using: {config_path}")
    else:
        config_path = sys.argv[1]
    
    # Run detailed debug
    run_debug_simulation(config_path, verbose_generations=30)
    
    # Optionally run extinction analysis
    print("\n" + "=" * 60)
    response = input("Run extinction probability analysis? (y/n): ")
    if response.lower() == 'y':
        analyze_extinction_probability(config_path, n_runs=50)
