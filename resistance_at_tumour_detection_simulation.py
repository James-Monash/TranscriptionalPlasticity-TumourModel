"""
Resistance at Tumour Detection Simulation

This script runs tumor simulations across multiple parameter sets with multiprocessing support.
For each parameter set, it runs simulations until 100 successful trials (tumors reaching 
4 billion cells) are collected. Extinct tumors are discarded and retried.

Parameter sets are loaded from a JSON file (parameter_sets.json) with the following format:
[
  {
    "INITIAL_SIZE": 100,
    "S": 0.01,
    "IDLE": 0.1,
    "R": 0,
    "M": 6.13e-07,
    "Q": 1.80e-07,
    "L": 9.0e-04
  },
  ...
]

Output:
- CSV file with results for each parameter set
- Master summary CSV with all results
- Timing logs
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool, cpu_count
from functools import partial
from tumour_simulation import TumourSimulation

# Simulation settings
GENERATIONS = 100000  # Large enough to reach 4 billion cells
TARGET_SUCCESSFUL_TRIALS = 100
OUTPUT_DIR = "./output"
PARAMETER_FILE = "parameter_sets.json"
NUM_WORKERS = cpu_count()  # Use all available CPU cores

def create_config(params, trial_number, config_filename):
    """
    Create a JSON configuration file for a simulation.
    
    Parameters:
    -----------
    params : dict
        Dictionary with parameter set (INITIAL_SIZE, S, IDLE, R, M, Q, L)
    trial_number : int
        Trial number for output naming
    config_filename : str
        Path to save the configuration file
    """
    config = {
        "simulation": {
            "generations": GENERATIONS,
            "initial_size": params['INITIAL_SIZE'],
            "output_dir": OUTPUT_DIR,
            "output_prefix": f"resist_detect_trial{trial_number}",
            "track_history": False
        },
        "biological_parameters": {
            "s": params['S'],
            "m": params['M'],
            "q": params['Q'],
            "r": params['R'],
            "l": params['L'],
            "idle": params['IDLE']
        },
        "treatment": {
            "schedule_type": "off",
            "drug_type": "abs",
            "treat_amt": 0.0,
            "pen_amt": 0.0,
            "dose_duration": 0,
            "penalty": False,
            "secondary_therapy": False
        }
    }
    
    with open(config_filename, 'w') as f:
        json.dump(config, f, indent=2)
    
    return config

def run_simulation_and_collect_resistance(args):
    """
    Run a single simulation and return the number of resistant cells at detection.
    This function is designed to work with multiprocessing.Pool.
    
    Parameters:
    -----------
    args : tuple
        Tuple of (params, trial_number, config_dir, param_set_id)
        
    Returns:
    --------
    dict or None
        Dictionary with simulation parameters and results if successful (tumor reached 4B cells),
        None if tumor went extinct
    """
    params, trial_number, config_dir, param_set_id = args
    
    try:
        # Create config file
        config_filename = Path(config_dir) / f"config_set{param_set_id}_trial{trial_number}.json"
        config = create_config(params, trial_number, str(config_filename))
        
        # Run simulation
        simulation = TumourSimulation(str(config_filename))
        state, results = simulation.run()
        
        # Check if tumor went extinct
        if state == 'extinct' or len(results) == 0:
            return None
        
        # Extract number of quasi-resistant cells
        n_quasi = int(results['Nq'].sum())
        n_total = int(results['N'].sum())
        n_sensitive = int(results['Ns'].sum())
        n_resistant = int(results['Nr'].sum())
        
        result = {
            'param_set_id': param_set_id,
            'trial_number': trial_number,
            'initial_size': params['INITIAL_SIZE'],
            's': params['S'],
            'idle': params['IDLE'],
            'r': params['R'],
            'm': params['M'],
            'q': params['Q'],
            'l': params['L'],
            'n_quasi': n_quasi,
            'n_total': n_total,
            'n_sensitive': n_sensitive,
            'n_resistant': n_resistant,
            'final_state': state,
            'quasi_fraction': n_quasi / n_total if n_total > 0 else 0
        }
        
        return result
        
    except Exception as e:
        print(f"  Trial {trial_number} ERROR: {e}")
        return None

def run_parameter_set(params, param_set_id, config_dir, output_dir):
    """
    Run simulations for a single parameter set until TARGET_SUCCESSFUL_TRIALS are collected.
    Uses multiprocessing to parallelize trials.
    
    Parameters:
    -----------
    params : dict
        Parameter set dictionary
    param_set_id : int
        ID of the parameter set
    config_dir : Path
        Directory to store config files
    output_dir : Path
        Directory to store output files
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with results from all successful trials
    dict
        Dictionary with timing and summary statistics
    """
    time_start = datetime.now()
    
    successful_results = []
    trial_number = 0
    total_attempts = 0
    
    # Create pool of workers
    with Pool(processes=NUM_WORKERS) as pool:
        while len(successful_results) < TARGET_SUCCESSFUL_TRIALS:
            # Generate batch of trial arguments
            # Run more trials than needed to account for extinctions
            batch_size = min(NUM_WORKERS * 2, TARGET_SUCCESSFUL_TRIALS - len(successful_results) + 10)
            trial_args = []
            
            for _ in range(batch_size):
                trial_number += 1
                total_attempts += 1
                trial_args.append((params, trial_number, config_dir, param_set_id))
            
            # Run trials in parallel
            results = pool.map(run_simulation_and_collect_resistance, trial_args)
            
            # Collect successful results
            for result in results:
                if result is not None:
                    successful_results.append(result)
                    if len(successful_results) >= TARGET_SUCCESSFUL_TRIALS:
                        break
    
    time_end = datetime.now()
    elapsed = time_end - time_start
    
    # Trim to exact number needed
    successful_results = successful_results[:TARGET_SUCCESSFUL_TRIALS]
    
    results_df = pd.DataFrame(successful_results)
    
    summary = {
        'param_set_id': param_set_id,
        'total_attempts': total_attempts,
        'successful_trials': len(successful_results),
        'extinct_tumors': total_attempts - len(successful_results),
        'extinction_rate': (total_attempts - len(successful_results)) / total_attempts,
        'elapsed_time': str(elapsed),
        'time_start': time_start.strftime('%Y-%m-%d %H:%M:%S'),
        'time_end': time_end.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    print(f"\nParameter set {param_set_id} complete:")
    print(f"  Total attempts: {total_attempts}")
    print(f"  Successful: {len(successful_results)}")
    print(f"  Extinct: {total_attempts - len(successful_results)}")
    print(f"  Elapsed time: {elapsed}")
    
    return results_df, summary


def load_parameter_sets(filename):
    """
    Load parameter sets from JSON file.
    
    Parameters:
    -----------
    filename : str
        Path to JSON file with parameter sets
        
    Returns:
    --------
    list
        List of parameter dictionaries
    """
    with open(filename, 'r') as f:
        param_sets = json.load(f)
    
    print(f"Loaded {len(param_sets)} parameter sets from {filename}")
    return param_sets


def main():
    """
    Main function to run simulations for all parameter sets.
    """
    print("="*80)
    print("RESISTANCE AT TUMOUR DETECTION SIMULATION - MULTI-PARAMETER")
    print("="*80)
    print(f"Target successful trials per parameter set: {TARGET_SUCCESSFUL_TRIALS}")
    print(f"Number of parallel workers: {NUM_WORKERS}")
    print(f"Parameter file: {PARAMETER_FILE}")
    print("="*80)
    
    # Load parameter sets
    try:
        parameter_sets = load_parameter_sets(PARAMETER_FILE)
    except FileNotFoundError:
        return
    
    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create config directory
    config_dir = Path("./configs_resistance_detection")
    config_dir.mkdir(parents=True, exist_ok=True)

    # Overall timing
    overall_start = datetime.now()
    
    # Store all results
    all_results = []
    all_summaries = []
    
    # Run simulations for each parameter set
    for param_set_id, params in enumerate(parameter_sets, start=1):
        results_df, summary = run_parameter_set(params, param_set_id, config_dir, output_dir)
        
        all_results.append(results_df)
        all_summaries.append(summary)
        
        # Save individual parameter set results (quasi_fraction only)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        param_results_file = output_dir / f"results_paramset{param_set_id}_{timestamp}.csv"
        results_df[['quasi_fraction']].to_csv(param_results_file, index=False)
    
    overall_end = datetime.now()
    overall_elapsed = overall_end - overall_start
    
    # Combine all results
    summaries_df = pd.DataFrame(all_summaries)
        
    # Save summary statistics
    summary_file = output_dir / f"summary_{timestamp}.csv"
    summaries_df.to_csv(summary_file, index=False)
    print(f"Summary statistics saved to: {summary_file}")
    
    # Save overall timing log
    timing_log_file = output_dir / f"overall_timing_{timestamp}.txt"
    with open(timing_log_file, 'w') as f:
        f.write(f"Overall simulation started: {overall_start.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Overall simulation ended:   {overall_end.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total elapsed time:         {str(overall_elapsed)}\n")
        f.write(f"Number of parameter sets:   {len(parameter_sets)}\n")
        f.write(f"Trials per parameter set:   {TARGET_SUCCESSFUL_TRIALS}\n")
        f.write(f"Number of workers:          {NUM_WORKERS}\n")
        f.write(f"\n{'='*80}\n")
        f.write("Per-parameter set summary:\n")
        f.write(f"{'='*80}\n")
        for summary in all_summaries:
            f.write(f"\nParameter Set {summary['param_set_id']}:\n")
            f.write(f"  Total attempts:    {summary['total_attempts']}\n")
            f.write(f"  Successful trials: {summary['successful_trials']}\n")
            f.write(f"  Extinct tumors:    {summary['extinct_tumors']}\n")
            f.write(f"  Extinction rate:   {summary['extinction_rate']*100:.1f}%\n")
            f.write(f"  Elapsed time:      {summary['elapsed_time']}\n")
    
    print(f"Overall timing log saved to: {timing_log_file}")
    
    # Print final summary
    print("\n" + "="*80)
    print("SIMULATION COMPLETE")
    print("="*80)
    print(f"Parameter sets processed: {len(parameter_sets)}")
    print(f"Total elapsed time: {overall_elapsed}")
    print("="*80)

if __name__ == "__main__":
    main()
