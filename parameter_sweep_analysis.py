"""
Parameter Sweep Analysis Script

Generates config files for systematic exploration of SPR and SPQ parameter space,
runs simulations, and creates heatmap visualizations showing resistance outcomes.

SPR = log(s/Pr) - ratio of selective advantage to permanent resistance acquisition
SPQ = log(s/Pq) - ratio of selective advantage to transient resistance acquisition
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from tumour_simulation import run_simulation_from_config
import os
import logging
import sys


def calculate_probabilities(s, spr, x_value, x_mode='SPQ'):
    """
    Calculate Pq and Pr from s, SPR, and either SPQ or PQR.
    
    Parameters:
    -----------
    s : float
        Selective advantage
    spr : float
        SPR = log(s/Pr)
    x_value : float
        Either SPQ = log(s/Pq) or PQR = log(Pq/Pr)
    x_mode : str
        Either 'SPQ' or 'PQR' to indicate which mode to use
    
    Returns:
    --------
    tuple of (float, float, float)
        (Pq, Pr, x_actual) - transient resistance, permanent resistance, and actual x-axis value
    """
    # First calculate Pr from SPR
    # SPR = log(s/Pr) => Pr = s / exp(SPR)
    pr = s / np.exp(spr)
    
    if x_mode == 'SPQ':
        # SPQ = log(s/Pq) => Pq = s / exp(SPQ)
        pq = s / np.exp(x_value)
        x_actual = x_value  # SPQ value
    else:  # PQR mode
        # PQR = log(Pq/Pr) => Pq = Pr * exp(PQR)
        pq = pr * np.exp(x_value)
        x_actual = x_value  # PQR value
    
    return pq, pr, x_actual


def is_realistic(pq, pr, min_prob=1e-11, max_prob=1e-2):
    """
    Check if Pq and Pr are within realistic bounds.
    
    Parameters:
    -----------
    pq : float
        Transient resistance probability
    pr : float
        Permanent resistance probability
    min_prob : float
        Minimum realistic probability
    max_prob : float
        Maximum realistic probability
    
    Returns:
    --------
    bool
        True if both probabilities are realistic
    """
    return (min_prob <= pq <= max_prob) and (min_prob <= pr <= max_prob)


def calculate_m_from_mutation_probability(target_mutation_prob, s, idle=0.1):
    """
    Calculate m parameter to achieve target mutation probability.
    
    The mutation probability is Pm = m*(1-Pd).
    For base case (k=1): Pd = ((1-idle)/2) * (1-s)^k
    We want m*(1-Pd) = target_mutation_prob
    
    Parameters:
    -----------
    target_mutation_prob : float
        Target probability for driver mutation (Pm)
    s : float
        Selective advantage
    idle : float
        Idle/quiescence probability
    
    Returns:
    --------
    float
        m parameter value
    """
    # For k=1: Pd = ((1-idle)/2) * (1-s)^1
    prob_death = ((1 - idle) / 2) * (1 - s)
    
    # Pm = m*(1-Pd) => m = Pm / (1-Pd)
    m = target_mutation_prob / (1 - prob_death)
    
    return m


def calculate_biological_parameters_from_probabilities(target_prob_q, target_prob_r, target_prob_l, s, idle=0.1):
    """
    Calculate biological parameters (q, r, l) from target probabilities.
    
    The actual probabilities are calculated as:
    - prob_to_transient = q * (1 - prob_death)
    - prob_to_resistant = r * (1 - prob_death)  
    - prob_to_sensitive = l * (1 - prob_death)
    
    Where prob_death = ((1-idle)/2) * (1-s)^k for k=1
    
    To achieve target probabilities, we solve for q, r, l:
    - q = target_prob_q / (1 - prob_death)
    - r = target_prob_r / (1 - prob_death)
    - l = target_prob_l / (1 - prob_death)
    
    Parameters:
    -----------
    target_prob_q : float
        Target probability for transient resistance gain (Pq)
    target_prob_r : float
        Target probability for permanent resistance gain (Pr)
    target_prob_l : float
        Target probability for transient resistance loss (Ps/Pl)
    s : float
        Selective advantage
    idle : float
        Idle/quiescence probability
    
    Returns:
    --------
    tuple of (float, float, float)
        (q, r, l) biological parameters
    """
    # For k=1: Pd = ((1-idle)/2) * (1-s)
    prob_death = ((1 - idle) / 2) * (1 - s)
    one_minus_prob_death = 1 - prob_death
    
    # Calculate biological parameters
    q = target_prob_q / one_minus_prob_death
    r = target_prob_r / one_minus_prob_death
    l = target_prob_l / one_minus_prob_death
    
    return q, r, l

def generate_multi_condition_config(s_value, realistic_combinations, m_value, output_dir, penalty=False, x_mode='SPQ', kill_mechanism=None):
    """
    Generate a single multi-condition   file with all parameter combinations.
    
    Parameters:
    -----------
    s_value : float
        Selective advantage
    realistic_combinations : list
        List of (spr, x_value, pq, pr) tuples
    m_value : float
        Driver mutation probability parameter
    output_dir : Path
        Directory for output files
    penalty : bool, optional
        Whether to apply penalty to resistance (default: False)
    x_mode : str, optional
        Either 'SPQ' or 'PQR' mode (default: 'SPQ')
    kill_mechanism : str, optional
        Either 'kbe' or 'kod' for treatment mechanism, or None for no treatment (default: None)
    
    Returns:
    --------
    str or None
        Path to generated config file, or None if file already exists
    """
    kill_suffix = f"_{kill_mechanism}" if kill_mechanism else ""
    config_path = output_dir / f"s{s_value}_{x_mode}_multi_condition_config_penalty_{'on' if penalty else 'off'}{kill_suffix}.json"
    
    logging.info(f"Generating multi-condition config with {len(realistic_combinations)} conditions...")
    
    # Create simulation conditions array
    simulations = []
    for idx, (spr, x_value, pq, pr) in enumerate(realistic_combinations):
        condition_name = f"s{s_value}_spr{spr:.1f}_{x_mode.lower()}{x_value:.1f}"
        
        # Calculate biological parameters from target probabilities
        # We want the actual probabilities to be pq, pr, and pq (for l)
        # So we need to solve: target_prob = biological_param * (1 - prob_death)
        q_param, r_param, l_param = calculate_biological_parameters_from_probabilities(
            target_prob_q=pq,
            target_prob_r=pr,
            target_prob_l=pq,  # Ps = Pq
            s=s_value,
            idle=0.1
        )
        
        condition = {
            "simulation": {
                "generations": 10000,
                "initial_size": 1000,
                "output_dir": str(output_dir),
                "output_prefix": condition_name,
                "number_of_replicates": 10,
                "output": {
                    "save_consolidated_summary": False
                }
            },
            "biological_parameters": {
                "s": s_value,
                "m": m_value,
                "q": q_param,
                "r": r_param,
                "l": l_param,
                "idle": 0.1
            },
            "treatment": {
                "schedule_type": "mtd" if kill_mechanism else "off",
                "drug_type": "abs",
                "treat_amt": 0.8,
                "pen_amt": 4.0,
                "dose_duration": 24,
                "treatment_start_size": 1000000000,
                "relapse_size": 4000000000,
                "penalty": penalty,
                "kill_mechanism": kill_mechanism,
                "secondary_therapy": False
            } if kill_mechanism else {
                "schedule_type": "off",
                "penalty": penalty
            }
        }
        simulations.append(condition)
    
    # Create master config with simulations array
    master_config = {
        "simulations": simulations,
        "use_multiprocessing": True, 
        "account_for_extinctions": False
    }
    
    with open(config_path, 'w') as f:
        json.dump(master_config, f, indent=2)
    
    logging.info(f"Config file created: {config_path}")
    return str(config_path)


def run_parameter_sweep(s_value, output_base_dir, penalty=False, x_mode='SPQ', kill_mechanism=None):
    """
    Run parameter sweep for a given selective advantage value.
    
    Parameters:
    -----------
    s_value : float
        Selective advantage (0.001, 0.01, or 0.1)
    output_base_dir : Path
        Base directory for all outputs
    penalty : bool, optional
        Whether to apply penalty to resistance (default: False)
    x_mode : str, optional
        Either 'SPQ' or 'PQR' mode (default: 'SPQ')
    kill_mechanism : str, optional
        Either 'kbe' or 'kod' for treatment mechanism, or None for no treatment (default: None)
    
    Returns:
    --------
    DataFrame
        Results containing SPR, SPQ/PQR, and resistance fractions
    """
    logging.info("="*80)
    logging.info(f"Starting parameter sweep for s = {s_value}")
    logging.info("="*80)
    
    # Create output directory for this s value
    s_dir = output_base_dir / f"s_{s_value}"
    s_dir.mkdir(parents=True, exist_ok=True)

    # Calculate m for target mutation probability of 3.4e-7
    target_mutation_prob = 3.4e-7
    m_value = calculate_m_from_mutation_probability(target_mutation_prob, s_value, idle=0.1)
    
    logging.info(f"Calculated parameters:")
    logging.info(f"  m = {m_value:.6e} (for target mutation prob {target_mutation_prob:.2e})")
    
    # Generate parameter space
    spr_range = np.arange(-5, 30.25, 0.25)
    x_range = np.arange(-5, 25.25, 0.25)  # Can be SPQ or PQR depending on mode
    
    # Store results
    results = []
    
    # Track progress
    total_combinations = len(spr_range) * len(x_range)
    valid_count = 0
    
    logging.info(f"Scanning parameter space...")
    logging.info(f"  Mode: {x_mode}")
    logging.info(f"  SPR range: {spr_range[0]:.2f} to {spr_range[-1]:.2f} (step 0.25)")
    logging.info(f"  {x_mode} range: {x_range[0]:.2f} to {x_range[-1]:.2f} (step 0.25)")
    logging.info(f"  Total combinations: {total_combinations:,}")
    
    # First pass: identify realistic combinations
    logging.info("Filtering for realistic parameter combinations...")
    realistic_combinations = []
    for spr in spr_range:
        for x_value in x_range:
            pq, pr, x_actual = calculate_probabilities(s_value, spr, x_value, x_mode)
            if is_realistic(pq, pr):
                realistic_combinations.append((spr, x_actual, pq, pr))
                valid_count += 1
    
    logging.info(f"Found {valid_count:,} realistic combinations ({100*valid_count/total_combinations:.1f}% of total)")
    
    # Generate multi-condition config file
    config_path = generate_multi_condition_config(
        s_value=s_value,
        realistic_combinations=realistic_combinations,
        m_value=m_value,
        output_dir=s_dir,
        penalty=penalty,
        x_mode=x_mode,
        kill_mechanism=kill_mechanism
    )
    
    # Run all simulations with multiprocessing
    logging.info(f"Starting {valid_count:,} simulations with multiprocessing...")
    logging.info(f"Penalty enabled: {penalty}")
    
    try:
        run_results = run_simulation_from_config(config_path, num_workers=None)
        
        # Get results from consolidated summary
        summaries = getattr(run_simulation_from_config, 'last_run_summaries', [])
        
        logging.info(f"Simulations complete. Processing {len(summaries)} results...")
        logging.info(f"Expected {valid_count} conditions × 10 replicates = {valid_count * 10} total results")
        
        # Sort summaries by condition_index and replicate_number for proper grouping
        summaries_sorted = sorted(summaries, key=lambda x: (x.get('condition_index', 0), x.get('replicate_number', 0)))
        
        # Process results and match back to SPR/x_mode values
        # Group results by condition (each condition has 10 replicates)
        for idx, (spr, x_value, pq, pr) in enumerate(realistic_combinations):
            # Get all summaries for this condition (10 replicates)
            condition_start = idx * 10
            condition_end = condition_start + 10
            condition_summaries = summaries_sorted[condition_start:condition_end]
            
            if len(condition_summaries) == 0:
                continue
            
            # Calculate median values across the 10 replicates
            total_cells_values = [s['final_total_cells'] for s in condition_summaries]
            resistant_cells_values = [s['final_resistant'] for s in condition_summaries]
            transient_cells_values = [s['final_transient'] for s in condition_summaries]
            
            median_total_cells = np.median(total_cells_values)
            median_resistant_cells = np.median(resistant_cells_values)
            median_transient_cells = np.median(transient_cells_values)
            median_total_resistant = median_resistant_cells + median_transient_cells
            
            resistant_fraction = median_resistant_cells / median_total_cells if median_total_cells > 0 else 0
            total_resistant_fraction = median_total_resistant / median_total_cells if median_total_cells > 0 else 0
            
            # For PQR mode, calculate transient fraction of resistant cells
            if x_mode == 'PQR':
                transient_fraction_of_resistant = median_transient_cells / median_total_resistant if median_total_resistant > 0 else 0
                primary_metric = transient_fraction_of_resistant
            else:
                primary_metric = total_resistant_fraction
            
            # Get most common final state
            final_states = [s['final_state'] for s in condition_summaries]
            final_state = max(set(final_states), key=final_states.count)
            
            # Create base result dictionary
            result = {
                'SPR': spr,
                x_mode: x_value,  # Dynamic column name based on mode
                's': s_value,
                'Pq': pq,
                'Pr': pr,
                'l': pq,
                'm': m_value,
                'median_total_cells': median_total_cells,
                'median_resistant_cells': median_resistant_cells,
                'median_transient_cells': median_transient_cells,
                'median_total_resistant_cells': median_total_resistant,
                'resistant_fraction': resistant_fraction,
                'total_resistant_fraction': total_resistant_fraction,
                'primary_metric': primary_metric,  # Either total_resistant_fraction (SPQ) or transient_fraction_of_resistant (PQR)
                'final_state': final_state,
                'num_replicates': len(condition_summaries)
            }
            
            # Add individual replicate values (up to 10 replicates)
            for rep_idx in range(10):
                if rep_idx < len(condition_summaries):
                    rep_summary = condition_summaries[rep_idx]
                    rep_total = rep_summary['final_total_cells']
                    rep_resistant = rep_summary['final_resistant']
                    rep_transient = rep_summary['final_transient']
                    rep_total_resistant = rep_resistant + rep_transient
                    
                    if x_mode == 'PQR':
                        rep_fraction = rep_transient / rep_total_resistant if rep_total_resistant > 0 else 0
                    else:
                        rep_fraction = rep_total_resistant / rep_total if rep_total > 0 else 0
                    
                    result[f'replicate_{rep_idx+1}_primary_metric'] = rep_fraction
                else:
                    # If fewer than 10 replicates, fill with NaN
                    result[f'replicate_{rep_idx+1}_primary_metric'] = np.nan
            
            results.append(result)
            
            # Progress logging every 10 conditions processed (= 100 simulations)
            if (idx + 1) % 10 == 0:
                successful_sims = (idx + 1) * 10
                logging.info(f"  Processed {idx + 1}/{len(realistic_combinations)} conditions ({successful_sims} total simulations)")
    
    except Exception as e:
        logging.error(f"Error running simulations: {e}")
        import traceback
        logging.error(traceback.format_exc())
        raise
    results_df = pd.DataFrame(results)
    
    if len(results_df) == 0:
        logging.warning("No results to save!")
        return results_df
    
    # Save results with penalty status and x_mode in filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    penalty_suffix = "penalty_on" if penalty else "penalty_off"
    kill_suffix = f"_{kill_mechanism}" if kill_mechanism else ""
    results_path = s_dir / f"parameter_sweep_results_s{s_value}_{x_mode}_{penalty_suffix}{kill_suffix}_{timestamp}.csv"
    results_df.to_csv(results_path, index=False)
    logging.info(f"Results saved to: {results_path}")
    logging.info(f"  Total rows: {len(results_df):,}")
    
    return results_df


def create_heatmap(results_df, s_value, output_dir, penalty=False, x_mode='SPQ', kill_mechanism=None):
    """
    Create heatmap visualization of resistance outcomes.
    
    Parameters:
    -----------
    results_df : DataFrame
        Results from parameter sweep
    s_value : float
        Selective advantage value
    output_dir : Path
        Directory to save heatmap
    penalty : bool, optional
        Whether penalty was enabled (for filename)
    x_mode : str, optional
        Either 'SPQ' or 'PQR' mode (default: 'SPQ')
    kill_mechanism : str, optional
        Either 'kbe' or 'kod' for treatment mechanism, or None for no treatment (default: None)
    """
    logging.info(f"Generating heatmap for s = {s_value}...")
    
    # Define color mapping based on metric (different interpretation for PQR vs SPQ)
    def get_color_index(fraction):
        if fraction < 0.05:
            return 0  # red
        elif fraction < 0.45:
            return 1  # orange
        elif fraction < 0.95:
            return 2  # yellow
        else:
            return 3  # blue
    
    # Get unique sorted values for SPR and x_mode
    spr_unique = np.sort(results_df['SPR'].unique())
    x_unique = np.sort(results_df[x_mode].unique())
    
    # Create 2D grid for heatmap data
    # Initialize with NaN (will show as white/empty)
    heatmap_data = np.full((len(spr_unique), len(x_unique)), np.nan)
    
    # Fill in the heatmap data using primary_metric
    for _, row in results_df.iterrows():
        spr_idx = np.where(spr_unique == row['SPR'])[0][0]
        x_idx = np.where(x_unique == row[x_mode])[0][0]
        heatmap_data[spr_idx, x_idx] = get_color_index(row['primary_metric'])
    
    # Create custom colormap
    from matplotlib.colors import ListedColormap
    colors = ['red', 'orange', 'gold', 'blue']
    cmap = ListedColormap(colors)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Plot heatmap using imshow
    # Use extent to map indices to actual SPR/x_mode values
    # Need to account for half-pixel offset for proper alignment
    step_size = 0.25
    extent = [
        x_unique[0] - step_size/2,  # left
        x_unique[-1] + step_size/2,  # right
        spr_unique[0] - step_size/2,  # bottom
        spr_unique[-1] + step_size/2   # top
    ]
    
    im = ax.imshow(heatmap_data, cmap=cmap, aspect='auto', 
                   origin='lower', extent=extent, 
                   vmin=0, vmax=3, interpolation='nearest')
    
    # Labels and title based on mode
    if x_mode == 'SPQ':
        x_label = 'SPQ = log(s/Pq)'
        metric_description = 'Total Resistant Fraction'
        legend_labels = [
            'Total Resistant < 5%',
            '5% ≤ Total Resistant < 45%',
            '45% ≤ Total Resistant < 95%',
            'Total Resistant ≥ 95%'
        ]
    else:  # PQR
        x_label = 'PQR = log(Pq/Pr)'
        metric_description = 'Transient Fraction of Resistant Cells'
        legend_labels = [
            'Transient/Resistant < 5%',
            '5% ≤ Transient/Resistant < 45%',
            '45% ≤ Transient/Resistant < 95%',
            'Transient/Resistant ≥ 95%'
        ]
    
    ax.set_xlabel(x_label, fontsize=14, fontweight='bold')
    ax.set_ylabel('SPR = log(s/Pr)', fontsize=14, fontweight='bold')
    ax.set_title(f'{metric_description} Heatmap (s = {s_value}, {x_mode} mode, median of 10 replicates)', 
                 fontsize=16, fontweight='bold', pad=20)
    
    handles = [plt.Rectangle((0, 0), 1, 1, fc=colors[i]) for i in range(4)]
    ax.legend(handles, legend_labels, loc='best', fontsize=11, framealpha=0.9)
    
    # Add gridlines at parameter boundaries
    # Set major ticks to show every 5 units
    ax.set_xticks(np.arange(x_unique[0], x_unique[-1] + 1, 5))
    ax.set_yticks(np.arange(spr_unique[0], spr_unique[-1] + 1, 5))
    
    # Add minor gridlines at step boundaries for reference
    ax.set_xticks(x_unique, minor=True)
    ax.set_yticks(spr_unique, minor=True)
    ax.grid(which='major', alpha=0.5, linestyle='-', linewidth=0.8)
    ax.grid(which='minor', alpha=0.2, linestyle=':', linewidth=0.5)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure with penalty status and x_mode in filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    penalty_suffix = "penalty_on" if penalty else "penalty_off"
    kill_suffix = f"_{kill_mechanism}" if kill_mechanism else ""
    heatmap_path = output_dir / f"resistance_heatmap_s{s_value}_{x_mode}_{penalty_suffix}{kill_suffix}_{timestamp}.png"
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    logging.info(f"Heatmap saved to: {heatmap_path}")
    
    plt.close()


def main(s_value=None, penalty=False, x_mode='SPQ', kill_mechanism=None):
    """
    Main execution function.
    
    Parameters:
    -----------
    s_value : float, optional
        Selective advantage value to analyze (e.g., 0.1, 0.01, 0.001)
        If not provided, will be read from command line arguments
    penalty : bool, optional
        Whether to apply penalty to resistance (default: False)
    x_mode : str, optional
        Either 'SPQ' or 'PQR' mode (default: 'SPQ')
    kill_mechanism : str, optional
        Either 'kbe' or 'kod' for treatment mechanism, or None for no treatment (default: None)
    """
    # Setup logging
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logging.info("="*80)
    logging.info("PARAMETER SWEEP ANALYSIS")
    logging.info("="*80)
    logging.info(f"This script performs systematic parameter sweeps across SPR and {x_mode} space")
    logging.info("Each condition is run with 10 replicates, and median values are reported.")
    logging.info("")
    logging.info("SPR = log(s/Pr) - Selective advantage to permanent resistance ratio")
    if x_mode == 'SPQ':
        logging.info("SPQ = log(s/Pq) - Selective advantage to transient resistance ratio")
    else:
        logging.info("PQR = log(Pq/Pr) - Transient to permanent resistance ratio")
    logging.info("Ps = Pq (transient resistance loss probability equals gain probability)")
    logging.info("")
    logging.info("Realistic constraint: 1e-11 ≤ Pq, Pr ≤ 1e-2")
    logging.info("="*80)
    
    # Create base output directory
    base_output_dir = Path("./parameter_sweep_output")
    base_output_dir.mkdir(exist_ok=True)
    logging.info(f"Output directory: {base_output_dir.absolute()}")
    
    # Determine number of workers (use all available cores)
    from multiprocessing import cpu_count
    num_workers = cpu_count()
    logging.info(f"Using {num_workers} parallel workers")
    logging.info(f"Penalty enabled: {penalty}")
    logging.info(f"Mode: {x_mode}")
    if kill_mechanism:
        logging.info(f"Treatment: MTD with {kill_mechanism.upper()} kill mechanism")
    else:
        logging.info(f"Treatment: OFF (no kill mechanism specified)")
    
    # Validate s_value
    if s_value is None:
        logging.error("Error: No selective advantage value provided")
        logging.error("Usage: python parameter_sweep_analysis.py <s_value> [penalty] [mode] [kill_mechanism]")
        logging.error("Example: python parameter_sweep_analysis.py 0.1")
        logging.error("         python parameter_sweep_analysis.py 0.01 true SPQ kbe")
        logging.error("         python parameter_sweep_analysis.py 0.1 false PQR kod")
        sys.exit(1)
    
    try:
        logging.info(f"\nProcessing selective advantage s = {s_value}")
        
        # Run parameter sweep
        results_df = run_parameter_sweep(s_value, base_output_dir, penalty=penalty, x_mode=x_mode, kill_mechanism=kill_mechanism)
        
        if len(results_df) == 0:
            logging.warning(f"No results for s = {s_value}, skipping heatmap")
            return
        
        # Create heatmap
        s_dir = base_output_dir / f"s_{s_value}"
        create_heatmap(results_df, s_value, s_dir, penalty=penalty, x_mode=x_mode, kill_mechanism=kill_mechanism)
        
        # Print summary statistics
        if x_mode == 'PQR':
            metric_name = "transient fraction of resistant cells"
        else:
            metric_name = "total resistant fraction"
        
        logging.info(f"Summary for s = {s_value}:")
        logging.info(f"  Total simulations: {len(results_df):,}")
        logging.info(f"  Mean {metric_name}: {results_df['primary_metric'].mean():.4f}")
        logging.info(f"  Median {metric_name}: {results_df['primary_metric'].median():.4f}")
        logging.info(f"  Min {metric_name}: {results_df['primary_metric'].min():.4f}")
        logging.info(f"  Max {metric_name}: {results_df['primary_metric'].max():.4f}")
        
    except Exception as e:
        logging.error(f"Error processing s = {s_value}: {e}")
        import traceback
        logging.error(traceback.format_exc())
        sys.exit(1)
    
    logging.info("\n" + "="*80)
    logging.info("ANALYSIS COMPLETE")
    logging.info("="*80)
    logging.info(f"Results saved to: {base_output_dir / f's_{s_value}'}")


if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Error: No selective advantage value provided")
        print("Usage: python parameter_sweep_analysis.py <s_value> [penalty] [mode] [kill_mechanism]")
        print("Example: python parameter_sweep_analysis.py 0.1")
        print("         python parameter_sweep_analysis.py 0.01 true SPQ kbe")
        print("         python parameter_sweep_analysis.py 0.1 false PQR kod")
        sys.exit(1)
    
    try:
        s_val = float(sys.argv[1])
    except ValueError:
        print(f"Error: Invalid selective advantage value '{sys.argv[1]}'. Must be a number.")
        sys.exit(1)
    
    penalty_flag = False
    if len(sys.argv) > 2:
        penalty_str = sys.argv[2].lower()
        if penalty_str in ['true', '1', 'yes', 't', 'y']:
            penalty_flag = True
        elif penalty_str in ['false', '0', 'no', 'f', 'n']:
            penalty_flag = False
        else:
            print(f"Error: Invalid penalty value '{sys.argv[2]}'. Must be true/false (or 1/0, yes/no, etc.).")
            sys.exit(1)
    
    mode_val = 'SPQ'  # Default mode
    if len(sys.argv) > 3:
        mode_str = sys.argv[3].upper()
        if mode_str in ['SPQ', 'PQR']:
            mode_val = mode_str
        else:
            print(f"Error: Invalid mode '{sys.argv[3]}'. Must be 'SPQ' or 'PQR'.")
            sys.exit(1)
    
    kill_mechanism_val = None  # Default: no treatment
    if len(sys.argv) > 4:
        kill_str = sys.argv[4].lower()
        if kill_str in ['kbe', 'kod']:
            kill_mechanism_val = kill_str
        else:
            print(f"Error: Invalid kill mechanism '{sys.argv[4]}'. Must be 'kbe' or 'kod'.")
            sys.exit(1)
    
    main(s_value=s_val, penalty=penalty_flag, x_mode=mode_val, kill_mechanism=kill_mechanism_val)
