"""
Resistance Distribution Analysis

This module runs tumor simulations from a configuration file and analyzes
the distribution of resistance proportions (transient + resistant cells / total cells).

It performs:
1. Simulation execution using tumour_simulation.py
2. Saves raw resistance proportion data for further analysis
3. Fits multiple distributions to log-transformed resistance proportions
4. Compares distributions using AIC, BIC, KS test, and R-squared
5. Visualizes the best-fitting distributions

Focus on distributions that can capture heavy skew and one-sided tails.
"""

import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import (gengamma, gamma, weibull_min, lognorm, invgauss, 
                         burr12, kstest)
from pathlib import Path
from datetime import datetime
import warnings

# Import the simulation runner
from tumour_simulation import run_simulation_from_config

# Suppress overflow warnings from scipy during fitting
warnings.filterwarnings('ignore', category=RuntimeWarning)


def calculate_resistance_proportions(results_list):
    """
    Calculate resistance proportion for each simulation result.
    
    Resistance proportion = (transient cells + resistant cells) / total cells
    
    Parameters:
    -----------
    results_list : list of tuple
        List of (state, DataFrame) from simulation runs
        
    Returns:
    --------
    numpy.array
        Array of resistance proportions for each successful simulation
    """
    proportions = []
    
    for state, df in results_list:
        if len(df) > 0:
            # Calculate totals from the dataframe
            total_cells = df['N'].sum()
            transient_cells = df['Nq'].sum()
            resistant_cells = df['Nr'].sum()
            
            if total_cells > 0:
                resistance_prop = (transient_cells + resistant_cells) / total_cells
                proportions.append(resistance_prop)
    
    return np.array(proportions)


def prepare_log_data(data):
    """
    Prepare log-transformed data with offset if needed.
    
    Parameters:
    -----------
    data : array-like
        Resistance proportions
        
    Returns:
    --------
    tuple
        (log_data_shifted, offset)
    """
    # Remove zeros and take log
    data_positive = data[data > 0]
    
    if len(data_positive) == 0:
        raise ValueError("No positive values in data")
    
    log_data = np.log(data_positive)
    
    # Shift to positive range if needed
    min_log = np.min(log_data)
    if min_log <= 0:
        offset = -min_log + 0.001
        log_data_shifted = log_data + offset
    else:
        offset = 0
        log_data_shifted = log_data
    
    return log_data_shifted, offset


def fit_single_distribution(log_data, dist_name, dist_obj):
    """
    Fit a single distribution to log-transformed data.
    
    Parameters:
    -----------
    log_data : array-like
        Log-transformed and shifted data
    dist_name : str
        Name of distribution
    dist_obj : scipy.stats distribution
        Distribution object from scipy.stats
        
    Returns:
    --------
    dict or None
        Dictionary with fit parameters and quality metrics, or None if fit failed
    """
    try:
        # Fit distribution
        params = dist_obj.fit(log_data, floc=0)
        
        # Calculate goodness-of-fit metrics
        n = len(log_data)
        
        # Log-likelihood
        log_likelihood = np.sum(dist_obj.logpdf(log_data, *params))
        
        # Number of parameters (excluding fixed loc)
        k = len(params) - 1
        
        # AIC and BIC
        aic = 2 * k - 2 * log_likelihood
        bic = k * np.log(n) - 2 * log_likelihood
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_pval = kstest(log_data, lambda x: dist_obj.cdf(x, *params))
        
        # R-squared (correlation between observed and theoretical quantiles)
        sorted_data = np.sort(log_data)
        theoretical_quantiles = dist_obj.ppf(np.linspace(0.01, 0.99, n), *params)
        
        ss_res = np.sum((sorted_data - theoretical_quantiles)**2)
        ss_tot = np.sum((sorted_data - np.mean(sorted_data))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            'distribution': dist_name,
            'params': params,
            'n_params': k,
            'aic': aic,
            'bic': bic,
            'log_likelihood': log_likelihood,
            'ks_statistic': ks_stat,
            'ks_pvalue': ks_pval,
            'r_squared': r_squared
        }
    
    except Exception as e:
        print(f"  Warning: Failed to fit {dist_name}: {str(e)}")
        return None


def fit_multiple_distributions(log_data):
    """
    Fit multiple distributions to log-transformed data and compare.
    
    Focuses on distributions that can handle heavy skew and one-sided tails.
    
    Parameters:
    -----------
    log_data : array-like
        Log-transformed and shifted resistance proportions
        
    Returns:
    --------
    list of dict
        List of fit results, sorted by AIC (best first)
    """
    # Dictionary of distributions to try
    # Focus on those good for skewed, heavy-tailed data
    distributions = {
        'Generalized Gamma': gengamma,
        'Gamma': gamma,
        'Weibull': weibull_min,
        'Log-Normal': lognorm,
        'Inverse Gaussian': invgauss,
        'Burr (Type XII)': burr12,
    }
    
    print("\nFitting multiple distributions to log-transformed data...")
    print("="*80)
    
    results = []
    
    for dist_name, dist_obj in distributions.items():
        print(f"  Fitting {dist_name}...", end=" ")
        fit_result = fit_single_distribution(log_data, dist_name, dist_obj)
        
        if fit_result is not None:
            results.append(fit_result)
            print(f"✓ (AIC={fit_result['aic']:.2f}, R²={fit_result['r_squared']:.4f})")
        else:
            print("✗ Failed")
    
    # Sort by AIC (lower is better)
    results.sort(key=lambda x: x['aic'])
    
    return results


def print_comparison_table(results):
    """
    Print a formatted comparison table of distribution fits.
    
    Parameters:
    -----------
    results : list of dict
        Fit results from fit_multiple_distributions
    """
    print("\n" + "="*80)
    print("DISTRIBUTION COMPARISON TABLE")
    print("(Sorted by AIC - lower is better)")
    print("="*80)
    
    # Print header
    print(f"{'Rank':<5} {'Distribution':<20} {'AIC':<12} {'BIC':<12} {'R²':<10} {'KS p-val':<12}")
    print("-" * 80)
    
    # Print each distribution
    for i, result in enumerate(results, 1):
        # Determine if fit is acceptable
        acceptable = "✓" if result['ks_pvalue'] > 0.05 and result['r_squared'] > 0.80 else "✗"
        
        print(f"{i:<5} {result['distribution']:<20} "
              f"{result['aic']:<12.2f} {result['bic']:<12.2f} "
              f"{result['r_squared']:<10.4f} {result['ks_pvalue']:<12.6f} {acceptable}")
    
    print("="*80)
    
    # Print best fit summary
    best = results[0]
    print(f"\nBEST FIT: {best['distribution']}")
    print(f"  AIC: {best['aic']:.2f}")
    print(f"  BIC: {best['bic']:.2f}")
    print(f"  R²: {best['r_squared']:.4f}")
    print(f"  KS p-value: {best['ks_pvalue']:.6f}")
    
    if best['ks_pvalue'] > 0.05:
        print(f"  ✓ Passes KS test at 5% significance")
    else:
        print(f"  ✗ Fails KS test at 5% significance")
    
    print("="*80)


def get_distribution_object(dist_name):
    """Get scipy distribution object by name."""
    dist_map = {
        'Generalized Gamma': gengamma,
        'Gamma': gamma,
        'Weibull': weibull_min,
        'Log-Normal': lognorm,
        'Inverse Gaussian': invgauss,
        'Burr (Type XII)': burr12,
    }
    return dist_map.get(dist_name)


def plot_top_distributions(log_data, offset, results, top_n=3, output_path=None):
    """
    Plot the top N best-fitting distributions.
    
    Parameters:
    -----------
    log_data : array-like
        Log-transformed and shifted data
    offset : float
        Offset applied to data
    results : list of dict
        Fit results sorted by quality
    top_n : int
        Number of top distributions to plot
    output_path : str, optional
        Path to save plot
    """
    n_plots = min(top_n, len(results))
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, n_plots, figsize=(6*n_plots, 10))
    if n_plots == 1:
        axes = axes.reshape(-1, 1)
    
    colors = ['red', 'green', 'blue', 'orange', 'purple']
    
    for i in range(n_plots):
        result = results[i]
        dist_name = result['distribution']
        params = result['params']
        dist_obj = get_distribution_object(dist_name)
        
        # Top row: Histogram with fitted PDF
        ax_pdf = axes[0, i]
        ax_pdf.hist(log_data, bins=30, density=True, alpha=0.6, 
                   color='steelblue', edgecolor='black', label='Data')
        
        x = np.linspace(log_data.min(), log_data.max(), 1000)
        fitted_pdf = dist_obj.pdf(x, *params)
        ax_pdf.plot(x, fitted_pdf, color=colors[i], linewidth=2.5, 
                   label=f'{dist_name}')
        
        xlabel = 'Log(Resistance)' + (f' + {offset:.3f}' if offset > 0 else '')
        ax_pdf.set_xlabel(xlabel, fontsize=11)
        ax_pdf.set_ylabel('Density', fontsize=11)
        ax_pdf.set_title(f'#{i+1}: {dist_name}\nAIC={result["aic"]:.1f}, R²={result["r_squared"]:.4f}', 
                        fontsize=12, fontweight='bold')
        ax_pdf.legend(fontsize=10)
        ax_pdf.grid(True, alpha=0.3)
        
        # Bottom row: Q-Q plot
        ax_qq = axes[1, i]
        sorted_data = np.sort(log_data)
        n = len(sorted_data)
        theoretical_q = dist_obj.ppf(np.linspace(0.01, 0.99, n), *params)
        
        ax_qq.scatter(theoretical_q, sorted_data, alpha=0.5, s=30,
                     color=colors[i], edgecolor='black')
        
        # Perfect fit line
        min_val = min(theoretical_q.min(), sorted_data.min())
        max_val = max(theoretical_q.max(), sorted_data.max())
        ax_qq.plot([min_val, max_val], [min_val, max_val], 'k--', 
                  linewidth=2, label='Perfect fit')
        
        ax_qq.set_xlabel('Theoretical Quantiles', fontsize=11)
        ax_qq.set_ylabel('Observed Quantiles', fontsize=11)
        ax_qq.set_title(f'Q-Q Plot (p={result["ks_pvalue"]:.4f})', fontsize=11)
        ax_qq.legend(fontsize=9)
        ax_qq.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nComparison plot saved to: {output_path}")
    else:
        plt.show()


def save_resistance_data(resistance_props, output_dir, timestamp, s_value=None):
    """
    Save raw resistance proportion data to CSV for further analysis.
    
    Parameters:
    -----------
    resistance_props : array-like
        Raw resistance proportions from simulations
    output_dir : Path
        Directory to save data
    timestamp : str
        Timestamp for filename
    s_value : float, optional
        Initial s value from configuration
    """
    s_str = f"_s{s_value}" if s_value is not None else ""
    data_path = output_dir / f"raw_resistance_proportions{s_str}_{timestamp}.csv"
    
    # Create DataFrame with resistance proportions
    df = pd.DataFrame({
        'resistance_proportion': resistance_props,
        'log_resistance_proportion': np.log(resistance_props[resistance_props > 0])
    })
    
    # Add summary statistics as a comment
    with open(data_path, 'w') as f:
        f.write(f"# Raw Resistance Proportion Data\n")
        f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        if s_value is not None:
            f.write(f"# Initial s value: {s_value}\n")
        f.write(f"# N samples: {len(resistance_props)}\n")
        f.write(f"# Mean: {np.mean(resistance_props):.6f}\n")
        f.write(f"# Median: {np.median(resistance_props):.6f}\n")
        f.write(f"# Std: {np.std(resistance_props):.6f}\n")
        f.write(f"# Min: {np.min(resistance_props):.6f}\n")
        f.write(f"# Max: {np.max(resistance_props):.6f}\n")
        f.write(f"#\n")
    
    # Append data
    df.to_csv(data_path, mode='a', index=False)
    
    print(f"Raw resistance data saved to: {data_path}")
    return data_path


def analyze_resistance_distribution(config_path, num_workers=None, save_plot=True, output_dir=None):
    """
    Main function to run simulations and analyze resistance distribution.
    
    Parameters:
    -----------
    config_path : str
        Path to simulation configuration file
    num_workers : int, optional
        Number of parallel workers for simulation
    save_plot : bool
        Whether to save the plot (True) or display it (False)
    output_dir : str, optional
        Directory to save outputs (default: same as simulation output_dir)
        
    Returns:
    --------
    dict
        Dictionary with all analysis results
    """
    # Extract s value from config file
    s_value = None
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            # Get s from biological_parameters
            if 'biological_parameters' in config and 's' in config['biological_parameters']:
                s_value = config['biological_parameters']['s']
    except Exception as e:
        print(f"Warning: Could not extract s value from config: {e}")
    
    print("="*80)
    print("RUNNING SIMULATIONS")
    print("="*80)
    if s_value is not None:
        print(f"Initial s value: {s_value}")
        print("="*80)
    
    # Run simulations
    results_list = run_simulation_from_config(config_path, num_workers=num_workers)
    
    if len(results_list) == 0:
        raise ValueError("No successful simulations to analyze")
    
    print("\n" + "="*80)
    print("CALCULATING RESISTANCE PROPORTIONS")
    print("="*80)
    
    # Calculate resistance proportions
    resistance_props = calculate_resistance_proportions(results_list)
    
    print(f"Calculated resistance proportions for {len(resistance_props)} simulations")
    print(f"Range: [{resistance_props.min():.6f}, {resistance_props.max():.6f}]")
    
    # Setup output directory
    if output_dir is None:
        output_dir = Path('./output')
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save raw resistance proportion data
    print("\n" + "="*80)
    print("SAVING RAW RESISTANCE DATA")
    print("="*80)
    
    save_resistance_data(resistance_props, output_dir, timestamp, s_value)
    
    # Prepare log-transformed data
    print("\n" + "="*80)
    print("PREPARING LOG-TRANSFORMED DATA")
    print("="*80)
    
    log_data, offset = prepare_log_data(resistance_props)
    print(f"Log-transformed {len(log_data)} resistance proportions")
    print(f"Offset applied: {offset:.6f}")
    print(f"Log data range: [{log_data.min():.4f}, {log_data.max():.4f}]")
    
    # Fit multiple distributions
    print("\n" + "="*80)
    print("FITTING MULTIPLE DISTRIBUTIONS")
    print("="*80)
    
    fit_results = fit_multiple_distributions(log_data)
    
    if len(fit_results) == 0:
        print("Error: No distributions could be fitted successfully")
        return None
    
    # Print comparison table
    print_comparison_table(fit_results)
    
    # Create visualization
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    s_str = f"_s{s_value}" if s_value is not None else ""
    
    if save_plot:
        plot_path = output_dir / f"distribution_comparison{s_str}_{timestamp}.png"
        plot_top_distributions(log_data, offset, fit_results, top_n=3, output_path=plot_path)
    else:
        plot_top_distributions(log_data, offset, fit_results, top_n=3, output_path=None)
    
    # Save comparison results to CSV
    comparison_path = output_dir / f"distribution_comparison{s_str}_{timestamp}.csv"
    comparison_df = pd.DataFrame([{
        'rank': i+1,
        'distribution': r['distribution'],
        'aic': r['aic'],
        'bic': r['bic'],
        'r_squared': r['r_squared'],
        'ks_statistic': r['ks_statistic'],
        'ks_pvalue': r['ks_pvalue'],
        'log_likelihood': r['log_likelihood'],
        'n_parameters': r['n_params'],
        'acceptable_fit': 'Yes' if r['ks_pvalue'] > 0.05 and r['r_squared'] > 0.80 else 'No'
    } for i, r in enumerate(fit_results)])
    
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\nDistribution comparison saved to: {comparison_path}")
    
    # Compile all results
    analysis_results = {
        'resistance_proportions': resistance_props,
        'log_data': log_data,
        'offset': offset,
        'fit_results': fit_results,
        'best_distribution': fit_results[0]['distribution'],
        'n_simulations': len(results_list),
        's_value': s_value
    }
    
    return analysis_results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_resistance_distribution.py <config_file.json> [num_workers]")
        print("\nExample:")
        print("  python analyze_resistance_distribution.py example_configs/multi_replicate_example.json 4")
        sys.exit(1)
    
    config_file = sys.argv[1]
    num_workers = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    # Run analysis
    results = analyze_resistance_distribution(
        config_path=config_file,
        num_workers=num_workers,
        save_plot=True
    )
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
