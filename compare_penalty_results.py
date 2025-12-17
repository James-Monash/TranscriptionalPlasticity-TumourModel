"""
Compare Penalty Results Analysis Script

Compares parameter sweep results between two result files using Mann-Whitney U test
to determine if the distributions of the primary metric differ between the two files.

The null hypothesis is that the two distributions are the same. A low p-value indicates
evidence that the distributions are significantly different.

Usage:
    python compare_penalty_results.py <file1_path> <file2_path>
    
Example:
    python compare_penalty_results.py results_penalty_on.csv results_penalty_off.csv
"""

import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
from pathlib import Path
import sys
import logging
from datetime import datetime


def load_results(file_path):
    """
    Load results file from given path.
    
    Parameters:
    -----------
    file_path : str or Path
        Path to CSV results file
    
    Returns:
    --------
    DataFrame or None
        Loaded results, or None if file not found
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        logging.error(f"File not found: {file_path}")
        return None
    
    logging.info(f"Loading data from: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        logging.info(f"  Loaded {len(df)} parameter combinations")
        return df
    except Exception as e:
        logging.error(f"Error reading file: {e}")
        return None


def validate_matching_parameters(df1, df2):
    """
    Validate that both dataframes have matching SPR and x-axis parameter values.
    
    Parameters:
    -----------
    df1 : DataFrame
        First results dataframe
    df2 : DataFrame
        Second results dataframe
    
    Returns:
    --------
    str or None
        Name of x-axis column ('SPQ' or 'PQR'), or None if validation fails
    """
    # Determine which mode we're using (SPQ or PQR)
    x_mode = None
    if 'SPQ' in df1.columns and 'SPQ' in df2.columns:
        x_mode = 'SPQ'
    elif 'PQR' in df1.columns and 'PQR' in df2.columns:
        x_mode = 'PQR'
    else:
        logging.error("Could not find SPQ or PQR column in both files")
        return None
    
    # Check if both have the same parameters
    params1 = set(zip(df1['SPR'].round(6), df1[x_mode].round(6)))
    params2 = set(zip(df2['SPR'].round(6), df2[x_mode].round(6)))
    
    if params1 != params2:
        logging.error("ERROR: Parameter combinations do not match between files")
        logging.error(f"File 1 has {len(params1)} unique parameter combinations")
        logging.error(f"File 2 has {len(params2)} unique parameter combinations")
        
        only_in_1 = params1 - params2
        only_in_2 = params2 - params1
        
        if only_in_1:
            logging.error(f"Parameters only in file 1 (first 5): {list(only_in_1)[:5]}")
        if only_in_2:
            logging.error(f"Parameters only in file 2 (first 5): {list(only_in_2)[:5]}")
        
        return None
    
    logging.info(f"✓ Both files have matching parameter combinations ({len(params1)} total)")
    logging.info(f"✓ Mode detected: {x_mode}")
    
    return x_mode


def compare_results(file1_path, file2_path):
    """
    Compare two result files using Mann-Whitney U test.
    
    The Mann-Whitney U test compares the distributions of the primary metric
    between the two files. The null hypothesis is that the distributions are
    the same. A low p-value indicates evidence that they differ.
    
    Parameters:
    -----------
    file1_path : str
        Path to first results file
    file2_path : str
        Path to second results file
    
    Returns:
    --------
    tuple of (DataFrame, dict, str)
        (detailed_results, mannwhitney_test_result, x_mode)
    """
    logging.info(f"Comparing two result files:")
    logging.info(f"  File 1: {file1_path}")
    logging.info(f"  File 2: {file2_path}")
    
    # Load both datasets
    df1 = load_results(file1_path)
    df2 = load_results(file2_path)
    
    if df1 is None or df2 is None:
        logging.error("Failed to load one or both datasets")
        return None, None, None
    
    # Validate matching parameters
    x_mode = validate_matching_parameters(df1, df2)
    if x_mode is None:
        return None, None, None
    
    # Merge on parameter values
    merged = pd.merge(
        df1[['SPR', x_mode, 'primary_metric']],
        df2[['SPR', x_mode, 'primary_metric']],
        on=['SPR', x_mode],
        suffixes=('_file1', '_file2'),
        how='inner'
    )
    
    # Calculate summary statistics
    logging.info(f"\nSummary statistics:")
    logging.info(f"  File 1: mean={df1['primary_metric'].mean():.4f}, median={df1['primary_metric'].median():.4f}, std={df1['primary_metric'].std():.4f}")
    logging.info(f"  File 2: mean={df2['primary_metric'].mean():.4f}, median={df2['primary_metric'].median():.4f}, std={df2['primary_metric'].std():.4f}")
    
    # Perform Mann-Whitney U test
    # Note: This tests if the two samples come from the same distribution
    statistic, p_value = mannwhitneyu(
        merged['primary_metric_file1'],
        merged['primary_metric_file2'],
        alternative='two-sided'
    )
    
    logging.info(f"\nMann-Whitney U Test Results:")
    logging.info(f"  U-statistic: {statistic:.2f}")
    logging.info(f"  P-value: {p_value}")
    logging.info(f"  Significant (α=0.05): {'YES' if p_value < 0.05 else 'NO'}")
    
    if p_value < 0.05:
        logging.info(f"  Interpretation: The distributions ARE significantly different")
    else:
        logging.info(f"  Interpretation: No evidence that distributions differ")
    
    mannwhitney_result = {
        'statistic': statistic,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'file1_mean': df1['primary_metric'].mean(),
        'file1_median': df1['primary_metric'].median(),
        'file1_std': df1['primary_metric'].std(),
        'file2_mean': df2['primary_metric'].mean(),
        'file2_median': df2['primary_metric'].median(),
        'file2_std': df2['primary_metric'].std(),
        'n_comparisons': len(merged)
    }
    
    return merged, mannwhitney_result, x_mode


def save_results(merged_df, mannwhitney_result, file1_path, file2_path, x_mode):
    """
    Save comparison results to CSV.
    
    Parameters:
    -----------
    merged_df : DataFrame
        Merged results with primary metrics
    mannwhitney_result : dict
        Mann-Whitney U test results
    file1_path : str
        Path to first file
    file2_path : str
        Path to second file
    x_mode : str
        Either 'SPQ' or 'PQR' mode
    """
    # Create output directory based on file locations
    file1_path = Path(file1_path)
    output_dir = file1_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed comparison
    output_path = output_dir / f"mannwhitney_comparison_{x_mode}_{timestamp}.csv"
    merged_df.to_csv(output_path, index=False)
    logging.info(f"\nDetailed results saved to: {output_path}")
    
    # Save Mann-Whitney test summary
    summary_path = output_dir / f"mannwhitney_summary_{x_mode}_{timestamp}.txt"
    with open(summary_path, 'w') as f:
        f.write("Mann-Whitney U Test Comparison\n")
        f.write("="*60 + "\n\n")
        f.write(f"File 1: {file1_path}\n")
        f.write(f"File 2: {file2_path}\n\n")
        f.write(f"Mode: {x_mode}\n")
        f.write(f"Number of parameter combinations: {mannwhitney_result['n_comparisons']}\n\n")
        f.write("Summary Statistics:\n")
        f.write(f"  File 1:\n")
        f.write(f"    Mean:   {mannwhitney_result['file1_mean']:.4f}\n")
        f.write(f"    Median: {mannwhitney_result['file1_median']:.4f}\n")
        f.write(f"    Std:    {mannwhitney_result['file1_std']:.4f}\n")
        f.write(f"  File 2:\n")
        f.write(f"    Mean:   {mannwhitney_result['file2_mean']:.4f}\n")
        f.write(f"    Median: {mannwhitney_result['file2_median']:.4f}\n")
        f.write(f"    Std:    {mannwhitney_result['file2_std']:.4f}\n\n")
        f.write(f"Test Results:\n")
        f.write(f"  U-statistic: {mannwhitney_result['statistic']:.2f}\n")
        f.write(f"  P-value: {mannwhitney_result['p_value']}\n")
        f.write(f"  Significant (α=0.05): {'YES' if mannwhitney_result['significant'] else 'NO'}\n\n")
        f.write(f"Interpretation:\n")
        if mannwhitney_result['significant']:
            f.write(f"  The distributions ARE significantly different (reject H0).\n")
        else:
            f.write(f"  No evidence that distributions differ (fail to reject H0).\n")
    
    logging.info(f"Summary saved to: {summary_path}")


def create_heatmap(merged_df, mannwhitney_result, file1_path, x_mode):
    """
    Create heatmap visualization showing differences between files.
    
    Parameters:
    -----------
    merged_df : DataFrame
        Merged results with primary metrics
    mannwhitney_result : dict
        Mann-Whitney U test results
    file1_path : str
        Path to first file (for output directory)
    x_mode : str
        Either 'SPQ' or 'PQR' mode
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    
    logging.info(f"Generating difference heatmap...")
    
    # Get unique sorted values for SPR and x_mode
    spr_unique = np.sort(merged_df['SPR'].unique())
    x_unique = np.sort(merged_df[x_mode].unique())
    
    # Create 2D grid for heatmap data (difference: file2 - file1)
    heatmap_data = np.full((len(spr_unique), len(x_unique)), np.nan)
    
    # Fill in the heatmap data with differences
    for _, row in merged_df.iterrows():
        spr_idx = np.where(spr_unique == row['SPR'])[0][0]
        x_idx = np.where(x_unique == row[x_mode])[0][0]
        heatmap_data[spr_idx, x_idx] = row['primary_metric_file2'] - row['primary_metric_file1']
    
    # Create diverging colormap (blue = file1 higher, red = file2 higher)
    cmap = plt.cm.RdBu_r
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Plot heatmap using imshow
    step_size = 0.25
    extent = [
        x_unique[0] - step_size/2,
        x_unique[-1] + step_size/2,
        spr_unique[0] - step_size/2,
        spr_unique[-1] + step_size/2
    ]
    
    # Use symmetric color scale around zero
    vmax = np.nanmax(np.abs(heatmap_data))
    im = ax.imshow(heatmap_data, cmap=cmap, aspect='auto',
                   origin='lower', extent=extent,
                   vmin=-vmax, vmax=vmax, interpolation='nearest')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Difference (File2 - File1)', rotation=270, labelpad=20, fontsize=12)
    
    # Labels and title based on mode
    if x_mode == 'SPQ':
        x_label = 'SPQ = log(s/Pq)'
    else:  # PQR
        x_label = 'PQR = log(Pq/Pr)'
    
    ax.set_xlabel(x_label, fontsize=14, fontweight='bold')
    ax.set_ylabel('SPR = log(s/Pr)', fontsize=14, fontweight='bold')
    
    sig_text = 'SIGNIFICANT' if mannwhitney_result['significant'] else 'NOT SIGNIFICANT'
    ax.set_title(f"Difference Heatmap ({x_mode} mode)\nMann-Whitney U P-value = {mannwhitney_result['p_value']:.6f} ({sig_text})",
                 fontsize=16, fontweight='bold', pad=20)
    
    # Add gridlines at parameter boundaries
    ax.set_xticks(np.arange(x_unique[0], x_unique[-1] + 1, 5))
    ax.set_yticks(np.arange(spr_unique[0], spr_unique[-1] + 1, 5))
    
    # Add minor gridlines
    ax.set_xticks(x_unique, minor=True)
    ax.set_yticks(spr_unique, minor=True)
    ax.grid(which='major', alpha=0.5, linestyle='-', linewidth=0.8)
    ax.grid(which='minor', alpha=0.2, linestyle=':', linewidth=0.5)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    file1_path = Path(file1_path)
    output_dir = file1_path.parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    heatmap_path = output_dir / f"mannwhitney_comparison_heatmap_{x_mode}_{timestamp}.png"
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    logging.info(f"Heatmap saved to: {heatmap_path}")
    
    plt.close()


def main(file1_path, file2_path):
    """
    Main execution function.
    
    Parameters:
    -----------
    file1_path : str
        Path to first results file
    file2_path : str
        Path to second results file
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
    logging.info("MANN-WHITNEY U TEST COMPARISON")
    logging.info("="*80)
    logging.info(f"Testing if distributions differ between two result files")
    logging.info(f"Null hypothesis: Distributions are the same")
    logging.info(f"Alternative: Distributions are different")
    logging.info("="*80)
    
    # Perform comparison
    merged_df, mannwhitney_result, x_mode = compare_results(file1_path, file2_path)
    
    if merged_df is None or mannwhitney_result is None:
        logging.error("Comparison failed")
        sys.exit(1)
    
    # Save results
    save_results(merged_df, mannwhitney_result, file1_path, file2_path, x_mode)
    
    # Create heatmap
    create_heatmap(merged_df, mannwhitney_result, file1_path, x_mode)
    
    logging.info("\n" + "="*80)
    logging.info("ANALYSIS COMPLETE")
    logging.info("="*80)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Error: Two file paths required")
        print("Usage: python compare_penalty_results.py <file1_path> <file2_path>")
        print("Example: python compare_penalty_results.py results_penalty_on.csv results_penalty_off.csv")
        sys.exit(1)
    
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    
    main(file1_path=file1, file2_path=file2)
