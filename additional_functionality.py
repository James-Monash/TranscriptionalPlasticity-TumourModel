"""
Additional functionality for data processing and analysis.

Performs linear regression analysis on simulation data.
Calculates: log(rate + 1) = beta0 + beta1 * ratio + error
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from pathlib import Path
from sklearn.linear_model import LinearRegression
import re


def parse_column_name(col_name: str) -> Tuple[int, int]:
    """
    Parse column name to extract ratio and q_batch.
    
    Parameters:
    -----------
    col_name : str
        Column name like '5000_q1', '10000_q2', etc.
        
    Returns:
    --------
    tuple
        (ratio, q_batch) e.g., (5000, 1)
    """
    match = re.match(r'(\d+)_q(\d+)', col_name)
    if match:
        ratio = int(match.group(1))
        q_batch = int(match.group(2))
        return ratio, q_batch
    return None, None


def perform_linear_regression_for_csv(
    csv_path: str,
    output_csv: str = None
) -> pd.DataFrame:
    """
    Perform linear regression for each q batch in a CSV file.
    
    The model is: log(rate + 1) = beta0 + beta1 * ratio + error
    
    Parameters:
    -----------
    csv_path : str
        Path to input CSV file
    output_csv : str, optional
        Path to save regression results. If None, doesn't save.
        
    Returns:
    --------
    DataFrame
        Results with columns: q_batch, beta0, beta1, r_squared, n_samples
    """
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    # Parse s value from filename
    filename = Path(csv_path).stem
    s_match = re.search(r's=([\d.]+)', filename)
    s_value = s_match.group(1) if s_match else 'unknown'
    
    print(f"\nProcessing {csv_path} (s={s_value})")
    print("="*80)
    
    # Group columns by q_batch
    q_batches = {}
    for col in df.columns:
        ratio, q_batch = parse_column_name(col)
        if ratio is not None and q_batch is not None:
            if q_batch not in q_batches:
                q_batches[q_batch] = []
            q_batches[q_batch].append((ratio, col))
    
    # Sort each q_batch by ratio
    for q_batch in q_batches:
        q_batches[q_batch].sort(key=lambda x: x[0])
    
    results = []
    
    # Perform linear regression for each q_batch
    for q_batch in sorted(q_batches.keys()):
        ratio_col_pairs = q_batches[q_batch]
        
        # Prepare data for regression
        X = []  # ratios
        y = []  # log(rate + 1) values
        
        for ratio, col in ratio_col_pairs:
            rates = df[col].values
            # Apply transformation: log(rate + 1)
            log_rates = np.log(rates + 1)
            
            # Add all samples for this ratio
            for log_rate in log_rates:
                X.append(ratio)
                y.append(log_rate)
        
        X = np.array(X).reshape(-1, 1)
        y = np.array(y)
        
        # Perform linear regression
        model = LinearRegression()
        model.fit(X, y)
        
        beta0 = model.intercept_
        beta1 = model.coef_[0]
        r_squared = model.score(X, y)
        n_samples = len(y)
        
        results.append({
            's_value': s_value,
            'q_batch': q_batch,
            'beta0': beta0,
            'beta1': beta1,
            'r_squared': r_squared,
            'n_samples': n_samples
        })
        
        print(f"\nQ Batch {q_batch}:")
        print(f"  Equation: log(rate + 1) = {beta0:.6e} + {beta1:.6e} * ratio")
        print(f"  R² = {r_squared:.6f}")
        print(f"  Samples: {n_samples}")
        print(f"  Ratios used: {[ratio for ratio, _ in ratio_col_pairs]}")
    
    results_df = pd.DataFrame(results)
    
    if output_csv:
        results_df.to_csv(output_csv, index=False)
        print(f"\nRegression results saved to: {output_csv}")
    
    return results_df


def process_all_s_values(
    csv_directory: str = "./csv_data_for_linear_regression",
    output_file: str = "linear_regression_results.csv"
) -> pd.DataFrame:
    """
    Process all s value CSV files in a directory.
    
    Parameters:
    -----------
    csv_directory : str
        Directory containing CSV files named like s=0.001.csv, s=0.01.csv, etc.
    output_file : str
        Path to save combined regression results
        
    Returns:
    --------
    DataFrame
        Combined results from all files
    """
    csv_dir = Path(csv_directory)
    
    # Find all CSV files matching s=*.csv pattern
    csv_files = sorted(csv_dir.glob("s=*.csv"))
    
    if not csv_files:
        print(f"No CSV files matching 's=*.csv' found in {csv_directory}")
        return pd.DataFrame()
    
    print(f"Found {len(csv_files)} CSV file(s) to process")
    print("="*80)
    
    all_results = []
    
    for csv_file in csv_files:
        results_df = perform_linear_regression_for_csv(str(csv_file))
        all_results.append(results_df)
    
    # Combine all results
    combined_results = pd.concat(all_results, ignore_index=True)
    
    # Save combined results
    combined_results.to_csv(output_file, index=False)
    print("\n" + "="*80)
    print(f"Combined regression results saved to: {output_file}")
    print("="*80)
    
    return combined_results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python additional_functionality.py <csv_file_or_directory> [output_csv]")
        print("\nExamples:")
        print("  # Process a single CSV file")
        print("  python additional_functionality.py csv_data_for_linear_regression/s=0.001.csv results.csv")
        print("\n  # Process all s=*.csv files in a directory")
        print("  python additional_functionality.py csv_data_for_linear_regression")
        print("\n  # Process all files in current directory")
        print("  python additional_functionality.py .")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    path = Path(input_path)
    
    if path.is_file():
        # Single file processing
        if output_file is None:
            output_file = f"{path.stem}_regression_results.csv"
        perform_linear_regression_for_csv(str(path), output_file)
    elif path.is_dir():
        # Directory processing
        if output_file is None:
            output_file = "linear_regression_results.csv"
        process_all_s_values(str(path), output_file)
    else:
        print(f"Error: {input_path} is not a valid file or directory")
        sys.exit(1)
