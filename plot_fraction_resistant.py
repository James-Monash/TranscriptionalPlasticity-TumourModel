import pandas as pd
import matplotlib.pyplot as plt
import sys
import seaborn as sns
import numpy as np
from scipy import stats

def plot_density(csv_file1, csv_file2):
    """
    Read two CSV files and create overlaid density plots of fraction_resistant values.
    
    Args:
        csv_file1: Path to the first CSV file
        csv_file2: Path to the second CSV file
    """
    # Read the CSV files
    df1 = pd.read_csv(csv_file1)
    df2 = pd.read_csv(csv_file2)
    
    # Extract fraction_resistant columns
    fraction_resistant1 = df1['fraction_resistant']
    fraction_resistant2 = df2['fraction_resistant']
    
    # Create the plot
    plt.figure(figsize=(12, 7))
    
    # Determine common range for both datasets
    all_data = pd.concat([fraction_resistant1, fraction_resistant2])
    min_val = all_data.min()
    max_val = all_data.max()
    
    # Use finer log-spaced bins for better visualization
    bins = np.logspace(np.log10(min_val), np.log10(max_val), 50)
    
    # Plot first dataset
    plt.hist(fraction_resistant1, bins=bins, density=True, alpha=0.5, 
             color='blue', edgecolor='black', label=f'Dataset 1: {csv_file1.split("/")[-1].split("\\")[-1]}')
    
    # Add KDE for first dataset
    log_data1 = np.log10(fraction_resistant1)
    kde1 = stats.gaussian_kde(log_data1)
    x_range = np.logspace(np.log10(min_val), np.log10(max_val), 2000)
    kde_values1 = kde1(np.log10(x_range)) / (x_range * np.log(10))
    plt.plot(x_range, kde_values1, color='darkblue', linewidth=2.5, label='Density Curve 1')
    
    # Plot second dataset
    plt.hist(fraction_resistant2, bins=bins, density=True, alpha=0.5, 
             color='red', edgecolor='black', label=f'Dataset 2: {csv_file2.split("/")[-1].split("\\")[-1]}')
    
    # Add KDE for second dataset
    log_data2 = np.log10(fraction_resistant2)
    kde2 = stats.gaussian_kde(log_data2)
    kde_values2 = kde2(np.log10(x_range)) / (x_range * np.log(10))
    plt.plot(x_range, kde_values2, color='darkred', linewidth=2.5, label='Density Curve 2')
    
    # Set log scale on x-axis
    plt.xscale('log')
    
    # Customize the plot
    plt.xlabel('Fraction Resistant (log scale)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Density Distribution Comparison of Fraction Resistant', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Show statistics for both datasets
    print(f"Statistics for Dataset 1 ({csv_file1}):")
    print(f"Mean: {fraction_resistant1.mean():.6e}")
    print(f"Median: {fraction_resistant1.median():.6e}")
    print(f"Std Dev: {fraction_resistant1.std():.6e}")
    print(f"Min: {fraction_resistant1.min():.6e}")
    print(f"Max: {fraction_resistant1.max():.6e}")
    
    print(f"\nStatistics for Dataset 2 ({csv_file2}):")
    print(f"Mean: {fraction_resistant2.mean():.6e}")
    print(f"Median: {fraction_resistant2.median():.6e}")
    print(f"Std Dev: {fraction_resistant2.std():.6e}")
    print(f"Min: {fraction_resistant2.min():.6e}")
    print(f"Max: {fraction_resistant2.max():.6e}")
    
    # Save the plot
    output_file = 'fraction_resistant_comparison_density_plot.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as: {output_file}")
    
    # Display the plot
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python plot_fraction_resistant.py <csv_file1> <csv_file2>")
        sys.exit(1)
    
    csv_file1 = sys.argv[1]
    csv_file2 = sys.argv[2]
    plot_density(csv_file1, csv_file2)