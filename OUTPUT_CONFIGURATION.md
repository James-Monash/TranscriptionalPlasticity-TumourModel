# Output Configuration Guide

## Overview

The simulation now supports flexible output options that let you control what files are generated. This is especially useful when running many replicates, as you can avoid generating hundreds of individual CSV files and instead get a single consolidated summary.

## Output Options

Add an `"output"` section to your JSON configuration under `"simulation"`:

```json
{
  "simulation": {
    "generations": 100000,
    "initial_size": 100,
    "number_of_replicates": 100,
    "output": {
      "save_individual_csvs": false,
      "save_summary_json": false,
      "save_consolidated_summary": true
    }
  }
}
```

### Output Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `save_individual_csvs` | boolean | `false` | Save detailed CSV file for each replicate with full clone data |
| `save_summary_json` | boolean | `false` | Save individual JSON summary file for each replicate |
| `save_consolidated_summary` | boolean | `true` | Save single CSV with one row per successful replicate |
| `output_columns` | array | `null` | List of column names to include in consolidated summary (null = all columns) |

## Consolidated Summary Output

The **consolidated summary** (default) creates a single CSV file containing one row for each successful simulation replicate. This is the **recommended output format** for most analyses.

### Output Filename Format

- Single condition: `{output_prefix}_consolidated_summary_{timestamp}.csv`
- Multiple conditions: `{condition_prefix}_consolidated_summary_{timestamp}.csv` (one file per condition)

### Columns in Consolidated Summary

The consolidated summary CSV includes:

**Metadata:**
- `condition_index`: Index of the condition (0-based)
- `replicate_number`: Replicate number within condition
- `final_generation`: Generation when simulation ended
- `final_state`: Outcome (`extinct`, `relapsed`, or `ongoing`)
- `initial_size`: Starting tumor size

**Biological Parameters:**
- `s`: Fitness advantage parameter
- `m`: Driver mutation rate
- `q`: Quasi-resistant transition rate
- `r`: Resistant transition rate  
- `l`: Reversion rate
- `idle`: Idle probability

**Final Cell Counts:**
- `final_total_cells`: Total cells at end
- `final_sensitive`: Sensitive cells
- `final_quasi`: Quasi-resistant cells
- `final_resistant`: Fully resistant cells
- `final_n_clones`: Number of distinct clones

**Fractions:**
- `fraction_sensitive`: Proportion of sensitive cells
- `fraction_quasi`: Proportion of quasi-resistant cells
- `fraction_resistant`: Proportion of resistant cells

**Treatment Information:**
- `treatment_schedule`: Treatment schedule type
- `drug_type`: Drug dosing type
- `doses_given`: Number of treatment doses administered
- `treatment_active_at_end`: Whether treatment was active when simulation ended

**Mutation Statistics:**
- `total_cell_events`: Total cell fate events simulated
- `total_driver_mutations`: Number of driver mutations that occurred
- `mutation_rate`: Proportion of events that were mutations

## Usage Examples

### Example 1: Minimal Output (Just Quasi Fraction)

Save only the quasi-resistant fraction for each replicate:

```json
{
  "simulation": {
    "number_of_replicates": 100,
    "output": {
      "save_consolidated_summary": true,
      "output_columns": ["fraction_quasi"]
    }
  }
}
```

**Result:** CSV with 100 rows, single column containing quasi-resistant fractions

### Example 2: Selected Key Metrics

Choose specific columns for your analysis:

```json
{
  "simulation": {
    "number_of_replicates": 100,
    "output": {
      "save_consolidated_summary": true,
      "output_columns": [
        "replicate_number",
        "final_state",
        "fraction_quasi",
        "fraction_resistant",
        "final_total_cells"
      ]
    }
  }
}
```

**Result:** CSV with 100 rows, 5 selected columns

### Example 3: All Columns (Default)

Get all available metrics:

```json
{
  "simulation": {
    "number_of_replicates": 100,
    "output": {
      "save_consolidated_summary": true
    }
  }
}
```

**Result:** CSV with 100 rows, ~25 columns with all available data

### Example 4: Detailed Clone Data

Save detailed clone information for each replicate plus consolidated summary:

```json
{
  "simulation": {
    "number_of_replicates": 10,
    "output": {
      "save_individual_csvs": true,
      "save_summary_json": false,
      "save_consolidated_summary": true,
      "output_columns": ["fraction_quasi", "fraction_resistant"]
    }
  }
}
```

**Result:** 
- 10 individual CSV files with full clone phylogenies
- 1 consolidated summary CSV with 10 rows, 2 columns

### Example 5: Everything

Save all output types:

```json
{
  "simulation": {
    "number_of_replicates": 5,
    "output": {
      "save_individual_csvs": true,
      "save_summary_json": true,
      "save_consolidated_summary": true
    }
  }
}
```

**Result:**
- 5 individual clone CSV files
- 5 individual JSON summaries  
- 1 consolidated summary CSV

### Example 6: Multiple Conditions

When running multiple conditions, each gets its own consolidated summary with custom columns:

```json
{
  "simulations": [
    {
      "simulation": {
        "output_prefix": "low_mutation",
        "number_of_replicates": 100,
        "output": {
          "save_consolidated_summary": true,
          "output_columns": ["fraction_quasi"]
        }
      },
      "biological_parameters": {
        "m": 1e-09,
        ...
      }
    },
    {
      "simulation": {
        "output_prefix": "high_mutation",
        "number_of_replicates": 100,
        "output": {
          "save_consolidated_summary": true,
          "output_columns": ["fraction_quasi"]
        }
      },
      "biological_parameters": {
        "m": 1e-07,
        ...
      }
    }
  ]
}
```

**Result:**
- `low_mutation_consolidated_summary_{timestamp}.csv` (100 rows, 1 column)
- `high_mutation_consolidated_summary_{timestamp}.csv` (100 rows, 1 column)

## Available Output Columns

When using `output_columns`, you can select from these available columns:

**Metadata:**
- `condition_index` - Index of the condition (0-based)
- `replicate_number` - Replicate number within condition
- `final_generation` - Generation when simulation ended
- `final_state` - Outcome (`extinct`, `relapsed`, or `ongoing`)
- `initial_size` - Starting tumor size

**Biological Parameters:**
- `s` - Fitness advantage parameter
- `m` - Driver mutation rate
- `q` - Quasi-resistant transition rate
- `r` - Resistant transition rate
- `l` - Reversion rate
- `idle` - Idle probability

**Cell Counts:**
- `final_total_cells` - Total cells at end
- `final_sensitive` - Sensitive cells
- `final_quasi` - Quasi-resistant cells
- `final_resistant` - Fully resistant cells
- `final_n_clones` - Number of distinct clones

**Cell Fractions (commonly used):**
- `fraction_sensitive` - Proportion of sensitive cells
- `fraction_quasi` - Proportion of quasi-resistant cells
- `fraction_resistant` - Proportion of resistant cells

**Treatment:**
- `treatment_schedule` - Treatment schedule type
- `drug_type` - Drug dosing type
- `doses_given` - Number of treatment doses
- `treatment_active_at_end` - Treatment status at end

**Statistics:**
- `total_cell_events` - Total cell fate events simulated
- `total_driver_mutations` - Number of driver mutations
- `mutation_rate` - Proportion of events that were mutations

**Common Usage:**
- Just quasi fraction: `["fraction_quasi"]`
- Cell fractions: `["fraction_sensitive", "fraction_quasi", "fraction_resistant"]`
- Basic metrics: `["final_state", "final_total_cells", "fraction_quasi"]`

## Analyzing Consolidated Output

The consolidated summary CSV can be easily loaded and analyzed:

### In Python (pandas):

```python
import pandas as pd

# Load consolidated summary
df = pd.read_csv('multi_replicate_consolidated_summary_20251206_120000.csv')

# Calculate statistics
print(f"Mean quasi-resistant fraction: {df['fraction_quasi'].mean():.4f}")
print(f"Median final tumor size: {df['final_total_cells'].median():.2e}")
print(f"Extinction rate: {(df['final_state'] == 'extinct').mean():.2%}")

# Group by condition if multiple conditions
if 'condition_index' in df.columns:
    summary = df.groupby('condition_index').agg({
        'fraction_quasi': ['mean', 'std'],
        'final_total_cells': 'median',
        'final_state': lambda x: (x == 'extinct').mean()
    })
    print(summary)
```

### In R:

```r
library(tidyverse)

# Load consolidated summary
df <- read_csv('multi_replicate_consolidated_summary_20251206_120000.csv')

# Calculate statistics
df %>%
  summarise(
    mean_quasi = mean(fraction_quasi),
    median_size = median(final_total_cells),
    extinction_rate = mean(final_state == 'extinct')
  )

# Visualize
ggplot(df, aes(x = fraction_quasi)) +
  geom_histogram(bins = 30) +
  labs(title = "Distribution of Quasi-Resistant Fractions",
       x = "Fraction Quasi-Resistant",
       y = "Count")
```

## Migration from Old Output Format

If you were previously using the simulation and getting individual files for each run:

**Old way:** Manually combine results from many CSV files
**New way:** Use `"save_consolidated_summary": true` to get all results in one file

The consolidated summary contains all the key information you need for analysis without the overhead of managing hundreds of individual files.

## Performance Considerations

- **Consolidated summary only** (recommended): Minimal disk I/O, fastest execution
- **Individual CSVs**: Significant disk I/O when running many replicates
- **All outputs**: Maximum disk usage, slowest execution

For large-scale parameter sweeps with 100+ replicates per condition, use consolidated summary only.
