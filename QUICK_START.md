# Quick Start Guide - Tumor Evolution Simulator

## Installation

No installation needed! Just ensure you have the required dependencies:

```bash
pip install numpy pandas
```

## File Structure

```
Code For Specific Scenario Simulation/
├── clone.py                    # Clone class definition
├── treatment.py                # Treatment class definition
├── tumour_simulation.py        # Main simulation orchestrator
├── test_simulation.py          # Test suite
├── example_configs/            # Example JSON configurations
│   ├── mtd_absolute.json
│   ├── adaptive_therapy.json
│   ├── combination_therapy.json
│   ├── proportional_drug.json
│   └── no_treatment.json
├── SIMULATOR_README.md         # Full documentation
├── ARCHITECTURE_MAPPING.md     # Maps to original code
└── QUICK_START.md             # This file
```

## Running Your First Simulation

### 1. Basic Usage (Command Line)

```bash
python tumour_simulation.py example_configs/mtd_absolute.json
```

### 2. Basic Usage (Python Script)

```python
from tumour_simulation import TumourSimulation

# Run simulation
sim = TumourSimulation("example_configs/mtd_absolute.json")
state, results = sim.run()

# View results
print(f"Final state: {state}")
print(f"Total cells: {results['N'].sum()}")
print(f"Number of clones: {len(results)}")
```

### 3. With Custom Configuration

```python
import json
from tumour_simulation import TumourSimulation

# Create custom config
config = {
    "use_multiprocessing": True,
    "simulation": {
        "generations": 5000,
        "initial_size": 1,
        "initial_quasi": 0,
        "initial_resistant": 0,
        "output_dir": "./my_results",
        "output_prefix": "my_simulation",
        "seed": 42,
        "enable_live_plot": False,
        "number_of_replicates": 1
    },
    "biological_parameters": {
        "s": 0.01,
        "m": 0.0000005,
        "q": 0.0000005,
        "r": 0.0000005,
        "l": 0.0000005,
        "idle": 0.1
    },
    "treatment": {
        "schedule_type": "mtd",
        "drug_type": "abs",
        "treat_amt": 0.8,
        "pen_amt": 4.0,
        "dose_duration": 24,
        "treatment_start_size": 1e9,
        "treatment_stop_size": 1e9,
        "relapse_size": 4e9,
        "penalty": False,
        "secondary_therapy": False,
        "secondary_therapy_type": "plast"
    }
}

# Save config
with open("my_config.json", 'w') as f:
    json.dump(config, f, indent=2)

# Run simulation
sim = TumourSimulation("my_config.json")
state, results = sim.run()
```

## Setting Random Seeds for Reproducibility

The simulator now supports built-in seed configuration:

```json
{
  "simulation": {
    "seed": 42,
    "number_of_replicates": 1
  }
}
```

For programmatic use:

```python
import random
import numpy as np

# Set seeds BEFORE creating simulation
random.seed(42)
np.random.seed(42)

# Now run simulation
sim = TumourSimulation("config.json")
state, results = sim.run()
```

## Visualization

Generate population dynamics plots:

```json
{
  "simulation": {
    "enable_live_plot": true
  }
}
```

This creates a plot at the end showing:
- Total population (blue)
- Quasi-resistant cells (orange)
- Resistant cells (red)

## Starting with Pre-existing Resistance

Simulate tumors that already have resistant cells:

```json
{
  "simulation": {
    "initial_size": 4000000000,
    "initial_quasi": 30000,
    "initial_resistant": 0
  }
}
```

This is useful for:
- Testing treatment on established tumors
- Modeling relapse scenarios
- Studying competition between cell types

## Common Use Cases

### MTD Treatment (Maximum Tolerated Dose)

```json
{
  "treatment": {
    "schedule_type": "mtd",
    "drug_type": "abs",
    "treat_amt": 0.8,
    "pen_amt": 4.0,
    "dose_duration": 24,
    "treatment_start_size": 1000000000,
    "relapse_size": 4000000000
  }
}
```

### Adaptive Therapy

```json
{
  "treatment": {
    "schedule_type": "adaptive",
    "drug_type": "abs",
    "treat_amt": 0.8,
    "pen_amt": 4.0,
    "dose_duration": 24,
    "treatment_start_size": 1000000000,
    "treatment_stop_size": 1000000000,
    "relapse_size": 4000000000
  }
}
```

### Combination Therapy (Primary + Plasticity Drug)

```json
{
  "treatment": {
    "schedule_type": "mtd",
    "drug_type": "abs",
    "treat_amt": 0.8,
    "pen_amt": 4.0,
    "secondary_therapy": true,
    "secondary_therapy_type": "plast"
  }
}
```

### No Treatment (Baseline)

```json
{
  "treatment": {
    "schedule_type": "off",
    "penalty": false
  }
}
```

## Reading Results

### Load Results CSV

```python
import pandas as pd

results = pd.read_csv("output/my_simulation_s0.01_m5e-07_state2_20251128_123456.csv")

print(f"Total cells: {results['N'].sum()}")
print(f"Sensitive cells: {results['Ns'].sum()}")
print(f"Quasi-resistant: {results['Nq'].sum()}")
print(f"Resistant cells: {results['Nr'].sum()}")
print(f"Number of clones: {len(results)}")
```

### Load Summary JSON

```python
import json

with open("output/my_simulation_summary_20251128_123456.json") as f:
    summary = json.load(f)

print(f"Final state: {summary['final_state']}")
print(f"Doses given: {summary['doses_given']}")
print(f"Final cells: {summary['final_total_cells']}")
```

## Accessing Simulation State Programmatically

```python
from tumour_simulation import TumourSimulation

sim = TumourSimulation("config.json")

# Run for a few generations
for i in range(100):
    sim._simulate_generation()
    
    # Get current state
    summary = sim.get_summary()
    print(f"Gen {i}: Total={summary['total_cells']}, "
          f"S={summary['sensitive_cells']}, "
          f"Q={summary['quasi_cells']}, "
          f"R={summary['resistant_cells']}")
    
    # Check extinction
    if sim.clones.get_total_cells() == 0:
        print("Tumor extinct!")
        break
```

## Running Multiple Simulations (Parameter Sweep)

```python
import json
from tumour_simulation import TumourSimulation
import random
import numpy as np

# Parameter values to test
treat_amts = [0.7, 0.8, 0.9]
pen_amts = [2, 3, 4]

results_summary = []

for treat_amt in treat_amts:
    for pen_amt in pen_amts:
        # Set seed for reproducibility
        seed = hash((treat_amt, pen_amt)) % 10000
        random.seed(seed)
        np.random.seed(seed)
        
        # Create config
        config = {
            "simulation": {
                "generations": 1000,
                "initial_size": 1,
                "output_dir": f"./sweep_results",
                "output_prefix": f"treat{treat_amt}_pen{pen_amt}"
            },
            "biological_parameters": {
                "s": 0.01,
                "m": 0.0000005,
                "q": 0.0000005,
                "r": 0.0000005,
                "l": 0.0000005,
                "idle": 0.1
            },
            "treatment": {
                "schedule_type": "mtd",
                "drug_type": "abs",
                "treat_amt": treat_amt,
                "pen_amt": pen_amt,
                "dose_duration": 24,
                "treatment_start_size": 1e7,
                "relapse_size": 1e9,
                "penalty": False,
                "secondary_therapy": False
            }
        }
        
        config_file = f"temp_config_{treat_amt}_{pen_amt}.json"
        with open(config_file, 'w') as f:
            json.dump(config, f)
        
        # Run simulation
        sim = TumourSimulation(config_file)
        state, results = sim.run()
        
        # Store summary
        summary = sim.get_summary()
        summary['treat_amt'] = treat_amt
        summary['pen_amt'] = pen_amt
        results_summary.append(summary)
        
        print(f"Completed: treat_amt={treat_amt}, pen_amt={pen_amt}, "
              f"state={state}, cells={summary['total_cells']}")

# Analyze results
import pandas as pd
summary_df = pd.DataFrame(results_summary)
print("\nParameter Sweep Results:")
print(summary_df[['treat_amt', 'pen_amt', 'state', 'total_cells', 'n_clones']])
```

## Running Tests

```bash
python test_simulation.py
```

This will run 5 test cases and create output in `./test_output/`.

## Troubleshooting

### Simulation runs too long
- Reduce `generations` parameter
- Increase size thresholds for treatment
- Check if tumor is growing explosively

### No treatment effects observed
- Verify `schedule_type` is not `"off"`
- Check `treatment_start_size` threshold is reached
- Confirm `treat_amt` and `pen_amt` are set
- Set `track_detailed_history` to True to view how the tumour is growing

### Different results each run
- Set random seeds before running simulation
- Use same seed value for reproducibility

### Output files not created
- Check `output_dir` exists or will be created
- Verify write permissions
- Check disk space

## Next Steps

1. **Read full documentation**: See `SIMULATOR_README.md`
2. **Understand architecture**: See `ARCHITECTURE_MAPPING.md`
3. **Try example configs**: Run all files in `example_configs/`
4. **Create custom treatments**: Modify JSON configurations
5. **Extend the code**: Add new drug types or schedules
