# Tumor Evolution Simulator - JSON-Driven Architecture

A modular, object-oriented tumor evolution simulator that uses JSON configuration files to specify treatment protocols and biological parameters.

## Overview

This simulator models tumor evolution using a discrete-time branching process with three cell types:
- **Sensitive (S)**: Drug-sensitive cells
- **Quasi-resistant (Q)**: Phenotypically plastic, partially resistant cells
- **Resistant (R)**: Genetically mutated cells with permanent resistance

## Architecture

The simulator consists of three main classes:

### 1. `Clone` (clone.py)
Stores information about clonal populations with identical driver mutations.

**Features:**
- Tracks cell counts by type (S, Q, R)
- Records lineage information (parent clone, generation)
- Maintains optional history for detailed analysis
- Provides extinction checking and data export

**Key Methods:**
- `update_counts()`: Update cell populations
- `to_dict()`: Export clone data
- `is_extinct()`: Check if clone has died out

### 2. `Treatment` (treatment.py)
Manages drug treatment schedules and calculates branching probabilities.

**Features:**
- Supports multiple treatment schedules (MTD, adaptive, continuous, etc.)
- Implements different drug types (absolute, proportional)
- Handles primary and secondary (plasticity) therapies
- Calculates cell-type-specific probabilities under treatment

**Key Methods:**
- `update_treatment_state()`: Update treatment based on tumor size
- `get_drug_concentration()`: Calculate current drug concentrations
- `calculate_probabilities()`: Get branching probabilities for each cell type

**Supported Schedules:**
- `continuous`: Constant treatment
- `mtd`: Maximum Tolerated Dose (cycling)
- `adaptive`: Size-triggered treatment on/off
- `off`: No treatment

**Drug Types:**
- `abs`: Absolute effect (sets death rate)
- `prop`: Proportional effect (based on growth rate)

### 3. `TumourSimulation` (tumour_simulation.py)
Orchestrates the complete simulation process.

**Features:**
- Reads JSON configuration files
- Initializes clones and treatment protocol
- Executes branching process for specified generations
- Tracks tumor dynamics and treatment response
- Outputs results to CSV files

**Key Methods:**
- `run()`: Execute complete simulation
- `get_summary()`: Get current simulation state
- `_simulate_generation()`: Execute one timestep

## Usage

### Basic Usage

```python
from tumour_simulation import TumourSimulation

# Run simulation from config file
simulation = TumourSimulation("example_configs/mtd_absolute.json")
state, results = simulation.run()

print(f"Final state: {state}")
print(f"Final tumor size: {results['N'].sum()}")
```

### Command Line

```bash
python tumour_simulation.py example_configs/mtd_absolute.json
```

### Programmatic Usage

```python
from tumour_simulation import run_simulation_from_config

# Quick simulation run
state, results = run_simulation_from_config("config.json")
```

## Configuration File Format

JSON configuration files specify all simulation parameters:

```json
{
  "simulation": {
    "generations": 10000,
    "initial_size": 1,
    "output_dir": "./output",
    "output_prefix": "simulation_name",
    "track_history": true
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
    "treatment_start_size": 1000000000,
    "treatment_stop_size": 1000000000,
    "relapse_size": 4000000000,
    "penalty": false,
    "secondary_therapy": false,
    "secondary_therapy_type": "plast"
  }
}
```

### Configuration Parameters

#### Simulation Section
- `generations`: Number of timesteps to simulate
- `initial_size`: Initial tumor size (typically 1)
- `output_dir`: Directory for output files
- `output_prefix`: Prefix for output filenames
- `track_history`: Whether to record detailed history (every 10 generations)

#### Biological Parameters
- `s`: Selective advantage per driver mutation
- `u`: Mutation rate (driver mutations per cell division)
- `pq`: Transition rate to quasi-resistant state
- `pr`: Transition rate to resistant state
- `ps`: Reversion rate from quasi to sensitive
- `idle`: Baseline idle/quiescence probability

#### Treatment Parameters
- `schedule_type`: Treatment schedule (`"mtd"`, `"adaptive"`, `"continuous"`, `"off"`)
- `drug_type`: Drug mechanism (`"abs"` or `"prop"`)
- `treat_amt`: Treatment efficacy (0-1, typically 0.7-0.9)
- `pen_amt`: Resistance penalty factor (2-6, typically 3-4)
- `dose_duration`: Duration of each dose in generations (typically 24)
- `treatment_start_size`: Tumor size to begin treatment (e.g., 1e9)
- `treatment_stop_size`: Tumor size to pause adaptive therapy (e.g., 1e9)
- `relapse_size`: Tumor size considered relapse/failure (e.g., 4e9)
- `penalty`: Enable fitness cost for resistant cells without treatment
- `secondary_therapy`: Enable plasticity-modulating secondary therapy
- `secondary_therapy_type`: Type of secondary therapy (`"plast"`)

## Example Configurations

Five example configurations are provided in `example_configs/`:

1. **`mtd_absolute.json`**: MTD schedule with absolute drug
2. **`adaptive_therapy.json`**: Adaptive therapy (size-triggered)
3. **`combination_therapy.json`**: MTD with plasticity therapy
4. **`proportional_drug.json`**: Proportional drug with penalty
5. **`no_treatment.json`**: No treatment baseline

## Output Files

Each simulation generates three output files:

### 1. Main Results CSV
Format: `{prefix}_s{s}_u{u}_state{state}_{timestamp}.csv`

Columns:
- `K`: Number of driver mutations (generation of emergence)
- `Id`: Unique clone identifier
- `Parent`: Parent clone ID
- `ParentType`: Cell type of parent ("S", "Q", "R")
- `N`: Total cells in clone
- `Ns`: Sensitive cells
- `Nr`: Resistant cells
- `Nq`: Quasi-resistant cells
- `T`: Final generation

### 2. History CSV (if track_history=true)
Format: `{prefix}_history_{timestamp}.csv`

Tracks tumor dynamics every 10 generations.

### 3. Summary JSON
Format: `{prefix}_summary_{timestamp}.json`

Contains metadata and final statistics.

## Reproducing Results from Original Simulators

To match results from the original `ParametrizedSimulator_*.py` files:

1. **Set random seeds** (you mentioned you'll handle this):
```python
import random
import numpy as np

random.seed(your_seed)
np.random.seed(your_seed)
```

2. **Use equivalent parameters**:
   - Match `s`, `u`, `pq`, `pr`, `ps`, `idle` values
   - Use same `treat_amt` and `pen_amt`
   - Set appropriate thresholds for treatment

3. **Select correct schedule**:
   - MTD → `"mtd"`
   - Adaptive → `"adaptive"`
   - No treatment → `"off"`

4. **Match drug type**:
   - Original uses `drugtype="abs"` → `"abs"`
   - For proportional → `"prop"`

## Implementation Details

### Branching Process
Each generation, for each clone:
1. Get probabilities based on treatment state and cell type
2. Sample cell fates using multinomial distribution
3. Update cell counts (births, deaths, mutations, transitions)
4. Create new clones from mutations
5. Apply phenotypic transitions (S↔Q↔R)

### Treatment Effects
The simulator implements the exact drug effects from the original code:

**Sensitive cells (absolute drug):**
```python
pdd = pdd + (treat_amt - pdd) * concentration
pb = 1 - pdd - pm - pq - pr - pnd
```

**Quasi/Resistant cells:**
```python
diff = (pb - pdd) * concentration
pb = pb + diff / pen_amt
pdd = pdd - diff / pen_amt
```

### Differences from Original
- Simplified tracking (no detailed history arrays by default)
- Cleaner object-oriented design
- JSON configuration instead of hardcoded parameters
- More modular and extensible architecture

## Extending the Simulator

### Adding New Treatment Schedules
Add to `treatment.py`:
```python
class ScheduleType(Enum):
    YOUR_SCHEDULE = "your_schedule"

# In Treatment.update_treatment_state():
elif self.schedule_type == ScheduleType.YOUR_SCHEDULE:
    # Implement scheduling logic
```

### Adding New Drug Types
Add to `treatment.py`:
```python
class DrugType(Enum):
    YOUR_DRUG = "your_drug"

# Implement _apply_treatment_to_X methods for new drug
```

### Custom Analysis
Access clone data programmatically:
```python
simulation = TumourSimulation("config.json")
state, results = simulation.run()

# Access final clones
for clone in simulation.clones:
    print(f"Clone {clone.clone_id}: S={clone.n_sensitive}, "
          f"Q={clone.n_quasi}, R={clone.n_resistant}")

# Access treatment state
treatment_summary = simulation.treatment.get_state_summary()
print(f"Doses given: {treatment_summary['doses_given']}")
```

## Dependencies

- numpy
- pandas
- Python 3.7+

## Testing

To verify the implementation matches original results:

1. Use the same seed:
```python
import random, numpy as np
random.seed(42)
np.random.seed(42)
```

2. Run equivalent configuration
3. Compare final clone distributions and tumor sizes

Expected minor differences due to:
- Rounding in probability calculations
- Order of operations in state transitions

## References

Based on the branching process model from the original `ParametrizedSimulator_*.py` files in this repository. See `DRUG_EFFECTS_DICTIONARY.md` for detailed documentation of drug effects and treatment strategies.

## License

[Your license here]

## Contact

[Your contact information]
