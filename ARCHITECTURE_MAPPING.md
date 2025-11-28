# Mapping New Architecture to Original Simulator Code

This document shows how the new object-oriented architecture maps to the original `ParametrizedSimulator_*.py` files.

## File Mapping

| New Architecture | Original Simulator |
|------------------|-------------------|
| `clone.py` | DataFrame rows in original `df` |
| `treatment.py` | `getProbabilities()` + `sampler()` functions |
| `tumour_simulation.py` | `BozicShort()` / `Bozic()` main functions |
| JSON config files | Hardcoded parameters in original |

## Key Function Mappings

### Clone Management

**Original (ParametrizedSimulatorMTDPlasticityTherapy.py):**
```python
# Lines 320-340: Clone represented as DataFrame row
dfcolumns = ["K", 'Id', "Parent", "ParentType", "N", "Ns", "Nr", "Nq", "T", ...]
ls = [1, 0, -1, "Sens", size0, size0, 0, 0, 0, ...]
df = pd.DataFrame(np.array(ls).reshape(1, len(ls)), columns=dfcolumns)
```

**New Architecture:**
```python
from clone import Clone, CloneCollection

clones = CloneCollection()
initial_clone = clones.add_clone(
    parent_id=-1,
    parent_type="Sens",
    generation=0,
    n_sensitive=size0,
    n_quasi=0,
    n_resistant=0
)
```

### Probability Calculation

**Original (ParametrizedSimulatorMTDPlasticityTherapy.py, lines 39-92):**
```python
def getProbabilities(idle, s, u, pq, pr, ps):
    klist = list(range(0, 50))
    pdlist = []
    pndlist = []
    pbpure = []
    # ... build probability table
    df = pd.DataFrame(...)
    return df
```

**New Architecture:**
```python
from treatment import Treatment

treatment = Treatment(config, base_params)
# Probability table computed in __init__
probs = treatment.calculate_probabilities(k=1, cell_type="S")
```

### Sampling Cell Fates

**Original (ParametrizedSimulatorMTDPlasticityTherapy.py, lines 93-173):**
```python
def sampler(row, probs, penalty=0, therapy=0, therapy2=0, penAmt=4, treatAmt=0.8, 
            drugtype="abs", conc1=1, conc2=1):
    k = row["K"]
    n = row["Ns"]  # Sensitive cells
    
    # Get base probabilities
    pp = probs[k+1]
    pnd, pb, pdd, pm, pq, pr, ps = pp["pnd"], pp["pb"], pp["pd"], ...
    
    # Apply treatment effects
    if therapy:
        if drugtype == "abs":
            pdd = pdd + (treatAmt - pdd) * conc1
            pb = 1 - pdd - pm - pq - pr - pnd
    
    # Sample fates
    snd, sb, sd, sm, sq, sr = list(np.int_(np.random.multinomial(n, [pnd, pb, pdd, pm, pq, pr])))
    
    # Return all outcomes
    return [k, row["id"], snd, sb, sd, sm, sq, sr, ...]
```

**New Architecture:**
```python
# In tumour_simulation.py, _simulate_cell_population() method
def _simulate_cell_population(self, clone, cell_type, n_cells):
    # Get probabilities (treatment effects automatically applied)
    probs = self.treatment.calculate_probabilities(clone.generation, cell_type)
    
    # Sample fates
    fate_probs = [probs['pnd'], probs['pb'], probs['pdd'], 
                  probs['pm'], probs['pq'], probs['pr']]
    fates = np.random.multinomial(n_cells, fate_probs)
    n_idle, n_birth, n_death, n_mutation, n_transition_q, n_transition_r = fates
    
    # Return deltas and new clones
    return deltas, new_clones
```

### Treatment Effects on Sensitive Cells

**Original (lines 106-122):**
```python
# Sensitive cells
n = row["Ns"]
pp = probs[k+1]
pnd, pb, pdd, pm, pq, pr, ps = pp["pnd"], pp["pb"], pp["pd"], ...

if therapy:
    if drugtype == "abs":
        pdd = pdd + (treatAmt - pdd) * conc1
        pb = 1 - pdd - pm - pq - pr - pnd
    elif drugtype == "prop":
        diff = min(((pb - pdd) * 3 * conc1), 0.9 * pb)
        pb = pb - diff
        pdd = pdd + diff

if therapy2:
    pnd = pnd + pq * conc2
    pq = pq - pq * conc2

snd, sb, sd, sm, sq, sr = list(np.int_(np.random.multinomial(n, [pnd, pb, pdd, pm, pq, pr])))
```

**New Architecture (treatment.py, lines 278-309):**
```python
def _apply_treatment_to_sensitive(self, probs, conc1, conc2):
    if not self.treatment_active:
        return probs
    
    # Primary therapy
    if self.drug_type == DrugType.ABSOLUTE:
        probs['pdd'] = probs['pdd'] + (self.treat_amt - probs['pdd']) * conc1
        probs['pb'] = 1 - probs['pdd'] - probs['pm'] - probs['pq'] - probs['pr'] - probs['pnd']
    
    elif self.drug_type == DrugType.PROPORTIONAL:
        diff = min(((probs['pb'] - probs['pdd']) * 3 * conc1), 0.9 * probs['pb'])
        probs['pb'] = probs['pb'] - diff
        probs['pdd'] = probs['pdd'] + diff
    
    # Secondary therapy
    if self.secondary_active and conc2 > 0:
        probs['pnd'] = probs['pnd'] + probs['pq'] * conc2
        probs['pq'] = probs['pq'] - probs['pq'] * conc2
    
    return probs
```

### Treatment Effects on Quasi-Resistant Cells

**Original (lines 131-151):**
```python
# Quasi cells
n = row["Nq"]
pp = probs[k+1]
pnd, pb, pdd, pm, pq, pr, ps = pp["pnd"], pp["pb"], pp["pd"], ...

if therapy:
    diff = (pb - pdd) * conc1
    pb = pb + diff / penAmt
    pdd = pdd - diff / penAmt

if therapy2:
    if thertype2 == "plast":
        pnd = pnd - 2 * pq * conc2
        pq = pq + 2 * pq * conc2
    else:
        red = min(pb - 0.01, pdd - 0.01, 0.1)
        pb = pb - red * conc2
        pdd = pdd - red * conc2
        pq = pq + 2 * red * conc2

qnd, qb, qd, qm, qq, qr = list(np.int_(np.random.multinomial(n, [pnd, pb, pdd, pm, pq, pr])))
```

**New Architecture (treatment.py, lines 311-342):**
```python
def _apply_treatment_to_quasi(self, probs, conc1, conc2):
    if not self.treatment_active:
        return probs
    
    # Primary therapy: partial protection
    diff = (probs['pb'] - probs['pdd']) * conc1
    probs['pb'] = probs['pb'] + diff / self.pen_amt
    probs['pdd'] = probs['pdd'] - diff / self.pen_amt
    
    # Secondary therapy
    if self.secondary_active and conc2 > 0:
        if self.secondary_therapy_type == "plast":
            probs['pnd'] = probs['pnd'] - 2 * probs['pq'] * conc2
            probs['pq'] = probs['pq'] + 2 * probs['pq'] * conc2
        else:
            red = min(probs['pb'] - 0.01, probs['pdd'] - 0.01, 0.1)
            probs['pb'] = probs['pb'] - red * conc2
            probs['pdd'] = probs['pdd'] - red * conc2
            probs['pq'] = probs['pq'] + 2 * red * conc2
    
    return probs
```

### Main Simulation Loop

**Original (ParametrizedSimulatorMTDPlasticityTherapy.py, lines 397-520):**
```python
def BozicShort(generations=500, init=1, s=0.1, u=0.000034, ...):
    # Initialize
    probs = getProbabilities(idle, s, u, pq, pr, ps)
    df = pd.DataFrame(...)  # Initial clone
    
    # Main loop
    for n in range(generations):
        # Remove extinct clones
        df = df.loc[df.loc[:, "N"] > 0, :].reset_index(drop=True)
        
        # Check extinction
        lastSize = df.N.sum()
        if lastSize == 0:
            state = 1
            return df, df, df, state
        
        # Update treatment state
        if drugstate == 1:
            if lastSize > 1000000000:
                onDrugs = True
                drugstate = 2
                conc1 = 1
        elif drugstate == 2:
            drugits = drugits + 1
            conc1 = max(0, 1 - drugits/24)
            if drugits >= 24:
                drugits = 0
        
        # Simulate generation
        df, maxid, ... = serialProcess(df, maxid, probs, n, ..., 
                                       onDrugs, onDrugs2, penAmt=penAmt, 
                                       treatAmt=treatAmt, drugtype=drugtype, 
                                       conc1=conc1, conc2=conc2)
        
        # Check relapse
        if lastSize > 4000000000:
            state = 2
            break
    
    return df, test2, test3, state
```

**New Architecture (tumour_simulation.py, lines 116-171):**
```python
def run(self):
    # Initialize (done in __init__)
    # self.clones already contains initial clone
    # self.treatment already configured
    
    # Main loop
    for generation in range(self.generations):
        # Remove extinct clones
        self.clones.remove_extinct_clones()
        
        # Check extinction
        tumor_size = self.clones.get_total_cells()
        if tumor_size == 0:
            self.state = 1
            break
        
        # Update treatment state (handles all scheduling logic)
        self.treatment.update_treatment_state(tumor_size, generation)
        
        # Simulate generation (samples all clones)
        self._simulate_generation()
        
        # Check relapse
        if self.treatment.relapsed:
            self.state = 2
            break
    
    # Generate and save results
    results = self._generate_results()
    self._save_results(results)
    
    return self.state, results
```

## Configuration Mapping

**Original Parameters (hardcoded or command-line):**
```python
# In original code
gens = 10000
size0 = 1
s0 = 0.01
u0 = 0.0000005
p = 0.0000005
r = 0.0000005
ps0 = 0.0000005
penAmt = 4
treatAmt = 0.8
drugtype = "abs"
therapytype = "mtd"
thertype2 = "plast"
```

**New JSON Configuration:**
```json
{
  "simulation": {
    "generations": 10000,
    "initial_size": 1
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
    "secondary_therapy": true,
    "secondary_therapy_type": "plast"
  }
}
```

## Output Format Mapping

**Original Output:**
```python
# DataFrame with columns
columns = ["K", 'Id', "Parent", "ParentType", "N", "Ns", "Nr", "Nq", "T"]
df.to_csv("output.csv", index=False)
```

**New Architecture Output:**
```python
# Same format - compatible!
results = simulation.run()
# Columns: K, Id, Parent, ParentType, N, Ns, Nr, Nq, T
# Automatically saved to: {prefix}_s{s}_u{u}_state{state}_{timestamp}.csv
```

## Key Improvements

1. **Modularity**: Each class has single responsibility
2. **Configuration**: JSON files instead of hardcoded parameters
3. **Extensibility**: Easy to add new drug types or schedules
4. **Clarity**: Object-oriented design is easier to understand
5. **Testability**: Each component can be tested independently
6. **Compatibility**: Output format matches original exactly

## Reproducing Original Results

To get identical results to original simulator:

1. **Use same seed:**
```python
import random, numpy as np
random.seed(your_seed)
np.random.seed(your_seed)
```

2. **Match parameters exactly:**
- All biological parameters (`s`, `u`, `pq`, `pr`, `ps`, `idle`)
- Treatment parameters (`treat_amt`, `pen_amt`, `dose_duration`)
- Thresholds (treatment start, stop, relapse sizes)

3. **Use equivalent schedule:**
- `therapytype="mtd"` → `"schedule_type": "mtd"`
- `therapytype="adt"` → `"schedule_type": "adaptive"`
- `treat=0` → `"schedule_type": "off"`

4. **Run simulation:**
```python
simulation = TumourSimulation("config.json")
state, results = simulation.run()
```

The results should match the original to within numerical precision.
