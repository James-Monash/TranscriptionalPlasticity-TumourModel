# Architecture Diagram - Tumor Evolution Simulator

## Class Hierarchy and Relationships

```
┌─────────────────────────────────────────────────────────────────┐
│                      TumourSimulation                           │
│  (Main orchestrator - reads config, runs simulation, outputs)   │
│                                                                  │
│  Attributes:                                                     │
│  - config: Dict                                                  │
│  - clones: CloneCollection                                       │
│  - treatment: Treatment                                          │
│  - current_generation: int                                       │
│  - state: int (0=ongoing, 1=extinct, 2=relapsed)                │
│                                                                  │
│  Methods:                                                        │
│  - run() → (state, DataFrame)                                    │
│  - _simulate_generation()                                        │
│  - _simulate_cell_population(clone, type, n)                    │
│  - _apply_state_transitions(clone)                               │
│  - _generate_results() → DataFrame                               │
└────────────────┬──────────────────────┬─────────────────────────┘
                 │                      │
                 ↓                      ↓
    ┌────────────────────┐   ┌──────────────────────┐
    │  CloneCollection   │   │     Treatment        │
    │                    │   │                      │
    │  Attributes:       │   │  Attributes:         │
    │  - clones: Dict    │   │  - schedule_type     │
    │  - next_id: int    │   │  - drug_type         │
    │                    │   │  - treat_amt         │
    │  Methods:          │   │  - pen_amt           │
    │  - add_clone()     │   │  - prob_table        │
    │  - get_clone()     │   │  - treatment_active  │
    │  - remove_extinct()│   │  - doses_given       │
    │  - get_total_*()   │   │                      │
    │  - to_dataframe()  │   │  Methods:            │
    └────────┬───────────┘   │  - calculate_probs() │
             │               │  - update_state()    │
             ↓               │  - get_concentration()│
    ┌────────────────┐      │  - _apply_treatment_*│
    │     Clone      │      └──────────────────────┘
    │                │
    │  Attributes:   │
    │  - clone_id    │
    │  - parent_id   │
    │  - generation  │
    │  - n_sensitive │
    │  - n_quasi     │
    │  - n_resistant │
    │  - history     │
    │                │
    │  Methods:      │
    │  - update_counts()│
    │  - is_extinct()│
    │  - to_dict()   │
    └────────────────┘
```

## Data Flow During Simulation

```
[JSON Config File]
       │
       ↓
[TumourSimulation.__init__]
       │
       ├─→ Parse config
       ├─→ Create CloneCollection
       ├─→ Create Treatment (with prob table)
       └─→ Add initial Clone (1 sensitive cell)
       │
       ↓
[TumourSimulation.run()] ─────────────────┐
       │                                    │
       ↓                                    │
┌──────────────────────────────────────┐   │
│  FOR each generation:                │   │
│                                      │   │
│  1. Remove extinct clones            │   │
│     CloneCollection.remove_extinct() │   │
│                                      │   │
│  2. Check tumor extinction           │   │
│     CloneCollection.get_total_cells()│   │
│                                      │   │
│  3. Update treatment state           │   │
│     Treatment.update_treatment_state()│  │
│     │                                │   │
│     ├─→ Check tumor size vs threshold│   │
│     ├─→ Activate/deactivate drugs    │   │
│     ├─→ Update dose counter          │   │
│     └─→ Calculate concentrations     │   │
│                                      │   │
│  4. Simulate generation              │   │
│     TumourSimulation._simulate_gen() │   │
│     │                                │   │
│     FOR each clone:                  │   │
│     │                                │   │
│     ├─→ For sensitive cells:         │   │
│     │   │                            │   │
│     │   ├─→ Treatment.calculate_probs(k, "S")│
│     │   │   │                        │   │
│     │   │   ├─→ Get base probs       │   │
│     │   │   ├─→ Apply drug effects   │   │
│     │   │   └─→ Return modified probs│   │
│     │   │                            │   │
│     │   ├─→ np.random.multinomial()  │   │
│     │   │   (sample cell fates)      │   │
│     │   │                            │   │
│     │   ├─→ Clone.update_counts()    │   │
│     │   └─→ Create new clones if mutations│
│     │                                │   │
│     ├─→ For quasi cells:             │   │
│     │   [same process]               │   │
│     │                                │   │
│     ├─→ For resistant cells:         │   │
│     │   [same process]               │   │
│     │                                │   │
│     └─→ Apply state transitions      │   │
│         (S↔Q↔R using binomial)       │   │
│                                      │   │
│  5. Record history (if enabled)      │   │
│                                      │   │
│  6. Check for relapse                │   │
│     Treatment.relapsed               │   │
│                                      │   │
└──────────────────────────────────────┘   │
       │                                    │
       ↓                                    │
       BREAK if extinct or relapsed ───────┘
       │
       ↓
[Generate Results]
       │
       ├─→ CloneCollection.to_dataframe_rows()
       ├─→ Create pandas DataFrame
       └─→ Format columns: K, Id, Parent, ParentType, N, Ns, Nr, Nq, T
       │
       ↓
[Save Results]
       │
       ├─→ Save main CSV (clone data)
       ├─→ Save history CSV (if tracked)
       └─→ Save summary JSON (metadata)
       │
       ↓
[Return (state, DataFrame)]
```

## Treatment Effect Flow

```
Treatment.calculate_probabilities(k, cell_type)
       │
       ↓
Get base probabilities from prob_table[k]
       │
       ├─→ pnd (idle)
       ├─→ pb  (birth)
       ├─→ pdd (death)
       ├─→ pm  (mutation)
       ├─→ pq  (transition to Q or revert to S)
       ├─→ pr  (transition to R)
       └─→ ps  (reversion to S)
       │
       ↓
Get drug concentrations
       │
       ├─→ conc1 (primary drug)
       └─→ conc2 (secondary drug)
       │
       ↓
Apply treatment based on cell_type
       │
       ├─→ if cell_type == "S":
       │   │
       │   ├─→ Primary therapy:
       │   │   ├─→ if drug_type == "abs":
       │   │   │   pdd = pdd + (treat_amt - pdd) * conc1
       │   │   │   pb = 1 - pdd - pm - pq - pr - pnd
       │   │   │
       │   │   └─→ if drug_type == "prop":
       │   │       diff = min((pb-pdd)*3*conc1, 0.9*pb)
       │   │       pb -= diff, pdd += diff
       │   │
       │   └─→ Secondary therapy (if active):
       │       pnd += pq * conc2
       │       pq -= pq * conc2
       │
       ├─→ if cell_type == "Q":
       │   │
       │   ├─→ Primary therapy:
       │   │   diff = (pb - pdd) * conc1
       │   │   pb += diff / pen_amt
       │   │   pdd -= diff / pen_amt
       │   │
       │   └─→ Secondary therapy (if active):
       │       if type == "plast":
       │           pnd -= 2*pq*conc2
       │           pq += 2*pq*conc2
       │       else:
       │           red = min(pb-0.01, pdd-0.01, 0.1)
       │           pb -= red*conc2
       │           pdd -= red*conc2
       │           pq += 2*red*conc2
       │
       └─→ if cell_type == "R":
           │
           └─→ Primary therapy:
               diff = (pb - pdd) * conc1
               pb += diff / pen_amt
               pdd -= diff / pen_amt
               (no secondary therapy effect)
       │
       ↓
Return modified probabilities
```

## Cell Fate Sampling

```
For each cell population in a clone:

    n_cells (e.g., n_sensitive = 1000)
         │
         ↓
    Get probabilities
         │
         ├─→ pnd = 0.1
         ├─→ pb  = 0.5
         ├─→ pdd = 0.3
         ├─→ pm  = 0.00001
         ├─→ pq  = 0.00001
         └─→ pr  = 0.00001
         │
         ↓
    np.random.multinomial(n_cells, [pnd,pb,pdd,pm,pq,pr])
         │
         ↓
    [n_idle, n_birth, n_death, n_mutation, n_trans_q, n_trans_r]
         │
         ↓
    Calculate net change:
         net = n_birth - n_death - n_mutation
         │
         ↓
    Update clone.n_sensitive += net
         │
         ↓
    If n_mutation > 0:
         Create new Clone with n_sensitive = n_mutation
         │
         ↓
    If n_trans_q > 0:
         clone.n_sensitive -= n_trans_q
         clone.n_quasi += n_trans_q
         │
         ↓
    If n_trans_r > 0:
         clone.n_sensitive -= n_trans_r
         clone.n_resistant += n_trans_r
```

## JSON Config → Classes Mapping

```
JSON Config File
├── "simulation"
│   ├── "generations" ────────→ TumourSimulation.generations
│   ├── "initial_size" ───────→ Initial Clone.n_sensitive
│   ├── "output_dir" ─────────→ TumourSimulation.output_dir
│   └── "output_prefix" ──────→ TumourSimulation.output_prefix
│
├── "biological_parameters"
│   ├── "s" ──────────────────→ Treatment.base_s
│   ├── "u" ──────────────────→ Treatment.base_u
│   ├── "pq" ─────────────────→ Treatment.base_pq
│   ├── "pr" ─────────────────→ Treatment.base_pr
│   ├── "ps" ─────────────────→ Treatment.base_ps
│   └── "idle" ───────────────→ Treatment.base_idle
│
└── "treatment"
    ├── "schedule_type" ──────→ Treatment.schedule_type
    ├── "drug_type" ──────────→ Treatment.drug_type
    ├── "treat_amt" ──────────→ Treatment.treat_amt
    ├── "pen_amt" ────────────→ Treatment.pen_amt
    ├── "dose_duration" ──────→ Treatment.dose_duration
    ├── "treatment_start_size"→ Treatment.treatment_start_size
    ├── "treatment_stop_size"─→ Treatment.treatment_stop_size
    ├── "relapse_size" ───────→ Treatment.relapse_size
    ├── "penalty" ────────────→ Treatment.penalty_enabled
    ├── "secondary_therapy" ──→ Treatment.secondary_therapy_enabled
    └── "secondary_therapy_type"→ Treatment.secondary_therapy_type
```

## Output Files Generated

```
Simulation Run
       │
       ↓
Output Directory (e.g., "./output/")
       │
       ├── {prefix}_s{s}_u{u}_state{state}_{timestamp}.csv
       │   (Main results: clone information)
       │   Columns: K, Id, Parent, ParentType, N, Ns, Nr, Nq, T
       │
       ├── {prefix}_history_{timestamp}.csv  (if track_history=true)
       │   (Tumor dynamics every 10 generations)
       │   Columns: generation, total_cells, sensitive, quasi, 
       │            resistant, n_clones, treatment_active, doses_given
       │
       └── {prefix}_summary_{timestamp}.json
           (Metadata and final statistics)
           {config_file, final_generation, final_state, final_cells,
            doses_given, treatment_schedule, drug_type, ...}
```

## State Machine for Treatment Schedules

```
MTD Schedule:
┌─────────────┐
│   Initial   │ (state=0, waiting)
│  No drugs   │
└──────┬──────┘
       │ tumor_size > treatment_start_size
       ↓
┌─────────────┐
│  Treating   │ (state=2, active)
│ Cycling 24  │ drug_iterations: 0→24→0→24...
│ generations │ conc1: 1.0 → 0.0 (linear decay)
└──────┬──────┘
       │ tumor_size > relapse_size
       ↓
┌─────────────┐
│  Relapsed   │ (state=5, failed)
│  Stop sim   │
└─────────────┘

Adaptive Schedule:
┌─────────────┐
│   Initial   │
│  No drugs   │
└──────┬──────┘
       │ tumor_size > treatment_start_size
       ↓
┌─────────────┐
│  Treating   │ ←──────────────┐
│ Cycling 24  │                │
│ generations │                │
└──────┬──────┘                │
       │                       │
       │ Complete cycle        │
       ├──→ if size > relapse_size → STOP (relapsed)
       ├──→ if size > stop_size → continue treating
       └──→ if size < stop_size ↓
                                │
           ┌─────────────┐      │
           │   Paused    │      │
           │  No drugs   │      │
           └──────┬──────┘      │
                  │             │
                  │ size > treatment_start_size
                  └─────────────┘
```

## Class Interactions Example

```python
# User creates simulation
sim = TumourSimulation("config.json")

# Internally:
sim.clones = CloneCollection()
  └─→ Initial clone created (ID=0, n_sensitive=1)

sim.treatment = Treatment(config, base_params)
  └─→ Probability table computed
  └─→ Treatment state initialized

# User runs simulation
state, results = sim.run()

# Each generation:
for gen in range(generations):
    # Update treatment
    sim.treatment.update_treatment_state(tumor_size, gen)
    
    # For each clone
    for clone in sim.clones:
        # Get probabilities (treatment effects applied)
        probs_s = sim.treatment.calculate_probabilities(clone.generation, "S")
        probs_q = sim.treatment.calculate_probabilities(clone.generation, "Q")
        probs_r = sim.treatment.calculate_probabilities(clone.generation, "R")
        
        # Sample fates
        fates_s = multinomial(clone.n_sensitive, probs_s)
        fates_q = multinomial(clone.n_quasi, probs_q)
        fates_r = multinomial(clone.n_resistant, probs_r)
        
        # Update clone
        clone.update_counts(...)
        
        # Create new clones from mutations
        if mutations_occurred:
            new_clone = sim.clones.add_clone(...)

# Generate output
results = sim._generate_results()
  └─→ Converts CloneCollection to DataFrame
  └─→ Saves CSV files
```

This architecture provides clean separation of concerns while maintaining compatibility with the original simulator behavior.
