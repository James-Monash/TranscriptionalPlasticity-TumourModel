# Bootstrap Parameter Space Manager

Comprehensive parameter space exploration using Latin Hypercube Sampling with bootstrap confidence intervals.

## Features

- **Latin Hypercube Sampling**: Efficiently samples 6D parameter space
- **Adaptive Sampling**: Adds more samples if time permits (up to 50 at a time)
- **Checkpointing**: Resumes from where it left off if interrupted
- **Parallel Execution**: Uses all CPU cores for simulations
- **Bootstrap Analysis**: Calculates 95% CI for 50th, 75th, 90th percentiles
- **Adaptive Bootstrap**: Increases bootstrap samples (100-1000) until precision targets met
- **Dual Metrics**: 
  - Total resistant fraction (transient + genetic)
  - Transient fraction of resistant cells
- **Emulator Building**: Generates training data for surrogate models

## Parameter Ranges

- **s** (selection coefficient): [0.001, 0.1] (log-uniform)
- **m** (driver mutation rate): [1e-8, 1e-5] (log-uniform)
- **q** (S→Q transition): [1e-11, 1e-2] (log-uniform)
- **r** (S→R transition): [1e-11, 1e-2] (log-uniform)
- **l** (Q→S transition): [1e-11, 1e-2] (log-uniform)
- **idle** (idle probability): [0.0, 0.5] (uniform)

## Building

In your existing build directory:
```bash
cmake --build . --config Release
```

This builds `bootstrap_manager` alongside your other executables.

## Usage

Run with default 10-hour budget:
```bash
./bootstrap_manager
```

Specify custom time budget (in hours):
```bash
./bootstrap_manager 5.0
```

## How It Works

1. **Initial Sampling**: Generates 50 LHS samples to start (conservative estimate for 10 hours)
2. **Timing Calibration**: Tracks how long each parameter set takes
3. **Sequential Processing**: 
   - For each parameter set:
     - Runs N simulations (100-300 based on s value)
     - Collects two metrics per simulation
     - Performs bootstrap resampling (100 samples)
     - Calculates CI for both metrics
4. **Adaptive Expansion**: After 10 samples, then every 20 samples, checks if time permits more
   - Estimates remaining capacity with 20% safety margin
   - Adds up to 50 new samples at a time
5. **Checkpointing**: Saves progress after each sample
6. **Emulator Data**: Builds training dataset for surrogate modeling

## Output Files

- `bootstrap_checkpoint.json`: Progress checkpoint (auto-resume)
- `bootstrap_results/sample_N.json`: Individual sample results
- `emulator_training_data.json`: Complete training dataset

## Resume Capability

If interrupted, simply re-run:
```bash
./bootstrap_manager
```

It will:
- Load the checkpoint
- Skip completed samples
- Continue from where it stopped
- Add adaptive samples based on remaining time

## Simulation Count Strategy

**Fixed at 500 simulations per parameter set** regardless of s value.

Rationale:
- Smaller s values create more volatile distributions requiring adequate sampling
- Larger s values also benefit from consistent sample size for comparison
- Fixed count simplifies time estimation and ensures statistical consistency

**Bootstrap samples**: Start at 100, increase by 100 up to 1000 until precision targets met:
- 50th & 75th percentiles: CI width ≤ 2 percentage points
- 90th percentile: CI width ≤ 3 percentage points

## Metrics Calculated

For each parameter set, bootstrap CIs for:

1. **Total Resistant Fraction**: `fraction_transient + fraction_resistant`
2. **Transient Proportion**: `fraction_transient / (fraction_transient + fraction_resistant)`

Each metric gets 50th, 75th, and 90th percentile CIs.

## Expected Coverage

With 10-hour budget and 500 simulations per parameter set:
- **Fast parameter sets** (s ≥ 0.05): ~2-3 min/sample → ~150-200 samples
- **Medium parameter sets** (s ~ 0.01): ~4-6 min/sample → ~100-150 samples  
- **Slow parameter sets** (s ≤ 0.005): ~8-15 min/sample → ~40-75 samples
- **Mixed distribution**: Estimated ~80-150 total samples with good parameter space coverage

Note: Bootstrap sample count adapts based on precision needs, typically 100-300 samples.

## Notes

- Starts with 50 samples (underestimate for safety)
- Uses recent timing (last 10 samples) for better adaptive estimates
- Adds samples aggressively (up to 50 at once) when time permits
- 20% safety margin on time estimates to avoid overrun
- Each check evaluates if ≥10 more samples are possible before adding
