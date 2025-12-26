# Bootstrap Percentile Analysis

This C++ program performs bootstrap analysis to determine confidence intervals for resistance distribution percentiles using your TumourSimulation framework.

## Features

- **Hardcoded Parameters**: Uses biological parameters from test.json (s=0.1, m=5.71e-07, q=1.68e-07, r=1.68e-07, l=8.4e-04, idle=0.1)
- **Multiprocessing**: Runs simulations in parallel using all available CPU cores
- **Adaptive Bootstrap**: Starts with 500 samples and adds 500 more until precision targets are met
- **Confidence Intervals**: Calculates 95% CI for 50th, 75th, and 90th percentiles

## Requirements

- C++17 compatible compiler (GCC, Clang, or MSVC)
- CMake 3.10 or higher
- nlohmann/json library (header-only)
- Your existing TumourSimulation codebase

## Building

### On Linux/Mac:
```bash
chmod +x build_bootstrap.sh
./build_bootstrap.sh
```

### On Windows:
```cmd
build_bootstrap.bat
```

### Manual Build:
```bash
mkdir build_bootstrap
cd build_bootstrap
cmake -C ../CMakeLists_bootstrap.txt ..
cmake --build . --config Release
```

## Usage

Run with default 1000 simulations:
```bash
./bootstrap_percentiles
```

Specify number of simulations:
```bash
./bootstrap_percentiles 5000
```

## How It Works

1. **Generate Resistance Distribution**: Runs N simulations (default 1000) using multiprocessing to generate resistance fraction data
2. **Bootstrap Analysis**: Performs bootstrap resampling starting with 500 samples
3. **Calculate Percentiles**: Computes 95% CI for 50th, 75th, and 90th percentiles
4. **Check Precision**: 
   - 50th & 75th percentiles: CI width ≤ 2 percentage points
   - 90th percentile: CI width ≤ 3 percentage points
5. **Iterate**: If not met, adds 500 more bootstrap samples (max 10,000)
6. **Save Results**: Outputs to `bootstrap_results.json`

## Output

**Console**: Progress updates and confidence interval statistics

**File**: `bootstrap_results.json` with structure:
```json
{
    "biological_parameters": {...},
    "num_simulations": 1000,
    "bootstrap_samples": 1500,
    "percentiles": {
        "50th": {
            "lower": 0.234,
            "upper": 0.256,
            "width_pct": 2.2
        },
        ...
    }
}
```

## Notes

- The program uses all available CPU cores for parallel simulation execution
- Each simulation runs independently with its own random seed
- Adjust `num_simulations` parameter based on desired statistical power
- The bootstrap uses the percentile method for CI calculation
