# TranscriptionalPlasticity-TumourModel

Reproducible Python toolkit implementing discrete stochastic tumour‑evolution models (based on Ivana Bozic's research), modular transcriptional‑state dynamics, and example notebooks for analysing treatment response and resistance.

## Features

- **JSON-driven configuration**: Define simulations with flexible JSON files
- **Multi-replicate support**: Run multiple simulations in parallel with multiprocessing
- **Reproducible**: Built-in seeding for deterministic results
- **Three cell states**: Sensitive (S), Quasi-resistant (Q), and Resistant (R) cells
- **Multiple treatment schedules**: MTD, adaptive therapy, continuous, or no treatment
- **Flexible output**: Individual CSVs, consolidated summaries, or custom column selection
- **Visualization**: Generate population dynamics plots showing tumor growth and resistance emergence
- **Initial conditions**: Start with any combination of cell types

## Quick Start

```bash
# Install dependencies
pip install numpy pandas matplotlib

# Run a simulation
python tumour_simulation.py example_configs/mtd_4billion_start.json
```

## Key Configuration Parameters

```json
{
  "simulation": {
    "generations": 100000,
    "initial_size": 1,
    "initial_quasi": 0,
    "initial_resistant": 0,
    "seed": 42,
    "enable_live_plot": true,
    "number_of_replicates": 1,
    "use_multiprocessing": true
  },
  "biological_parameters": {
    "s": 0.01,
    "m": 5.0e-7,
    "q": 1.0e-7,
    "r": 1.0e-07,
    "l": 5.0e-4,
    "idle": 0.1
  },
  "treatment": {
    "schedule_type": "mtd",
    "drug_type": "abs",
    "treat_amt": 0.8,
    "treatment_start_size": 1000000000,
    "relapse_size": 4000000000
  }
}
```

## Documentation

- **[QUICK_START.md](QUICK_START.md)**: Get started in 5 minutes
- **[SIMULATOR_README.md](SIMULATOR_README.md)**: Complete documentation
- **[OUTPUT_CONFIGURATION.md](OUTPUT_CONFIGURATION.md)**: Output options and visualization
- **[ARCHITECTURE_DIAGRAMS.md](ARCHITECTURE_DIAGRAMS.md)**: System architecture

## Recent Updates

- ✅ Configurable initial cell populations (initial_quasi, initial_resistant)
- ✅ Built-in random seed support for reproducibility
- ✅ Population dynamics visualization (enable_live_plot)
- ✅ Configurable relapse size threshold
- ✅ Multi-replicate parallel processing
- ✅ Flexible output options with consolidated summaries

## Example Use Cases

1. **Test MTD vs Adaptive Therapy**: Compare treatment schedules
2. **Study Resistance Evolution**: Track quasi-resistant and resistant cell emergence
3. **Parameter Sweeps**: Run multiple conditions with different biological parameters
4. **Initial Resistance**: Start with pre-existing resistant populations
5. **Reproducible Science**: Use seeds for exact replication of results

## License

[Your license information]

## Contact

[Your contact information]
