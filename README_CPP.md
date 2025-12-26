# C++ Tumor Simulation

High-performance C++ implementation of the tumor evolution simulator.

## Building

### Requirements
- C++17 compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)
- CMake 3.10+
- OpenMP (optional, for multiprocessing)

### Windows (Visual Studio)
```bash
mkdir build
cd build
cmake .. -G "Visual Studio 16 2019"
cmake --build . --config Release
```

### Linux/Mac
```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## Usage

```bash
# Single replicate
./tumour_sim config.json

# Multiple replicates with 8 threads
./tumour_sim config.json 8
```

## Performance Notes

- Uses OpenMP for parallel replicate execution
- Efficient multinomial sampling with C++ random library
- Probability caching to avoid recalculations
- Optimized memory layout with unique_ptr
- ~10-50x faster than Python version depending on configuration

## Key Differences from Python

1. **Probability table**: Defaults to first 10 k values (configurable by changing constant in Treatment.cpp)
2. **Multiprocessing**: Uses OpenMP threads instead of Python multiprocessing
3. **Random sampling**: Uses C++17 std::discrete_distribution for multinomial sampling
4. **Memory**: More efficient with std::unique_ptr and move semantics
