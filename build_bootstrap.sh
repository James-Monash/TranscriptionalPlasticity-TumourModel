#!/bin/bash

# Create build directory for bootstrap
mkdir -p build_bootstrap
cd build_bootstrap

# Run CMake with the bootstrap CMakeLists
cmake -DCMAKE_BUILD_TYPE=Release -C ../CMakeLists_bootstrap.txt ..

# Build the project
cmake --build . --config Release

echo "Bootstrap executable built successfully!"
echo "Run with: ./bootstrap_percentiles [num_simulations]"
