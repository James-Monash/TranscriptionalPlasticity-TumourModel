cmake_minimum_required(VERSION 3.10)
cmake_policy(SET CMP0135 NEW)
project(TumourSimulation)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenMP for multiprocessing
find_package(OpenMP)

# Find Threads for bootstrap_percentiles
find_package(Threads REQUIRED)

# Add executable
add_executable(tumour_sim
    main.cpp
    Clone.cpp
    Treatment.cpp
    TumourSimulation.cpp
)

# Link OpenMP if available
if(OpenMP_CXX_FOUND)
    target_link_libraries(tumour_sim PUBLIC OpenMP::OpenMP_CXX)
endif()

# Add optimization flags
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(MSVC)
        target_compile_options(tumour_sim PRIVATE /O2 /arch:AVX2)
    else()
        target_compile_options(tumour_sim PRIVATE -O3 -march=native)
    endif()
endif()

find_package(nlohmann_json CONFIG REQUIRED)
target_link_libraries(tumour_sim PRIVATE nlohmann_json::nlohmann_json)

# Add bootstrap_percentiles executable
add_executable(bootstrap_percentiles 
    bootstrap_percentiles.cpp
    TumourSimulation.cpp
    Clone.cpp
    Treatment.cpp
)

# Link threads library and nlohmann_json to bootstrap_percentiles
target_link_libraries(bootstrap_percentiles PRIVATE Threads::Threads nlohmann_json::nlohmann_json)

# Apply same compiler optimizations to bootstrap_percentiles
if(MSVC)
    target_compile_options(bootstrap_percentiles PRIVATE /O2 /W3)
else()
    target_compile_options(bootstrap_percentiles PRIVATE -O3 -Wall)
endif()

# Add bootstrap_manager executable
add_executable(bootstrap_manager 
    bootstrap_manager.cpp
    TumourSimulation.cpp
    Clone.cpp
    Treatment.cpp
)

# Link threads library and nlohmann_json to bootstrap_manager
target_link_libraries(bootstrap_manager PRIVATE Threads::Threads nlohmann_json::nlohmann_json)

# Apply same compiler optimizations to bootstrap_manager
if(MSVC)
    target_compile_options(bootstrap_manager PRIVATE /O2 /W3)
else()
    target_compile_options(bootstrap_manager PRIVATE -O3 -Wall)
endif()