#include "TumourSimulation.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <random>
#include <cmath>
#include <iomanip>
#include <thread>
#include <mutex>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

struct BootstrapResult {
    double percentile_50_lower;
    double percentile_50_upper;
    double percentile_50_width;
    double percentile_75_lower;
    double percentile_75_upper;
    double percentile_75_width;
    double percentile_90_lower;
    double percentile_90_upper;
    double percentile_90_width;
    int bootstrap_samples;
};

// Mutex for thread-safe vector operations
std::mutex results_mutex;

// Function to calculate percentile from sorted data
double calculatePercentile(const std::vector<double>& sorted_data, double percentile) {
    if (sorted_data.empty()) return 0.0;
    
    double index = (percentile / 100.0) * (sorted_data.size() - 1);
    int lower_index = static_cast<int>(std::floor(index));
    int upper_index = static_cast<int>(std::ceil(index));
    
    if (lower_index == upper_index) {
        return sorted_data[lower_index];
    }
    
    double weight = index - lower_index;
    return sorted_data[lower_index] * (1.0 - weight) + sorted_data[upper_index] * weight;
}

// Function to perform one bootstrap sample
std::vector<double> bootstrapSample(const std::vector<double>& original_data, std::mt19937& rng) {
    std::vector<double> sample;
    sample.reserve(original_data.size());
    std::uniform_int_distribution<size_t> dist(0, original_data.size() - 1);
    
    for (size_t i = 0; i < original_data.size(); ++i) {
        sample.push_back(original_data[dist(rng)]);
    }
    
    return sample;
}

// Function to calculate bootstrap confidence intervals
BootstrapResult calculateBootstrapCI(const std::vector<double>& data, int num_bootstrap_samples, double confidence_level = 0.95) {
    std::random_device rd;
    std::mt19937 rng(rd());
    
    std::vector<double> percentile_50_estimates;
    std::vector<double> percentile_75_estimates;
    std::vector<double> percentile_90_estimates;
    
    percentile_50_estimates.reserve(num_bootstrap_samples);
    percentile_75_estimates.reserve(num_bootstrap_samples);
    percentile_90_estimates.reserve(num_bootstrap_samples);
    
    // Perform bootstrap resampling
    for (int i = 0; i < num_bootstrap_samples; ++i) {
        std::vector<double> sample = bootstrapSample(data, rng);
        std::sort(sample.begin(), sample.end());
        
        percentile_50_estimates.push_back(calculatePercentile(sample, 50.0));
        percentile_75_estimates.push_back(calculatePercentile(sample, 75.0));
        percentile_90_estimates.push_back(calculatePercentile(sample, 90.0));
    }
    
    // Sort bootstrap estimates
    std::sort(percentile_50_estimates.begin(), percentile_50_estimates.end());
    std::sort(percentile_75_estimates.begin(), percentile_75_estimates.end());
    std::sort(percentile_90_estimates.begin(), percentile_90_estimates.end());
    
    // Calculate confidence intervals (using percentile method)
    double alpha = 1.0 - confidence_level;
    double lower_percentile = (alpha / 2.0) * 100.0;
    double upper_percentile = (1.0 - alpha / 2.0) * 100.0;
    
    BootstrapResult result;
    result.percentile_50_lower = calculatePercentile(percentile_50_estimates, lower_percentile);
    result.percentile_50_upper = calculatePercentile(percentile_50_estimates, upper_percentile);
    result.percentile_50_width = (result.percentile_50_upper - result.percentile_50_lower) * 100.0;
    
    result.percentile_75_lower = calculatePercentile(percentile_75_estimates, lower_percentile);
    result.percentile_75_upper = calculatePercentile(percentile_75_estimates, upper_percentile);
    result.percentile_75_width = (result.percentile_75_upper - result.percentile_75_lower) * 100.0;
    
    result.percentile_90_lower = calculatePercentile(percentile_90_estimates, lower_percentile);
    result.percentile_90_upper = calculatePercentile(percentile_90_estimates, upper_percentile);
    result.percentile_90_width = (result.percentile_90_upper - result.percentile_90_lower) * 100.0;
    
    result.bootstrap_samples = num_bootstrap_samples;
    
    return result;
}

// Create config JSON with hardcoded parameters from test.json
json createConfig() {
    json config;
    
    // Simulation parameters
    config["simulation"]["generations"] = 100000;
    config["simulation"]["initial_size"] = 1000;
    config["simulation"]["output_dir"] = "./output";
    config["simulation"]["output_prefix"] = "bootstrap_simulation";
    config["simulation"]["track_history"] = false;
    config["simulation"]["number_of_replicates"] = 1;
    config["simulation"]["output"]["save_individual_csvs"] = false;
    config["simulation"]["output"]["save_summary_json"] = false;
    config["simulation"]["output"]["save_consolidated_summary"] = false;
    
    // Biological parameters (hardcoded from test.json)
    config["biological_parameters"]["s"] = 0.001;
    config["biological_parameters"]["m"] = 5.71e-07;
    config["biological_parameters"]["q"] = 1.68e-07;
    config["biological_parameters"]["r"] = 1.68e-07;
    config["biological_parameters"]["l"] = 8.4e-04;
    config["biological_parameters"]["idle"] = 0.1;
    
    // Treatment parameters
    config["treatment"]["schedule_type"] = "off";
    config["treatment"]["drug_type"] = "abs";
    config["treatment"]["treat_amt"] = 0.0;
    config["treatment"]["pen_amt"] = 0.0;
    config["treatment"]["dose_duration"] = 0;
    config["treatment"]["penalty"] = false;
    config["treatment"]["secondary_therapy"] = false;
    
    config["use_multiprocessing"] = true;
    
    return config;
}

// Worker function for parallel simulation execution
void runSimulationWorker(const json& config, int start_idx, int end_idx, 
                        std::vector<double>& resistance_data) {
    std::vector<double> local_results;
    
    for (int i = start_idx; i < end_idx; ++i) {
        try {
            TumourSimulation sim(config, 0, i);
            auto [final_state, clone_data] = sim.run();
            
            auto summary = sim.get_summary();
            double fraction_resistant = summary.fraction_resistant;
            
            local_results.push_back(fraction_resistant);
            
        } catch (const std::exception& e) {
            std::cerr << "Error in simulation " << i << ": " << e.what() << std::endl;
        }
    }
    
    // Thread-safe append to results
    std::lock_guard<std::mutex> lock(results_mutex);
    resistance_data.insert(resistance_data.end(), local_results.begin(), local_results.end());
}

// Function to run simulations with multiprocessing
std::vector<double> runSimulations(const json& config, int num_simulations) {
    std::vector<double> resistance_data;
    resistance_data.reserve(num_simulations);
    
    // Determine number of threads
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4; // Default fallback
    
    std::cout << "Running " << num_simulations << " simulations using " 
              << num_threads << " threads..." << std::endl;
    
    // Calculate work distribution
    int sims_per_thread = num_simulations / num_threads;
    int remainder = num_simulations % num_threads;
    
    std::vector<std::thread> threads;
    int current_idx = 0;
    
    for (unsigned int t = 0; t < num_threads; ++t) {
        int thread_sims = sims_per_thread + (static_cast<int>(t) < remainder ? 1 : 0);
        int start_idx = current_idx;
        int end_idx = current_idx + thread_sims;
        
        threads.emplace_back(runSimulationWorker, std::cref(config), 
                            start_idx, end_idx, std::ref(resistance_data));
        
        current_idx = end_idx;
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
    
    std::cout << "Completed " << resistance_data.size() << " simulations." << std::endl;
    
    return resistance_data;
}

int main(int argc, char* argv[]) {
    try {
        // Create config with hardcoded parameters
        json config = createConfig();
        
        std::cout << "=== Bootstrap Percentile Analysis ===" << std::endl;
        std::cout << "Biological Parameters (Hardcoded):" << std::endl;
        std::cout << "  s = " << config["biological_parameters"]["s"] << std::endl;
        std::cout << "  m = " << config["biological_parameters"]["m"] << std::endl;
        std::cout << "  q = " << config["biological_parameters"]["q"] << std::endl;
        std::cout << "  r = " << config["biological_parameters"]["r"] << std::endl;
        std::cout << "  l = " << config["biological_parameters"]["l"] << std::endl;
        std::cout << "  idle = " << config["biological_parameters"]["idle"] << std::endl;
        std::cout << std::endl;
        
        // Number of simulations to generate resistance distribution
        int num_simulations = 1000; // Adjust based on your needs
        if (argc > 1) {
            num_simulations = std::stoi(argv[1]);
        }
        
        // Run simulations to get resistance distribution
        std::cout << "Running simulations to generate resistance distribution..." << std::endl;
        std::vector<double> resistance_data = runSimulations(config, num_simulations);
        
        if (resistance_data.empty()) {
            std::cerr << "Error: No simulation data generated." << std::endl;
            return 1;
        }
        
        std::sort(resistance_data.begin(), resistance_data.end());
        std::cout << std::endl;
        
        // Target precision
        const double target_width_50_75 = 2.0; // 1-2 percentage points
        const double target_width_90 = 3.0;    // 1-3 percentage points
        
        int num_bootstrap_samples = 500;
        bool criteria_met = false;
        
        while (!criteria_met && num_bootstrap_samples <= 10000) {
            std::cout << "Performing bootstrap analysis with " << num_bootstrap_samples << " samples..." << std::endl;
            
            BootstrapResult result = calculateBootstrapCI(resistance_data, num_bootstrap_samples);
            
            std::cout << std::fixed << std::setprecision(6);
            std::cout << "\n50th Percentile:" << std::endl;
            std::cout << "  95% CI: [" << result.percentile_50_lower << ", " << result.percentile_50_upper << "]" << std::endl;
            std::cout << "  Width: " << result.percentile_50_width << " percentage points" << std::endl;
            
            std::cout << "\n75th Percentile:" << std::endl;
            std::cout << "  95% CI: [" << result.percentile_75_lower << ", " << result.percentile_75_upper << "]" << std::endl;
            std::cout << "  Width: " << result.percentile_75_width << " percentage points" << std::endl;
            
            std::cout << "\n90th Percentile:" << std::endl;
            std::cout << "  95% CI: [" << result.percentile_90_lower << ", " << result.percentile_90_upper << "]" << std::endl;
            std::cout << "  Width: " << result.percentile_90_width << " percentage points" << std::endl;
            std::cout << std::endl;
            
            // Check if criteria are met
            if (result.percentile_50_width <= target_width_50_75 &&
                result.percentile_75_width <= target_width_50_75 &&
                result.percentile_90_width <= target_width_90) {
                criteria_met = true;
                std::cout << "✓ Precision criteria met!" << std::endl;
                
                // Save results to file
                json output;
                output["biological_parameters"] = config["biological_parameters"];
                output["num_simulations"] = num_simulations;
                output["bootstrap_samples"] = result.bootstrap_samples;
                output["percentiles"] = {
                    {"50th", {
                        {"lower", result.percentile_50_lower},
                        {"upper", result.percentile_50_upper},
                        {"width_pct", result.percentile_50_width}
                    }},
                    {"75th", {
                        {"lower", result.percentile_75_lower},
                        {"upper", result.percentile_75_upper},
                        {"width_pct", result.percentile_75_width}
                    }},
                    {"90th", {
                        {"lower", result.percentile_90_lower},
                        {"upper", result.percentile_90_upper},
                        {"width_pct", result.percentile_90_width}
                    }}
                };
                
                std::ofstream out("bootstrap_results.json");
                out << std::setw(4) << output << std::endl;
                out.close();
                
                std::cout << "\nResults saved to bootstrap_results.json" << std::endl;
            } else {
                std::cout << "✗ Precision criteria not met. Adding 500 more samples..." << std::endl;
                num_bootstrap_samples += 500;
            }
        }
        
        if (!criteria_met) {
            std::cout << "\nWarning: Maximum bootstrap samples reached without meeting criteria." << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
