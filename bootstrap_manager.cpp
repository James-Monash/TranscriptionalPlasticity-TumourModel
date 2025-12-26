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
#include <atomic>
#include <chrono>
#include <filesystem>
#include <optional>
#include <nlohmann/json.hpp>
#include <cstdlib>  // Add for system()

using json = nlohmann::json;
namespace fs = std::filesystem;

struct ParameterSet {
    double s, m, q, r, l, idle;
    int sample_id;
    bool completed;
    
    json to_json() const {
        return {
            {"sample_id", sample_id},
            {"s", s}, {"m", m}, {"q", q}, {"r", r}, {"l", l}, {"idle", idle},
            {"completed", completed}
        };
    }
    
    static ParameterSet from_json(const json& j) {
        return {
            j["s"], j["m"], j["q"], j["r"], j["l"], j["idle"],
            j["sample_id"], j.value("completed", false)
        };
    }
};

struct BootstrapResults {
    double total_resistant_50th_lower, total_resistant_50th_upper;
    double total_resistant_75th_lower, total_resistant_75th_upper;
    double total_resistant_90th_lower, total_resistant_90th_upper;
    
    double transient_fraction_50th_lower, transient_fraction_50th_upper;
    double transient_fraction_75th_lower, transient_fraction_75th_upper;
    double transient_fraction_90th_lower, transient_fraction_90th_upper;
    
    int bootstrap_samples;
    int num_simulations;
    
    json to_json() const {
        return {
            {"total_resistant", {
                {"50th", {{"lower", total_resistant_50th_lower}, {"upper", total_resistant_50th_upper}}},
                {"75th", {{"lower", total_resistant_75th_lower}, {"upper", total_resistant_75th_upper}}},
                {"90th", {{"lower", total_resistant_90th_lower}, {"upper", total_resistant_90th_upper}}}
            }},
            {"transient_fraction", {
                {"50th", {{"lower", transient_fraction_50th_lower}, {"upper", transient_fraction_50th_upper}}},
                {"75th", {{"lower", transient_fraction_75th_lower}, {"upper", transient_fraction_75th_upper}}},
                {"90th", {{"lower", transient_fraction_90th_lower}, {"upper", transient_fraction_90th_upper}}}
            }},
            {"bootstrap_samples", bootstrap_samples},
            {"num_simulations", num_simulations}
        };
    }
};

class BootstrapManager {
private:
    std::vector<ParameterSet> parameter_samples;
    std::map<int, BootstrapResults> results_cache;
    std::string checkpoint_file;
    std::string results_dir;
    std::mt19937 rng;
    std::mutex io_mutex;
    
    // Timing information
    std::chrono::steady_clock::time_point start_time;
    double time_budget_hours;
    std::vector<double> simulation_times; // Track time per parameter set
    
    // Track problematic parameter regions
    std::vector<ParameterSet> slow_parameter_regions;
    mutable std::mutex slow_regions_mutex;  // Make mutable for const methods
    
    struct ProblemRegion {
        int sample_id;
        std::vector<double> parameters;
        double total_distance;
    };
    
    struct ValidationResults {
        double coverage;
        std::vector<ProblemRegion> problem_regions;
    };
    
    bool is_valid_parameter_set(const ParameterSet& params) const {
        // Calculate probabilities for k=1 (base case)
        double prob_death = ((1 - params.idle) / 2.0) * (1 - params.s);
        double prob_mutation = params.m * (1 - prob_death);
        double prob_to_transient = params.q * (1 - prob_death);
        double prob_to_resistant = params.r * (1 - prob_death);
        double prob_to_sensitive = params.l * (1 - prob_death);
        double prob_birth = 1 - params.idle - prob_death - prob_mutation - 
                           prob_to_transient - prob_to_resistant - prob_to_sensitive;
        
        // Check if birth > death
        return prob_birth > prob_death;
    }
    
    bool is_in_slow_region(const ParameterSet& params) const {
        std::lock_guard<std::mutex> lock(slow_regions_mutex);
        
        for (const auto& slow_params : slow_parameter_regions) {
            // Check if this parameter set is in a known slow region
            // It's in a slow region if s <= slow_s AND idle >= slow_idle
            if (params.s <= slow_params.s && params.idle >= slow_params.idle) {
                return true;
            }
        }
        return false;
    }
    
    void mark_as_slow_region(const ParameterSet& params) {
        std::lock_guard<std::mutex> lock(slow_regions_mutex);
        slow_parameter_regions.push_back(params);
        
        std::cout << "\n  ⚠ Marking parameter region as too slow:" << std::endl;
        std::cout << "    s <= " << params.s << ", idle >= " << params.idle << std::endl;
        std::cout << "    Future samples in this region will be skipped." << std::endl;
    }
    
public:
    // Add emulator state tracking
    int emulator_iteration;
    std::string emulator_model_file;
    std::string validation_results_file;
    
    BootstrapManager(double time_budget_hours = 10.0)
        : checkpoint_file("bootstrap_checkpoint.json"),
          results_dir("bootstrap_results"),
          time_budget_hours(time_budget_hours),
          emulator_iteration(0),
          emulator_model_file("gp_emulator_model.pkl"),
          validation_results_file("gp_validation_results.json") {
        
        std::random_device rd;
        rng.seed(rd());
        
        // Create results directory
        fs::create_directories(results_dir);
        
        // Load checkpoint if exists
        load_checkpoint();
    }
    
    void run() {
        start_time = std::chrono::steady_clock::now();
        
        std::cout << "=== Bootstrap Parameter Space Manager ===" << std::endl;
        std::cout << "Time budget: " << time_budget_hours << " hours" << std::endl;
        
        // Check if we're continuing from a previous run
        if (fs::exists(emulator_model_file)) {
            std::cout << "Found existing emulator model - continuing refinement" << std::endl;
        }
        
        std::cout << std::endl;
        
        // Initial sampling if needed
        if (parameter_samples.empty()) {
            int initial_samples = estimate_initial_samples();
            std::cout << "Generating initial " << initial_samples << " LHS samples..." << std::endl;
            generate_lhs_samples(initial_samples);
            save_checkpoint();
        } else {
            std::cout << "Loaded " << parameter_samples.size() << " parameter samples from checkpoint" << std::endl;
            int completed = std::count_if(parameter_samples.begin(), parameter_samples.end(),
                                         [](const auto& p) { return p.completed; });
            std::cout << "Completed: " << completed << "/" << parameter_samples.size() << std::endl;
        }
        
        // Process parameter sets
        process_parameter_space();
        
        // Build and iteratively improve emulator
        std::cout << "\n=== Emulator Training and Validation ===" << std::endl;
        build_and_improve_emulator();
        
        std::cout << "\n=== Analysis Complete ===" << std::endl;
        print_final_summary();
    }
    
private:
    int estimate_initial_samples() {
        // Reduce slightly to 40 to reserve time for validation and resampling
        return 40;
    }
    
    void generate_lhs_samples(int n_samples) {
        // Latin Hypercube Sampling
        // Parameter ranges:
        // s: [0.001, 0.1]
        // m: [1e-8, 1e-5]
        // q: [1e-11, 1e-2]
        // r: [1e-11, 1e-2]
        // l: [1e-11, 1e-2]
        // idle: [0, 0.5]
        
        std::vector<ParameterSet> valid_samples;
        valid_samples.reserve(n_samples);
        
        // Generate more samples than needed to account for filtering
        int max_attempts = n_samples * 10;
        int attempts = 0;
        
        while (valid_samples.size() < static_cast<size_t>(n_samples) && attempts < max_attempts) {
            // Generate a batch of samples
            int batch_size = n_samples * 2;
            std::vector<std::vector<double>> lhs_samples(6, std::vector<double>(batch_size));
            
            // Generate permutations for each dimension
            for (int dim = 0; dim < 6; ++dim) {
                std::vector<int> indices(batch_size);
                std::iota(indices.begin(), indices.end(), 0);
                std::shuffle(indices.begin(), indices.end(), rng);
                
                for (int i = 0; i < batch_size; ++i) {
                    std::uniform_real_distribution<double> dist(0.0, 1.0);
                    double rand_val = dist(rng);
                    lhs_samples[dim][i] = (indices[i] + rand_val) / batch_size;
                }
            }
            
            // Transform to actual parameter ranges and validate
            for (int i = 0; i < batch_size && valid_samples.size() < static_cast<size_t>(n_samples); ++i) {
                ParameterSet params;
                params.completed = false;
                
                // s: log-uniform between 0.001 and 0.1
                double log_s_min = std::log10(0.001);
                double log_s_max = std::log10(0.1);
                params.s = std::pow(10, log_s_min + lhs_samples[0][i] * (log_s_max - log_s_min));
                
                // m: log-uniform between 1e-8 and 1e-5
                double log_m_min = -8;
                double log_m_max = -5;
                params.m = std::pow(10, log_m_min + lhs_samples[1][i] * (log_m_max - log_m_min));
                
                // q: log-uniform between 1e-11 and 1e-2
                double log_q_min = -11;
                double log_q_max = -2;
                params.q = std::pow(10, log_q_min + lhs_samples[2][i] * (log_q_max - log_q_min));
                
                // r: log-uniform between 1e-11 and 1e-2
                params.r = std::pow(10, log_q_min + lhs_samples[3][i] * (log_q_max - log_q_min));
                
                // l: log-uniform between 1e-11 and 1e-2
                params.l = std::pow(10, log_q_min + lhs_samples[4][i] * (log_q_max - log_q_min));
                
                // idle: uniform between 0 and 0.5
                params.idle = lhs_samples[5][i] * 0.5;
                
                // Validate: only keep if birth > death AND not in slow region
                if (is_valid_parameter_set(params) && !is_in_slow_region(params)) {
                    valid_samples.push_back(params);
                }
            }
            
            attempts++;
        }
        
        if (valid_samples.size() < static_cast<size_t>(n_samples)) {
            std::cout << "Warning: Only generated " << valid_samples.size() 
                     << " valid samples out of " << n_samples << " requested" << std::endl;
        }
        
        // Assign sample IDs and add to parameter_samples
        int start_id = parameter_samples.size();
        for (size_t i = 0; i < valid_samples.size(); ++i) {
            valid_samples[i].sample_id = start_id + i;
            parameter_samples.push_back(valid_samples[i]);
        }
    }
    
    void process_parameter_space() {
        int total_samples = parameter_samples.size();
        int completed_count = 0;
        int skipped_count = 0;
        
        for (auto& param_set : parameter_samples) {
            if (param_set.completed) {
                completed_count++;
                continue;
            }
            
            // Check if this parameter set is in a known slow region
            if (is_in_slow_region(param_set)) {
                std::cout << "\n--- Skipping Sample " << param_set.sample_id 
                         << " (in slow region) ---" << std::endl;
                param_set.completed = true;  // Mark as completed to skip it
                skipped_count++;
                save_checkpoint();
                continue;
            }
            
            // Check time budget
            if (!check_time_budget()) {
                std::cout << "\nTime budget exhausted. Stopping..." << std::endl;
                break;
            }
            
            std::cout << "\n--- Processing Sample " << (completed_count + 1) 
                     << "/" << total_samples << " (ID: " << param_set.sample_id << ") ---" << std::endl;
            std::cout << "Parameters: s=" << param_set.s << ", m=" << param_set.m 
                     << ", q=" << param_set.q << ", r=" << param_set.r 
                     << ", l=" << param_set.l << ", idle=" << param_set.idle << std::endl;
            
            auto sample_start = std::chrono::steady_clock::now();
            
            // Run bootstrap analysis for this parameter set
            auto results = run_bootstrap_for_parameters(param_set);
            
            auto sample_end = std::chrono::steady_clock::now();
            double sample_time = std::chrono::duration<double>(sample_end - sample_start).count();
            simulation_times.push_back(sample_time);
            
            std::cout << "Sample completed in " << sample_time << " seconds" << std::endl;
            
            // Save results
            results_cache[param_set.sample_id] = results;
            param_set.completed = true;
            completed_count++;
            
            // Save checkpoint
            save_checkpoint();
            save_results(param_set.sample_id, param_set, results);
            
            // Adaptive sampling: add more samples after initial batch
            if (completed_count == 10 || completed_count % 20 == 0) {
                maybe_add_more_samples();
            }
        }
        
        std::cout << "\nCompleted " << completed_count << "/" << total_samples << " samples" << std::endl;
        if (skipped_count > 0) {
            std::cout << "Skipped " << skipped_count << " samples in slow regions" << std::endl;
        }
    }
    
    BootstrapResults run_bootstrap_for_parameters(const ParameterSet& params) {
        // Start with base number of simulations
        int num_simulations = 500;
        const int max_simulations = 5000;  // Upper limit for simulations
        
        std::cout << "Running " << num_simulations << " simulations..." << std::endl;
        
        // Create config with these parameters
        json config = create_config(params);
        
        // Run simulations - catch slow simulation exception
        std::vector<double> total_resistant_data;
        std::vector<double> transient_fraction_data;
        
        try {
            auto [tr_data, tf_data] = run_simulations(config, num_simulations);
            total_resistant_data = tr_data;
            transient_fraction_data = tf_data;
        } catch (const std::runtime_error& e) {
            // Mark this parameter region as slow
            mark_as_slow_region(params);
            
            // Return empty results
            BootstrapResults empty_results;
            empty_results.total_resistant_50th_lower = 0.0;
            empty_results.total_resistant_50th_upper = 0.0;
            empty_results.total_resistant_75th_lower = 0.0;
            empty_results.total_resistant_75th_upper = 0.0;
            empty_results.total_resistant_90th_lower = 0.0;
            empty_results.total_resistant_90th_upper = 0.0;
            empty_results.transient_fraction_50th_lower = 0.0;
            empty_results.transient_fraction_50th_upper = 0.0;
            empty_results.transient_fraction_75th_lower = 0.0;
            empty_results.transient_fraction_75th_upper = 0.0;
            empty_results.transient_fraction_90th_lower = 0.0;
            empty_results.transient_fraction_90th_upper = 0.0;
            empty_results.bootstrap_samples = 0;
            empty_results.num_simulations = 0;
            return empty_results;
        }
        
        // Perform bootstrap analysis with adaptive sampling
        std::cout << "Performing bootstrap analysis..." << std::endl;
        
        BootstrapResults results;
        
        // Target precision
        const double target_width_50_75 = 2.0;  // 2 percentage points
        const double target_width_90 = 3.0;     // 3 percentage points
        
        // Start with 100 bootstrap samples
        int num_bootstrap = 100;
        const int max_bootstrap = 2000;  // Increased from 1000
        
        bool precision_met = false;
        int iteration = 0;
        const int max_iterations = 20;  // Increased to allow more attempts
        
        while (!precision_met && iteration < max_iterations) {
            iteration++;
            
            // Bootstrap for total resistant
            auto total_resistant_ci = calculate_bootstrap_ci(total_resistant_data, num_bootstrap);
            double tr_50_width = (total_resistant_ci.percentile_50_upper - total_resistant_ci.percentile_50_lower) * 100.0;
            double tr_75_width = (total_resistant_ci.percentile_75_upper - total_resistant_ci.percentile_75_lower) * 100.0;
            double tr_90_width = (total_resistant_ci.percentile_90_upper - total_resistant_ci.percentile_90_lower) * 100.0;
            
            // Check if precision targets are met
            if (tr_50_width <= target_width_50_75 && 
                tr_75_width <= target_width_50_75 && 
                tr_90_width <= target_width_90) {
                precision_met = true;
                
                results.total_resistant_50th_lower = total_resistant_ci.percentile_50_lower;
                results.total_resistant_50th_upper = total_resistant_ci.percentile_50_upper;
                results.total_resistant_75th_lower = total_resistant_ci.percentile_75_lower;
                results.total_resistant_75th_upper = total_resistant_ci.percentile_75_upper;
                results.total_resistant_90th_lower = total_resistant_ci.percentile_90_lower;
                results.total_resistant_90th_upper = total_resistant_ci.percentile_90_upper;
                
                // Bootstrap for transient fraction (use same number of bootstrap samples)
                auto transient_fraction_ci = calculate_bootstrap_ci(transient_fraction_data, num_bootstrap);
                results.transient_fraction_50th_lower = transient_fraction_ci.percentile_50_lower;
                results.transient_fraction_50th_upper = transient_fraction_ci.percentile_50_upper;
                results.transient_fraction_75th_lower = transient_fraction_ci.percentile_75_lower;
                results.transient_fraction_75th_upper = transient_fraction_ci.percentile_75_upper;
                results.transient_fraction_90th_lower = transient_fraction_ci.percentile_90_lower;
                results.transient_fraction_90th_upper = transient_fraction_ci.percentile_90_upper;
                
                results.bootstrap_samples = num_bootstrap;
                results.num_simulations = num_simulations;
                
                std::cout << "  ✓ Precision targets met!" << std::endl;
                std::cout << "  Final: " << num_simulations << " simulations, " 
                         << num_bootstrap << " bootstrap samples" << std::endl;
            } else {
                // Strategy: Try increasing bootstrap samples first, but switch to simulations sooner
                // if bootstrap increases aren't helping with the 90th percentile
                
                bool should_increase_sims = false;
                
                // If we've tried many bootstrap samples but 90th percentile still too wide
                if (num_bootstrap >= 1000 && tr_90_width > target_width_90 * 1.5) {
                    should_increase_sims = true;
                }
                // Or if we've maxed out bootstrap samples
                else if (num_bootstrap >= max_bootstrap) {
                    should_increase_sims = true;
                }
                
                if (should_increase_sims && num_simulations < max_simulations) {
                    // Calculate how many more simulations we need
                    // More aggressive increase for 90th percentile issues
                    int additional_sims = std::min(1000, max_simulations - num_simulations);
                    num_simulations += additional_sims;
                    
                    std::cout << "  → Wide CI (especially 90th: " << tr_90_width 
                             << "%). Increasing simulations to " << num_simulations << std::endl;
                    
                    // Run additional simulations - catch slow simulation exception
                    try {
                        auto [new_tr_data, new_tf_data] = run_simulations(config, additional_sims);
                        
                        // Append new data to existing
                        total_resistant_data.insert(total_resistant_data.end(), 
                                                  new_tr_data.begin(), new_tr_data.end());
                        transient_fraction_data.insert(transient_fraction_data.end(), 
                                                      new_tf_data.begin(), new_tf_data.end());
                        
                        // Re-sort the combined data
                        std::sort(total_resistant_data.begin(), total_resistant_data.end());
                        std::sort(transient_fraction_data.begin(), transient_fraction_data.end());
                        
                        // Reset bootstrap samples to reasonable level with more data
                        num_bootstrap = std::min(500, max_bootstrap);
                        std::cout << "  → Reset bootstrap samples to " << num_bootstrap << " with larger dataset" << std::endl;
                    } catch (const std::runtime_error& e) {
                        // Slow simulation detected during additional runs
                        mark_as_slow_region(params);
                        
                        // Return empty results
                        BootstrapResults empty_results;
                        empty_results.total_resistant_50th_lower = 0.0;
                        empty_results.total_resistant_50th_upper = 0.0;
                        empty_results.total_resistant_75th_lower = 0.0;
                        empty_results.total_resistant_75th_upper = 0.0;
                        empty_results.total_resistant_90th_lower = 0.0;
                        empty_results.total_resistant_90th_upper = 0.0;
                        empty_results.transient_fraction_50th_lower = 0.0;
                        empty_results.transient_fraction_50th_upper = 0.0;
                        empty_results.transient_fraction_75th_lower = 0.0;
                        empty_results.transient_fraction_75th_upper = 0.0;
                        empty_results.transient_fraction_90th_lower = 0.0;
                        empty_results.transient_fraction_90th_upper = 0.0;
                        empty_results.bootstrap_samples = 0;
                        empty_results.num_simulations = 0;
                        return empty_results;
                    }
                }
                else if (num_bootstrap < max_bootstrap) {
                    // Still room to increase bootstrap samples
                    num_bootstrap = std::min(num_bootstrap + 200, max_bootstrap);
                    std::cout << "  → Increasing bootstrap samples to " << num_bootstrap << std::endl;
                }
                else {
                    // We've maxed out both - accept what we have
                    std::cout << "  ⚠ Maximum simulations and bootstrap samples reached" << std::endl;
                    std::cout << "    Accepting current precision:" << std::endl;
                    std::cout << "    50th CI width: " << tr_50_width << "%" << std::endl;
                    std::cout << "    75th CI width: " << tr_75_width << "%" << std::endl;
                    std::cout << "    90th CI width: " << tr_90_width << "%" << std::endl;
                    
                    results.total_resistant_50th_lower = total_resistant_ci.percentile_50_lower;
                    results.total_resistant_50th_upper = total_resistant_ci.percentile_50_upper;
                    results.total_resistant_75th_lower = total_resistant_ci.percentile_75_lower;
                    results.total_resistant_75th_upper = total_resistant_ci.percentile_75_upper;
                    results.total_resistant_90th_lower = total_resistant_ci.percentile_90_lower;
                    results.total_resistant_90th_upper = total_resistant_ci.percentile_90_upper;
                    
                    auto transient_fraction_ci = calculate_bootstrap_ci(transient_fraction_data, num_bootstrap);
                    results.transient_fraction_50th_lower = transient_fraction_ci.percentile_50_lower;
                    results.transient_fraction_50th_upper = transient_fraction_ci.percentile_50_upper;
                    results.transient_fraction_75th_lower = transient_fraction_ci.percentile_75_lower;
                    results.transient_fraction_75th_upper = transient_fraction_ci.percentile_75_upper;
                    results.transient_fraction_90th_lower = transient_fraction_ci.percentile_90_lower;
                    results.transient_fraction_90th_upper = transient_fraction_ci.percentile_90_upper;
                    
                    results.bootstrap_samples = num_bootstrap;
                    results.num_simulations = num_simulations;
                    
                    precision_met = true;  // Force exit
                }
                
                std::cout << "  Current CI widths: 50th=" << tr_50_width 
                         << "%, 75th=" << tr_75_width << "%, 90th=" << tr_90_width << "%" << std::endl;
            }
        }
        
        return results;
    }
    
    int estimate_num_simulations(double s) {
        // Fixed at 500 for all parameter sets
        return 500;
    }
    
    json create_config(const ParameterSet& params) {
        json config;
        
        config["simulation"]["generations"] = 100000;
        config["simulation"]["initial_size"] = 1000;
        config["simulation"]["output_dir"] = "./output";
        config["simulation"]["output_prefix"] = "bootstrap_sim";
        config["simulation"]["track_history"] = false;
        config["simulation"]["number_of_replicates"] = 1;
        config["simulation"]["output"]["save_individual_csvs"] = false;
        config["simulation"]["output"]["save_summary_json"] = false;
        config["simulation"]["output"]["save_consolidated_summary"] = false;
        
        config["biological_parameters"]["s"] = params.s;
        config["biological_parameters"]["m"] = params.m;
        config["biological_parameters"]["q"] = params.q;
        config["biological_parameters"]["r"] = params.r;
        config["biological_parameters"]["l"] = params.l;
        config["biological_parameters"]["idle"] = params.idle;
        
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
    
    std::pair<std::vector<double>, std::vector<double>> run_simulations(const json& config, int num_simulations) {
        std::vector<double> total_resistant_data;
        std::vector<double> transient_fraction_data;
        
        std::mutex data_mutex;
        std::atomic<bool> slow_simulation_detected(false);
        
        unsigned int num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0) num_threads = 4;
        
        int sims_per_thread = num_simulations / num_threads;
        int remainder = num_simulations % num_threads;
        
        std::vector<std::thread> threads;
        int current_idx = 0;
        
        for (unsigned int t = 0; t < num_threads; ++t) {
            int thread_sims = sims_per_thread + (static_cast<int>(t) < remainder ? 1 : 0);
            int start_idx = current_idx;
            int end_idx = current_idx + thread_sims;
            
            threads.emplace_back([&, start_idx, end_idx]() {
                std::vector<double> local_total_resistant;
                std::vector<double> local_transient_fraction;
                
                for (int i = start_idx; i < end_idx; ++i) {
                    // Check if another thread detected a slow simulation
                    if (slow_simulation_detected.load()) {
                        break;
                    }
                    
                    try {
                        TumourSimulation sim(config, 0, i);
                        auto [final_state, clone_data] = sim.run();
                        
                        // Get summary which contains generation information
                        auto summary = sim.get_summary();
                        
                        // Check if simulation reached generation 36500 (too slow)
                        if (summary.final_generation >= 36500) {
                            slow_simulation_detected.store(true);
                            std::cerr << "\n⚠ Simulation " << i << " reached generation " 
                                     << summary.final_generation << " - marking region as too slow" << std::endl;
                            break;
                        }
                        
                        double total_resistant = summary.fraction_transient + summary.fraction_resistant;
                        double transient_frac = (total_resistant > 0) ? 
                            summary.fraction_transient / total_resistant : 0.0;
                        
                        local_total_resistant.push_back(total_resistant);
                        local_transient_fraction.push_back(transient_frac);
                        
                    } catch (const std::exception& e) {
                        std::cerr << "Error in simulation " << i << ": " << e.what() << std::endl;
                    }
                }
                
                std::lock_guard<std::mutex> lock(data_mutex);
                total_resistant_data.insert(total_resistant_data.end(), 
                    local_total_resistant.begin(), local_total_resistant.end());
                transient_fraction_data.insert(transient_fraction_data.end(), 
                    local_transient_fraction.begin(), local_transient_fraction.end());
            });
            
            current_idx = end_idx;
        }
        
        for (auto& thread : threads) {
            thread.join();
        }
        
        // If a slow simulation was detected, throw an exception to signal to caller
        if (slow_simulation_detected.load()) {
            throw std::runtime_error("Slow simulation detected - region marked as problematic");
        }
        
        std::sort(total_resistant_data.begin(), total_resistant_data.end());
        std::sort(transient_fraction_data.begin(), transient_fraction_data.end());
        
        return {total_resistant_data, transient_fraction_data};
    }
    
    struct SimpleBootstrapResult {
        double percentile_50_lower, percentile_50_upper;
        double percentile_75_lower, percentile_75_upper;
        double percentile_90_lower, percentile_90_upper;
    };
    
    SimpleBootstrapResult calculate_bootstrap_ci(const std::vector<double>& data, int num_bootstrap) {
        std::mt19937 local_rng(std::random_device{}());
        std::vector<double> p50_estimates, p75_estimates, p90_estimates;
        
        for (int i = 0; i < num_bootstrap; ++i) {
            std::vector<double> sample;
            sample.reserve(data.size());
            std::uniform_int_distribution<size_t> dist(0, data.size() - 1);
            
            for (size_t j = 0; j < data.size(); ++j) {
                sample.push_back(data[dist(local_rng)]);
            }
            std::sort(sample.begin(), sample.end());
            
            p50_estimates.push_back(calculate_percentile(sample, 50.0));
            p75_estimates.push_back(calculate_percentile(sample, 75.0));
            p90_estimates.push_back(calculate_percentile(sample, 90.0));
        }
        
        std::sort(p50_estimates.begin(), p50_estimates.end());
        std::sort(p75_estimates.begin(), p75_estimates.end());
        std::sort(p90_estimates.begin(), p90_estimates.end());
        
        SimpleBootstrapResult result;
        result.percentile_50_lower = calculate_percentile(p50_estimates, 2.5);
        result.percentile_50_upper = calculate_percentile(p50_estimates, 97.5);
        result.percentile_75_lower = calculate_percentile(p75_estimates, 2.5);
        result.percentile_75_upper = calculate_percentile(p75_estimates, 97.5);
        result.percentile_90_lower = calculate_percentile(p90_estimates, 2.5);
        result.percentile_90_upper = calculate_percentile(p90_estimates, 97.5);
        
        return result;
    }
    
    double calculate_percentile(const std::vector<double>& sorted_data, double percentile) {
        if (sorted_data.empty()) return 0.0;
        double index = (percentile / 100.0) * (sorted_data.size() - 1);
        int lower = static_cast<int>(std::floor(index));
        int upper = static_cast<int>(std::ceil(index));
        if (lower == upper) return sorted_data[lower];
        double weight = index - lower;
        return sorted_data[lower] * (1.0 - weight) + sorted_data[upper] * weight;
    }
    
    bool check_time_budget() {
        auto current = std::chrono::steady_clock::now();
        double elapsed_hours = std::chrono::duration<double>(current - start_time).count() / 3600.0;
        return elapsed_hours < time_budget_hours;
    }
    
    void maybe_add_more_samples() {
        if (simulation_times.empty()) return;
        
        // Estimate remaining time
        auto current = std::chrono::steady_clock::now();
        double elapsed_hours = std::chrono::duration<double>(current - start_time).count() / 3600.0;
        double remaining_hours = time_budget_hours - elapsed_hours;
        
        // Average time per sample (use recent samples for better estimate)
        int recent_window = std::min(10, static_cast<int>(simulation_times.size()));
        double avg_time = 0;
        for (size_t i = simulation_times.size() - recent_window; i < simulation_times.size(); i++) {
            avg_time += simulation_times[i];
        }
        avg_time /= recent_window;
        
        // How many more samples can we do? (with 20% safety margin)
        int possible_additional = static_cast<int>((remaining_hours * 3600.0 * 0.8) / avg_time);
        
        if (possible_additional > 10) {
            // Add samples more aggressively
            int to_add = std::min(possible_additional / 3, 50); // Add up to 50 more at once
            std::cout << "\nAdding " << to_add << " more LHS samples" << std::endl;
            std::cout << "  Estimated time per sample: " << avg_time << " seconds" << std::endl;
            std::cout << "  Remaining time: " << remaining_hours << " hours" << std::endl;
            std::cout << "  Estimated capacity: " << possible_additional << " more samples" << std::endl;
            generate_lhs_samples(to_add);
            save_checkpoint();
        }
    }
    
    void build_and_improve_emulator() {
        const int max_iterations = 10;
        const double target_coverage = 0.95;
        
        bool converged = false;
        
        for (emulator_iteration = 0; emulator_iteration < max_iterations; ++emulator_iteration) {
            std::cout << "\n--- Emulator Iteration " << (emulator_iteration + 1) << " ---" << std::endl;
            
            // Check time budget
            if (!check_time_budget()) {
                std::cout << "Time budget exhausted." << std::endl;
                break;
            }
            
            // Save current training data
            save_training_data();
            
            // Call Python to validate emulator
            std::cout << "Validating GP emulator via Python..." << std::endl;
            if (!call_python_validator()) {
                std::cout << "Error: Python validation failed. Check Python installation and dependencies." << std::endl;
                break;
            }
            
            // Load validation results
            auto validation = load_validation_results();
            if (!validation.has_value()) {
                std::cout << "Error: Could not load validation results." << std::endl;
                break;
            }
            
            double coverage = validation->coverage;
            int problem_count = validation->problem_regions.size();
            
            std::cout << "Emulator coverage: " << (coverage * 100.0) << "% (target: " 
                     << (target_coverage * 100.0) << "%)" << std::endl;
            std::cout << "Problem regions: " << problem_count << std::endl;
            
            // Check if we've reached target coverage
            if (coverage >= target_coverage) {
                std::cout << "✓ Target coverage achieved!" << std::endl;
                converged = true;
                
                // Train final model
                call_python_trainer();
                break;
            }
            
            // Check if we have time/capacity to add more samples
            if (problem_count == 0) {
                std::cout << "No problem regions identified but coverage below target." << std::endl;
                std::cout << "May need more diverse sampling." << std::endl;
                break;
            }
            
            // Estimate remaining capacity
            auto remaining_capacity = estimate_sampling_capacity();
            if (remaining_capacity == 0) {
                std::cout << "No time remaining for additional samples." << std::endl;
                break;
            }
            
            // Add samples in problem regions
            int samples_added = add_samples_near_problems(validation->problem_regions, remaining_capacity);
            
            if (samples_added == 0) {
                std::cout << "Could not add samples in problem regions." << std::endl;
                break;
            }
            
            std::cout << "Added " << samples_added << " targeted samples." << std::endl;
            
            // Process new samples
            std::cout << "Processing new samples..." << std::endl;
            process_new_samples();
            
            save_checkpoint();
        }
        
        // Train final model if we haven't already
        if (!converged) {
            std::cout << "\nTraining final GP emulator..." << std::endl;
            save_training_data();
            call_python_trainer();
        }
    }
    
    bool call_python_validator() {
        // Call Python script to validate emulator
        std::string command = "python gp_emulator.py validate";
        int result = std::system(command.c_str());
        return result == 0;
    }
    
    bool call_python_trainer() {
        // Call Python script to train and save final emulator
        std::string command = "python gp_emulator.py train";
        int result = std::system(command.c_str());
        return result == 0;
    }
    
    std::optional<ValidationResults> load_validation_results() {
        if (!fs::exists(validation_results_file)) {
            return std::nullopt;
        }
        
        try {
            std::ifstream in(validation_results_file);
            json data;
            in >> data;
            in.close();
            
            ValidationResults results;
            results.coverage = data["coverage_stats"]["overall"]["coverage"];
            
            for (const auto& problem : data["problem_regions"]) {
                ProblemRegion region;
                region.sample_id = problem["sample_id"];
                region.total_distance = problem["total_distance"];
                
                for (const auto& param : problem["parameters"]) {
                    region.parameters.push_back(param);
                }
                
                results.problem_regions.push_back(region);
            }
            
            return results;
        } catch (const std::exception& e) {
            std::cerr << "Error loading validation results: " << e.what() << std::endl;
            return std::nullopt;
        }
    }
    
    int estimate_sampling_capacity() {
        if (simulation_times.empty()) return 0;
        
        auto current = std::chrono::steady_clock::now();
        double elapsed_hours = std::chrono::duration<double>(current - start_time).count() / 3600.0;
        double remaining_hours = time_budget_hours - elapsed_hours;
        
        // Average time per sample (recent samples)
        int recent_window = std::min(10, static_cast<int>(simulation_times.size()));
        double avg_time = 0;
        for (size_t i = simulation_times.size() - recent_window; i < simulation_times.size(); i++) {
            avg_time += simulation_times[i];
        }
        avg_time /= recent_window;
        
        // Conservative estimate (70% of remaining time)
        int capacity = static_cast<int>((remaining_hours * 3600.0 * 0.7) / avg_time);
        
        return std::max(0, capacity);
    }
    
    int add_samples_near_problems(const std::vector<ProblemRegion>& problems, int max_samples) {
        if (problems.empty() || max_samples < 5) return 0;
        
        // Limit to top 20 worst regions
        int num_problems = std::min(static_cast<int>(problems.size()), 20);
        
        // Add 2-3 samples near each problem region
        int samples_per_problem = std::max(2, max_samples / num_problems);
        samples_per_problem = std::min(samples_per_problem, 5);
        
        int added = 0;
        
        for (int i = 0; i < num_problems && added < max_samples; ++i) {
            const auto& problem = problems[i];
            
            // Convert log-space parameters back to original space
            ParameterSet base_params;
            base_params.s = std::pow(10, problem.parameters[0]);
            base_params.m = std::pow(10, problem.parameters[1]);
            base_params.q = std::pow(10, problem.parameters[2]);
            base_params.r = std::pow(10, problem.parameters[3]);
            base_params.l = std::pow(10, problem.parameters[4]);
            base_params.idle = problem.parameters[5];
            
            // Add perturbed samples around this region
            for (int j = 0; j < samples_per_problem && added < max_samples; ++j) {
                ParameterSet new_params = perturb_parameters(base_params);
                new_params.sample_id = parameter_samples.size();
                new_params.completed = false;
                parameter_samples.push_back(new_params);
                added++;
            }
        }
        
        std::cout << "  Added " << added << " samples targeting " << num_problems << " problem regions" << std::endl;
        
        return added;
    }
    
    void process_new_samples() {
        for (auto& param_set : parameter_samples) {
            if (param_set.completed) continue;
            
            // Check if in slow region
            if (is_in_slow_region(param_set)) {
                param_set.completed = true;
                continue;
            }
            
            if (!check_time_budget()) {
                std::cout << "Time budget exhausted during new sample processing." << std::endl;
                return;
            }
            
            std::cout << "  Processing sample " << param_set.sample_id << "..." << std::flush;
            
            auto sample_start = std::chrono::steady_clock::now();
            auto results = run_bootstrap_for_parameters(param_set);
            auto sample_end = std::chrono::steady_clock::now();
            
            double sample_time = std::chrono::duration<double>(sample_end - sample_start).count();
            simulation_times.push_back(sample_time);
            
            results_cache[param_set.sample_id] = results;
            param_set.completed = true;
            
            std::cout << " Done (" << sample_time << "s)" << std::endl;
            
            save_checkpoint();
            save_results(param_set.sample_id, param_set, results);
        }
    }
    
    void save_training_data() {
        json emulator_data;
        emulator_data["samples"] = json::array();
        
        int valid_count = 0;
        int skipped_slow_count = 0;
        
        for (const auto& param_set : parameter_samples) {
            if (!param_set.completed) continue;
            
            auto it = results_cache.find(param_set.sample_id);
            if (it == results_cache.end()) continue;
            
            // Skip samples from slow regions (zero bootstrap samples)
            if (it->second.bootstrap_samples == 0) {
                skipped_slow_count++;
                continue;
            }
            
            // Also check if this sample is in a slow region (may have been marked after completion)
            if (is_in_slow_region(param_set)) {
                skipped_slow_count++;
                continue;
            }
            
            json sample;
            sample["parameters"] = param_set.to_json();
            sample["results"] = it->second.to_json();
            emulator_data["samples"].push_back(sample);
            valid_count++;
        }
        
        emulator_data["metadata"] = {
            {"total_samples", parameter_samples.size()},
            {"completed_samples", results_cache.size()},
            {"valid_samples", valid_count},
            {"skipped_slow_regions", skipped_slow_count},
            {"parameter_ranges", {
                {"s", {{"min", 0.001}, {"max", 0.1}}},
                {"m", {{"min", 1e-8}, {"max", 1e-5}}},
                {"q", {{"min", 1e-11}, {"max", 1e-2}}},
                {"r", {{"min", 1e-11}, {"max", 1e-2}}},
                {"l", {{"min", 1e-11}, {"max", 1e-2}}},
                {"idle", {{"min", 0.0}, {"max", 0.5}}}
            }},
            {"constraints", {
                {"birth_greater_than_death", true},
                {"slow_regions_excluded", true}
            }}
        };
        
        std::ofstream out("emulator_training_data.json");
        out << std::setw(4) << emulator_data << std::endl;
        out.close();
        
        std::cout << "  Training data saved (" << valid_count << " valid samples";
        if (skipped_slow_count > 0) {
            std::cout << ", " << skipped_slow_count << " slow region samples excluded";
        }
        std::cout << ")" << std::endl;
    }
    
    void print_final_summary() {
        int completed = std::count_if(parameter_samples.begin(), parameter_samples.end(),
                                     [](const auto& p) { return p.completed; });
        int valid = 0;
        for (const auto& param : parameter_samples) {
            if (param.completed) {
                auto it = results_cache.find(param.sample_id);
                if (it != results_cache.end() && it->second.bootstrap_samples > 0) {
                    valid++;
                }
            }
        }
        
        std::cout << "\nFinal Statistics:" << std::endl;
        std::cout << "  Total samples: " << parameter_samples.size() << std::endl;
        std::cout << "  Completed: " << completed << std::endl;
        std::cout << "  Valid (used for emulator): " << valid << std::endl;
        std::cout << "  Slow regions skipped: " << slow_parameter_regions.size() << std::endl;
        std::cout << "  Emulator iterations: " << emulator_iteration << std::endl;
        
        auto end_time = std::chrono::steady_clock::now();
        double total_hours = std::chrono::duration<double>(end_time - start_time).count() / 3600.0;
        std::cout << "  Total time: " << total_hours << " hours" << std::endl;
        
        if (fs::exists(emulator_model_file)) {
            std::cout << "\n✓ Emulator model saved to: " << emulator_model_file << std::endl;
            std::cout << "  Training data: emulator_training_data.json" << std::endl;
            std::cout << "  Validation results: " << validation_results_file << std::endl;
        }
    }
    
    void save_checkpoint() {
        std::lock_guard<std::mutex> lock(io_mutex);
        
        json checkpoint;
        checkpoint["parameter_samples"] = json::array();
        
        for (const auto& param : parameter_samples) {
            checkpoint["parameter_samples"].push_back(param.to_json());
        }
        
        checkpoint["results_cache"] = json::array();
        for (const auto& [id, results] : results_cache) {
            json result_entry;
            result_entry["sample_id"] = id;
            result_entry["results"] = results.to_json();
            checkpoint["results_cache"].push_back(result_entry);
        }
        
        checkpoint["slow_regions"] = json::array();
        {
            std::lock_guard<std::mutex> slow_lock(slow_regions_mutex);
            for (const auto& slow_params : slow_parameter_regions) {
                checkpoint["slow_regions"].push_back(slow_params.to_json());
            }
        }
        
        checkpoint["emulator_iteration"] = emulator_iteration;
        
        std::ofstream out(checkpoint_file);
        out << std::setw(4) << checkpoint << std::endl;
        out.close();
    }
    
    void load_checkpoint() {
        if (!fs::exists(checkpoint_file)) return;
        
        std::ifstream in(checkpoint_file);
        json checkpoint;
        in >> checkpoint;
        in.close();
        
        for (const auto& param_json : checkpoint["parameter_samples"]) {
            parameter_samples.push_back(ParameterSet::from_json(param_json));
        }
        
        if (checkpoint.contains("slow_regions")) {
            for (const auto& slow_json : checkpoint["slow_regions"]) {
                slow_parameter_regions.push_back(ParameterSet::from_json(slow_json));
            }
            std::cout << "Loaded " << slow_parameter_regions.size() << " slow parameter regions" << std::endl;
        }
        
        if (checkpoint.contains("emulator_iteration")) {
            emulator_iteration = checkpoint["emulator_iteration"];
        }
    }
    
    void save_results(int sample_id, const ParameterSet& params, const BootstrapResults& results) {
        std::lock_guard<std::mutex> lock(io_mutex);
        
        json output;
        output["sample_id"] = sample_id;
        output["parameters"] = params.to_json();
        output["results"] = results.to_json();
        
        std::string filename = results_dir + "/sample_" + std::to_string(sample_id) + ".json";
        std::ofstream out(filename);
        out << std::setw(4) << output << std::endl;
        out.close();
    }
    
    ParameterSet perturb_parameters(const ParameterSet& base) {
        const int max_attempts = 100;
        
        for (int attempt = 0; attempt < max_attempts; ++attempt) {
            std::normal_distribution<double> perturb(0.0, 0.1); // 10% std dev in log-space
            
            ParameterSet result = base;
            
            // Perturb in log-space for log-uniform parameters
            double log_s = std::log10(base.s) + perturb(rng);
            result.s = std::pow(10, std::clamp(log_s, std::log10(0.001), std::log10(0.1)));
            
            double log_m = std::log10(base.m) + perturb(rng);
            result.m = std::pow(10, std::clamp(log_m, -8.0, -5.0));
            
            double log_q = std::log10(base.q) + perturb(rng);
            result.q = std::pow(10, std::clamp(log_q, -11.0, -2.0));
            
            double log_r = std::log10(base.r) + perturb(rng);
            result.r = std::pow(10, std::clamp(log_r, -11.0, -2.0));
            
            double log_l = std::log10(base.l) + perturb(rng);
            result.l = std::pow(10, std::clamp(log_l, -11.0, -2.0));
            
            // Perturb idle in linear space
            std::normal_distribution<double> idle_perturb(0.0, 0.05);
            result.idle = std::clamp(base.idle + idle_perturb(rng), 0.0, 0.5);
            
            // Validate: only return if birth > death AND not in slow region
            if (is_valid_parameter_set(result) && !is_in_slow_region(result)) {
                return result;
            }
        }
        
        // If we couldn't find a valid perturbation, return the base (already validated)
        std::cout << "Warning: Could not find valid perturbation, using base parameters" << std::endl;
        return base;
    }
};

int main(int argc, char* argv[]) {
    try {
        double time_budget = 10.0; // hours
        bool clean_start = false;
        
        // Parse command line arguments
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--clean" || arg == "-c") {
                clean_start = true;
            } else {
                time_budget = std::stod(arg);
            }
        }
        
        // Clean up previous run if requested
        if (clean_start) {
            std::cout << "Cleaning up previous run..." << std::endl;
            fs::remove("bootstrap_checkpoint.json");
            fs::remove("emulator_training_data.json");
            fs::remove("cross_validation_results.json");
            fs::remove_all("bootstrap_results");
            std::cout << "Previous results cleared.\n" << std::endl;
        }
        
        BootstrapManager manager(time_budget);
        manager.run();
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
