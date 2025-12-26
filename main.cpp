#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <filesystem>
#include "TumourSimulation.h"
#include <nlohmann/json.hpp>

#ifdef _OPENMP
#include <omp.h>
#endif

using json = nlohmann::json;
namespace fs = std::filesystem;

void save_results_csv(const std::string& filename, const std::vector<CloneData>& data) {
    std::ofstream file(filename);
    file << "K,Id,Parent,ParentType,N,Ns,Nr,Nq,T\n";
    for (const auto& row : data) {
        file << row.generation << "," << row.clone_id << "," << row.parent_id << ","
             << row.parent_type << "," << (row.n_sensitive + row.n_transient + row.n_resistant) << ","
             << row.n_sensitive << "," << row.n_resistant << "," << row.n_transient << ","
             << row.generation << "\n";
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <config.json> [num_workers]\n";
        return 1;
    }
    
    std::string config_path = argv[1];
    int num_workers = (argc > 2) ? std::atoi(argv[2]) : -1;
    
    // Load config first to check use_multiprocessing setting
    std::ifstream f(config_path);
    json config;
    f >> config;
    
    // Check if multiprocessing should be used (default to true if not specified)
    bool use_multiprocessing = config.value("use_multiprocessing", true);
    
#ifdef _OPENMP
    if (!use_multiprocessing) {
        // Disable OpenMP by setting to 1 thread
        omp_set_num_threads(1);
        std::cout << "Multiprocessing disabled (use_multiprocessing: false) - running sequentially\n";
    } else {
        if (num_workers > 0) {
            omp_set_num_threads(num_workers);
        }
        std::cout << "OpenMP enabled with " << omp_get_max_threads() << " threads\n";
    }
#else
    std::cout << "OpenMP not available - running sequentially\n";
#endif
    
    // Create output directory
    std::string output_dir = config.value("simulation", json::object()).value("output_dir", "./output");
    fs::create_directories(output_dir);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Determine if multi-condition
    bool multi_condition = config.contains("simulations");
    std::vector<TumourSimulation::Summary> all_summaries;
    
    // Check if we should save individual CSVs
    bool save_csvs = config.value("simulation", json::object())
                          .value("output", json::object())
                          .value("save_individual_csvs", false);
    
    if (multi_condition) {
        auto simulations = config["simulations"];
        std::cout << "Found " << simulations.size() << " conditions\n";
        
        for (size_t cond_idx = 0; cond_idx < simulations.size(); cond_idx++) {
            json sim_config = simulations[cond_idx];
            int num_replicates_needed = sim_config["simulation"].value("number_of_replicates", 1);
            
            std::cout << "Condition " << cond_idx << ": Running until " << num_replicates_needed 
                     << " successful replicates...\n";
            
            // Map attempt_num -> (summary, clone_data)
            std::map<int, TumourSimulation::Summary> successful_results;
            std::map<int, std::vector<CloneData>> successful_csvs;
            
            int total_attempts = 0;
            int extinct_count = 0;
            
#pragma omp parallel
            {
                while (true) {
                    int my_attempt_num;
                    bool should_run = false;
                    
#pragma omp critical
                    {
                        // Keep trying until we have N successful attempts (by attempt number)
                        my_attempt_num = total_attempts++;
                        should_run = true;
                        
                        // Check if we have enough successful results from the first N attempts
                        if (successful_results.size() >= static_cast<size_t>(num_replicates_needed)) {
                            // Find the Nth smallest attempt number
                            std::vector<int> attempt_nums;
                            for (const auto& [attempt_num, _] : successful_results) {
                                attempt_nums.push_back(attempt_num);
                            }
                            std::sort(attempt_nums.begin(), attempt_nums.end());
                            
                            int nth_attempt = attempt_nums[num_replicates_needed - 1];
                            
                            // Stop if current attempt is beyond the Nth successful attempt
                            if (my_attempt_num > nth_attempt) {
                                should_run = false;
                            }
                        }
                    }
                    
                    if (!should_run) break;
                    
                    TumourSimulation sim(sim_config, cond_idx, my_attempt_num);
                    auto [state, results] = sim.run();
                    
                    bool is_successful = (state != "extinct");
                    
#pragma omp critical
                    {
                        if (is_successful) {
                            auto summary = sim.get_summary();
                            summary.condition_index = cond_idx;
                            successful_results[my_attempt_num] = summary;
                            
                            if (save_csvs) {
                                successful_csvs[my_attempt_num] = std::move(results);
                            }
                            
                            std::cout << "  Attempt " << my_attempt_num << " succeeded (" 
                                     << successful_results.size() << " successes so far): " << state << "\n";
                        } else {
                            extinct_count++;
                            std::cout << "  Attempt " << my_attempt_num << " extinct\n";
                        }
                    }
                }
            }
            
            // Select first N successful attempts (by attempt number)
            std::vector<int> attempt_nums;
            for (const auto& [attempt_num, _] : successful_results) {
                attempt_nums.push_back(attempt_num);
            }
            std::sort(attempt_nums.begin(), attempt_nums.end());
            
            // Take first N
            std::vector<TumourSimulation::Summary> selected_results;
            for (int i = 0; i < num_replicates_needed && i < static_cast<int>(attempt_nums.size()); ++i) {
                int attempt_num = attempt_nums[i];
                auto summary = successful_results[attempt_num];
                summary.replicate_number = i;
                selected_results.push_back(summary);
                
                std::cout << "  Selected attempt " << attempt_num << " as replicate " << i << "\n";
                
                // Save CSV if needed
                if (save_csvs && successful_csvs.count(attempt_num)) {
                    std::string prefix = sim_config["simulation"].value("output_prefix", "simulation");
                    std::string filename = output_dir + "/" + prefix + "_cond" + 
                                         std::to_string(cond_idx) + "_rep" + std::to_string(i) + ".csv";
                    save_results_csv(filename, successful_csvs[attempt_num]);
                }
            }
            
            // Add to all_summaries
            for (const auto& summary : selected_results) {
                all_summaries.push_back(summary);
            }
            
            std::cout << "  Condition " << cond_idx << " complete: " << selected_results.size() << " replicates, "
                     << extinct_count << " extinct, " << total_attempts << " total attempts\n";
        }
    } else {
        // Single condition
        int num_replicates_needed = config["simulation"].value("number_of_replicates", 1);
        std::cout << "Running until " << num_replicates_needed << " successful replicate(s)...\n";
        
        std::string output_prefix = config["simulation"].value("output_prefix", "simulation");
        
        // Map attempt_num -> (summary, clone_data)
        std::map<int, TumourSimulation::Summary> successful_results;
        std::map<int, std::vector<CloneData>> successful_csvs;
        
        int total_attempts = 0;
        int extinct_count = 0;
        
#pragma omp parallel
        {
            while (true) {
                int my_attempt_num;
                bool should_run = false;
                
#pragma omp critical
                {
                    // Keep trying until we have N successful attempts (by attempt number)
                    my_attempt_num = total_attempts++;
                    should_run = true;
                    
                    // Check if we have enough successful results from the first N attempts
                    if (successful_results.size() >= static_cast<size_t>(num_replicates_needed)) {
                        // Find the Nth smallest attempt number
                        std::vector<int> attempt_nums;
                        for (const auto& [attempt_num, _] : successful_results) {
                            attempt_nums.push_back(attempt_num);
                        }
                        std::sort(attempt_nums.begin(), attempt_nums.end());
                        
                        int nth_attempt = attempt_nums[num_replicates_needed - 1];
                        
                        // Stop if current attempt is beyond the Nth successful attempt
                        if (my_attempt_num > nth_attempt) {
                            should_run = false;
                        }
                    }
                }
                
                if (!should_run) break;
                
                TumourSimulation sim(config_path, my_attempt_num);
                auto [state, results] = sim.run();
                
                bool is_successful = (state != "extinct");
                
#pragma omp critical
                {
                    if (is_successful) {
                        auto summary = sim.get_summary();
                        summary.condition_index = 0;
                        successful_results[my_attempt_num] = summary;
                        
                        if (save_csvs) {
                            successful_csvs[my_attempt_num] = std::move(results);
                        }
                        
                        std::cout << "Attempt " << my_attempt_num << " succeeded (" 
                                 << successful_results.size() << " successes so far): " << state << "\n";
                    } else {
                        extinct_count++;
                        std::cout << "  Attempt " << my_attempt_num << " extinct\n";
                    }
                }
            }
        }
        
        // Select first N successful attempts (by attempt number)
        std::vector<int> attempt_nums;
        for (const auto& [attempt_num, _] : successful_results) {
            attempt_nums.push_back(attempt_num);
        }
        std::sort(attempt_nums.begin(), attempt_nums.end());
        
        // Take first N
        std::vector<TumourSimulation::Summary> selected_results;
        for (int i = 0; i < num_replicates_needed && i < static_cast<int>(attempt_nums.size()); ++i) {
            int attempt_num = attempt_nums[i];
            auto summary = successful_results[attempt_num];
            summary.replicate_number = i;
            selected_results.push_back(summary);
            
            std::cout << "Selected attempt " << attempt_num << " as replicate " << i << "\n";
            
            // Save CSV if needed
            if (save_csvs && successful_csvs.count(attempt_num)) {
                std::string filename = output_dir + "/" + output_prefix + "_rep" + 
                                     std::to_string(i) + ".csv";
                save_results_csv(filename, successful_csvs[attempt_num]);
            }
        }
        
        // Add to all_summaries
        for (const auto& summary : selected_results) {
            all_summaries.push_back(summary);
        }
        
        std::cout << "\nComplete: " << selected_results.size() << " replicates, " << extinct_count 
                 << " extinct, " << total_attempts << " total attempts\n";
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    
    // Save consolidated summary
    std::string output_prefix = config.value("simulation", json::object()).value("output_prefix", "simulation");
    std::string summary_filename = output_dir + "/" + output_prefix + "_summary.csv";
    
    std::ofstream summary_file(summary_filename);
    summary_file << "condition_index,replicate_number,final_generation,final_state,"
                 << "final_total_cells,final_sensitive,final_transient,final_resistant,"
                 << "fraction_sensitive,fraction_transient,fraction_resistant,"
                 << "doses_given,mutation_rate\n";
    
    for (const auto& s : all_summaries) {
        summary_file << s.condition_index << "," << s.replicate_number << ","
                    << s.final_generation << "," << s.final_state << ","
                    << s.final_total_cells << "," << s.final_sensitive << ","
                    << s.final_transient << "," << s.final_resistant << ","
                    << s.fraction_sensitive << "," << s.fraction_transient << ","
                    << s.fraction_resistant << "," << s.doses_given << ","
                    << s.mutation_rate << "\n";
    }
    
    std::cout << "\n===========================================\n";
    std::cout << "SIMULATION COMPLETE\n";
    std::cout << "Total time: " << duration.count() << " seconds\n";
    std::cout << "Output directory: " << fs::absolute(output_dir) << "\n";
    std::cout << "Summary file: " << summary_filename << "\n";
    std::cout << "Successful replicates: " << all_summaries.size() << "\n";
    std::cout << "===========================================\n";
    
    return 0;
}
