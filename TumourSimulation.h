#ifndef TUMOUR_SIMULATION_H
#define TUMOUR_SIMULATION_H

#include <string>
#include <random>
#include <memory>
#include "Clone.h"
#include "Treatment.h"
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// Custom hash function for std::pair<int, std::string>
struct PairHash {
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2>& pair) const {
        auto hash1 = std::hash<T1>{}(pair.first);
        auto hash2 = std::hash<T2>{}(pair.second);
        return hash1 ^ (hash2 << 1);
    }
};

class TumourSimulation {
public:
    TumourSimulation(const std::string& config_path, int replicate_number = 0);
    TumourSimulation(const json& config, int condition_index, int replicate_number);
    
    std::pair<std::string, std::vector<CloneData>> run();
    
    struct Summary {
        int condition_index;
        int replicate_number;
        int final_generation;
        std::string final_state;
        long long final_total_cells;
        long long final_sensitive;
        long long final_transient;
        long long final_resistant;
        int final_n_clones;
        double fraction_sensitive;
        double fraction_transient;
        double fraction_resistant;
        std::string treatment_schedule;
        int doses_given;
        
        BaseParams base_params;
        long long initial_size;
        long long total_cell_events;
        long long total_driver_mutations;
        double mutation_rate;
    };
    
    Summary get_summary() const;
    
private:
    void initialize_tumor();
    void simulate_generation();
    
    struct CellDeltas {
        long long net_change;
        long long births;
        long long deaths;
        long long mutations;
        long long transition_q;
        long long transition_r;
    };
    
    std::pair<CellDeltas, std::vector<Clone*>> simulate_cell_population(
        Clone* clone, const std::string& cell_type, long long n_cells, const Probabilities& probs);
    
    json config;
    int replicate_number;  // Moved before condition_index to match initialization order
    int condition_index;
    int generations;
    long long initial_size;
    long long initial_transient;
    long long initial_resistant;
    std::string output_dir;
    std::string output_prefix;
    
    BaseParams base_params;
    CloneCollection clones;
    std::unique_ptr<Treatment> treatment;
    
    int current_generation;
    std::string state;
    long long total_cell_events;
    long long total_driver_mutations;
    
    std::mt19937_64 rng;
    std::unordered_map<std::pair<int, std::string>, Probabilities, PairHash> probability_cache;
};

#endif // TUMOUR_SIMULATION_H
