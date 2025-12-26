#include "TumourSimulation.h"
#include <fstream>
#include <iostream>
#include <ctime>

// Hash function for std::pair in cache
namespace std {
    template<>
    struct hash<std::pair<int, std::string>> {
        size_t operator()(const std::pair<int, std::string>& p) const {
            return std::hash<int>()(p.first) ^ (std::hash<std::string>()(p.second) << 1);
        }
    };
}

TumourSimulation::TumourSimulation(const std::string& config_path, int replicate_number)
    : replicate_number(replicate_number), condition_index(-1), current_generation(0),
      state("ongoing"), total_cell_events(0), total_driver_mutations(0) {
    
    std::ifstream f(config_path);
    f >> config;
    
    // Extract parameters
    generations = config["simulation"].value("generations", 1000000);
    initial_size = config["simulation"]["initial_size"];
    initial_transient = config["simulation"].value("initial_transient", 0LL);
    initial_resistant = config["simulation"].value("initial_resistant", 0LL);
    output_dir = config["simulation"].value("output_dir", "./output");
    output_prefix = config["simulation"].value("output_prefix", "simulation");
    
    // Base parameters
    auto bio = config["biological_parameters"];
    base_params = {bio["s"], bio["m"], bio["q"], bio["r"], bio["l"], bio["idle"]};
    
    // Initialize RNG with seed matching Python's strategy
    unsigned int seed = config["simulation"].value("seed", static_cast<unsigned int>(std::time(nullptr)));
    unsigned int effective_seed = seed + replicate_number;
    rng.seed(effective_seed);
    
    // Initialize treatment
    treatment = std::make_unique<Treatment>(config["treatment"], base_params);
    
    initialize_tumor();
}

TumourSimulation::TumourSimulation(const nlohmann::json& config_obj, int condition_index, int replicate_number)
    : replicate_number(replicate_number), condition_index(condition_index), current_generation(0),
      state("ongoing"), total_cell_events(0), total_driver_mutations(0) {
    
    config = config_obj;
    
    // Extract parameters
    generations = config["simulation"].value("generations", 1000000);
    initial_size = config["simulation"]["initial_size"];
    initial_transient = config["simulation"].value("initial_transient", 0LL);
    initial_resistant = config["simulation"].value("initial_resistant", 0LL);
    output_dir = config["simulation"].value("output_dir", "./output");
    output_prefix = config["simulation"].value("output_prefix", "simulation");
    
    // Base parameters
    auto bio = config["biological_parameters"];
    base_params = {bio["s"], bio["m"], bio["q"], bio["r"], bio["l"], bio["idle"]};
    
    // Initialize RNG with seed matching Python's strategy (100000 offset for conditions)
    unsigned int seed = config["simulation"].value("seed", static_cast<unsigned int>(std::time(nullptr)));
    unsigned int effective_seed = seed + replicate_number + (condition_index * 100000);
    rng.seed(effective_seed);
    
    // Initialize treatment
    treatment = std::make_unique<Treatment>(config["treatment"], base_params);
    
    initialize_tumor();
}

void TumourSimulation::initialize_tumor() {
    clones.add_clone(-1, "Sens", 0, initial_size, initial_transient, initial_resistant, 1);
}

std::pair<std::string, std::vector<CloneData>> TumourSimulation::run() {
    for (current_generation = 0; current_generation < generations; current_generation++) {
        clones.remove_extinct_clones();
        
        long long tumor_size = clones.get_total_cells();
        if (tumor_size == 0) {
            state = "extinct";
            break;
        }
        
        if (tumor_size >= treatment->get_relapse_size()) {
            state = "relapsed";
            break;
        }
        
        treatment->update_treatment_state(tumor_size, current_generation);
        
        static bool prev_active = false;
        if (treatment->is_treatment_active() != prev_active) {
            probability_cache.clear();
            prev_active = treatment->is_treatment_active();
        }
        
        simulate_generation();
        
        if (treatment->is_relapsed()) {
            state = "relapsed";
            break;
        }
    }
    
    return {state, clones.to_dataframe_rows()};
}

void TumourSimulation::simulate_generation() {
    std::vector<Clone*> new_clones;
    
    // Pre-calculate probabilities for unique k values
    for (int k : clones.get_unique_driver_mutations()) {
        for (const auto& cell_type : {"S", "Q", "R"}) {
            auto key = std::make_pair(k, std::string(cell_type));
            if (probability_cache.find(key) == probability_cache.end()) {
                probability_cache[key] = treatment->calculate_probabilities(k, cell_type);
            }
        }
    }
    
    // Simulate each clone
    for (auto& [id, clone_ptr] : clones) {
        Clone* clone = clone_ptr.get();
        int k = clone->n_driver_mutations;
        
        CellDeltas deltas_s{0,0,0,0,0,0}, deltas_q{0,0,0,0,0,0}, deltas_r{0,0,0,0,0,0};
        std::vector<Clone*> mutations_s, mutations_q, mutations_r;
        
        if (clone->n_sensitive > 0) {
            auto probs_s = probability_cache[{k, "S"}];
            auto [deltas, mutations] = simulate_cell_population(clone, "S", clone->n_sensitive, probs_s);
            deltas_s = deltas;
            mutations_s = mutations;
        }
        
        if (clone->n_transient > 0) {
            auto probs_q = probability_cache[{k, "Q"}];
            auto [deltas, mutations] = simulate_cell_population(clone, "Q", clone->n_transient, probs_q);
            deltas_q = deltas;
            mutations_q = mutations;
        }
        
        if (clone->n_resistant > 0) {
            auto probs_r = probability_cache[{k, "R"}];
            auto [deltas, mutations] = simulate_cell_population(clone, "R", clone->n_resistant, probs_r);
            deltas_r = deltas;
            mutations_r = mutations;
        }
        
        // Update clone counts
        clone->update_counts(deltas_s.net_change, deltas_q.net_change, deltas_r.net_change);
        
        // Apply transitions
        clone->n_sensitive -= (deltas_s.transition_q + deltas_s.transition_r);
        clone->n_transient += deltas_s.transition_q;
        clone->n_resistant += deltas_s.transition_r;
        
        clone->n_transient -= (deltas_q.transition_q + deltas_q.transition_r);
        clone->n_sensitive += deltas_q.transition_q;
        clone->n_resistant += deltas_q.transition_r;
        
        // Collect new clones
        new_clones.insert(new_clones.end(), mutations_s.begin(), mutations_s.end());
        new_clones.insert(new_clones.end(), mutations_q.begin(), mutations_q.end());
        new_clones.insert(new_clones.end(), mutations_r.begin(), mutations_r.end());
    }
}

std::pair<TumourSimulation::CellDeltas, std::vector<Clone*>> 
TumourSimulation::simulate_cell_population(Clone* clone, const std::string& cell_type,
                                          long long n_cells, const Probabilities& probs) {
    // Multinomial sampling
    double fate_probs[6];
    if (cell_type == "S") {
        fate_probs[0] = probs.prob_idle;
        fate_probs[1] = probs.prob_birth;
        fate_probs[2] = probs.prob_death;
        fate_probs[3] = probs.prob_mutation;
        fate_probs[4] = probs.prob_to_transient;
        fate_probs[5] = probs.prob_to_resistant;
    } else if (cell_type == "Q") {
        fate_probs[0] = probs.prob_idle;
        fate_probs[1] = probs.prob_birth;
        fate_probs[2] = probs.prob_death;
        fate_probs[3] = probs.prob_mutation;
        fate_probs[4] = probs.prob_to_sensitive;
        fate_probs[5] = probs.prob_to_resistant;
    } else { // R
        fate_probs[0] = probs.prob_idle;
        fate_probs[1] = probs.prob_birth;
        fate_probs[2] = probs.prob_death;
        fate_probs[3] = probs.prob_mutation;
        fate_probs[4] = 0.0;
        fate_probs[5] = 0.0;
    }
    
    // Efficient multinomial sampling using successive binomial draws
    std::vector<long long> fates(6, 0);
    
    if (n_cells > 0) {
        long long remaining = n_cells;
        double cumulative_prob = 0.0;
        
        for (int i = 0; i < 5; i++) {
            if (remaining == 0) break;
            
            double p = fate_probs[i] / (1.0 - cumulative_prob);
            p = std::max(0.0, std::min(1.0, p));
            
            std::binomial_distribution<long long> binom(remaining, p);
            fates[i] = binom(rng);
            
            remaining -= fates[i];
            cumulative_prob += fate_probs[i];
        }
        
        fates[5] = remaining;
    }
    
    long long n_birth = fates[1];
    long long n_death = fates[2];
    long long n_mutation = fates[3];
    long long n_transition_q = fates[4];
    long long n_transition_r = fates[5];
    
    total_cell_events += n_cells;
    total_driver_mutations += n_mutation;
    
    long long net_change = n_birth - n_death;
    
    // Create new clones from mutations
    std::vector<Clone*> new_clones;
    for (long long i = 0; i < n_mutation; i++) {
        Clone* new_clone = clones.add_clone(
            clone->clone_id, cell_type, current_generation,
            (cell_type == "S") ? 1 : 0,
            (cell_type == "Q") ? 1 : 0,
            (cell_type == "R") ? 1 : 0,
            clone->n_driver_mutations + 1
        );
        new_clones.push_back(new_clone);
    }
    
    CellDeltas deltas{net_change, n_birth, n_death, n_mutation, n_transition_q, n_transition_r};
    return {deltas, new_clones};
}

TumourSimulation::Summary TumourSimulation::get_summary() const {
    auto counts = clones.get_total_by_type();
    double total = static_cast<double>(counts.total);
    
    Summary s;
    s.condition_index = condition_index;
    s.replicate_number = replicate_number;
    s.final_generation = current_generation;
    s.final_state = state;
    s.final_total_cells = counts.total;
    s.final_sensitive = counts.sensitive;
    s.final_transient = counts.transient;
    s.final_resistant = counts.resistant;
    s.final_n_clones = clones.size();
    s.fraction_sensitive = total > 0 ? counts.sensitive / total : 0;
    s.fraction_transient = total > 0 ? counts.transient / total : 0;
    s.fraction_resistant = total > 0 ? counts.resistant / total : 0;
    s.doses_given = treatment->get_doses_given();
    s.base_params = base_params;
    s.initial_size = initial_size;
    s.total_cell_events = total_cell_events;
    s.total_driver_mutations = total_driver_mutations;
    s.mutation_rate = total_cell_events > 0 ? static_cast<double>(total_driver_mutations) / total_cell_events : 0;
    
    return s;
}
