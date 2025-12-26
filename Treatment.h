#ifndef TREATMENT_H
#define TREATMENT_H

#include <string>
#include <unordered_map>
#include <utility>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

enum class DrugType { ABSOLUTE, PROPORTIONAL };
enum class KillMechanism { KBE, KOD };
enum class ScheduleType { CONTINUOUS, MTD, ADAPTIVE, FIXED_DOSES, CUSTOM, THRESHOLD, INTERMITTENT, OFF };

struct Probabilities {
    double prob_idle;
    double prob_birth;
    double prob_death;
    double prob_mutation;
    double prob_to_transient;
    double prob_to_resistant;
    double prob_to_sensitive;
};

struct BaseParams {
    double s, m, q, r, l, idle;
};

class Treatment {
public:
    Treatment(const json& config, const BaseParams& base_params);
    
    void update_treatment_state(long long tumor_size, int generation);
    std::pair<double, double> get_drug_concentration() const;
    Probabilities calculate_probabilities(int k, const std::string& cell_type);
    Probabilities get_base_probabilities(int k);
    
    // State accessors
    bool is_treatment_active() const { return treatment_active; }
    bool is_relapsed() const { return relapsed; }
    int get_doses_given() const { return doses_given; }
    long long get_relapse_size() const { return relapse_size; }
    
private:
    void compute_probability_table();
    void extend_probability_table(int new_k);
    Probabilities apply_treatment_to_sensitive(Probabilities probs, double conc1, double conc2);
    Probabilities apply_treatment_to_transient(Probabilities probs, double conc1, double conc2);
    Probabilities apply_treatment_to_resistant(Probabilities probs, double conc1, double conc2);
    Probabilities apply_penalty(Probabilities probs);
    
    // Configuration
    ScheduleType schedule_type;
    DrugType drug_type;
    KillMechanism kill_mechanism;
    double treat_amt;
    double pen_amt;
    int dose_duration;
    bool secondary_therapy_enabled;
    std::string secondary_therapy_type;
    long long treatment_start_size;
    long long treatment_stop_size;
    long long relapse_size;
    bool penalty_enabled;
    
    // Base parameters
    BaseParams base_params;
    
    // State
    int current_generation;
    int drug_iterations;
    int drug_iterations2;
    int doses_given;
    bool treatment_active;
    bool secondary_active;
    bool treatment_started;
    bool relapsed;
    
    // Probability table
    std::unordered_map<int, Probabilities> prob_table;
};

#endif // TREATMENT_H
