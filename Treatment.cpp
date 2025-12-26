#include "Treatment.h"
#include <cmath>
#include <algorithm>

Treatment::Treatment(const json& config, const BaseParams& base_params)
    : base_params(base_params), current_generation(0), drug_iterations(0),
      drug_iterations2(0), doses_given(0), treatment_active(false),
      secondary_active(false), treatment_started(false), relapsed(false) {
    
    // Parse configuration
    std::string sched = config.value("schedule_type", "off");
    if (sched == "continuous") schedule_type = ScheduleType::CONTINUOUS;
    else if (sched == "mtd") schedule_type = ScheduleType::MTD;
    else if (sched == "adaptive") schedule_type = ScheduleType::ADAPTIVE;
    else schedule_type = ScheduleType::OFF;
    
    std::string drug = config.value("drug_type", "abs");
    drug_type = (drug == "prop") ? DrugType::PROPORTIONAL : DrugType::ABSOLUTE;
    
    std::string kill = config.value("kill_mechanism", "kbe");
    kill_mechanism = (kill == "kod") ? KillMechanism::KOD : KillMechanism::KBE;
    
    treat_amt = config.value("treat_amt", 0.8);
    pen_amt = config.value("pen_amt", 4.0);
    dose_duration = config.value("dose_duration", 24);
    secondary_therapy_enabled = config.value("secondary_therapy", false);
    secondary_therapy_type = config.value("secondary_therapy_type", "plast");
    treatment_start_size = config.value("treatment_start_size", 1000000000LL);
    treatment_stop_size = config.value("treatment_stop_size", 1000000000LL);
    relapse_size = config.value("relapse_size", 4000000000LL);
    penalty_enabled = config.value("penalty", false);
    
    compute_probability_table();
}

void Treatment::compute_probability_table() {
    // Compute first 10 k values as requested
    for (int k = 1; k <= 10; k++) {
        double prob_death = ((1 - base_params.idle) / 2.0) * std::pow(1 - base_params.s, k);
        double prob_mutation = base_params.m * (1 - prob_death);
        double prob_to_transient = base_params.q * (1 - prob_death);
        double prob_to_resistant = base_params.r * (1 - prob_death);
        double prob_to_sensitive = base_params.l * (1 - prob_death);
        double prob_birth = 1 - base_params.idle - prob_death - prob_mutation - 
                           prob_to_transient - prob_to_resistant - prob_to_sensitive;
        
        prob_table[k] = {base_params.idle, prob_birth, prob_death, prob_mutation,
                        prob_to_transient, prob_to_resistant, prob_to_sensitive};
    }
}

void Treatment::extend_probability_table(int new_k) {
    // Find current maximum k value
    int current_max = 0;
    if (!prob_table.empty()) {
        for (const auto& pair : prob_table) {
            if (pair.first > current_max) {
                current_max = pair.first;
            }
        }
    }
    
    for (int k = current_max + 1; k <= new_k; k++) {
        double prob_death = ((1 - base_params.idle) / 2.0) * std::pow(1 - base_params.s, k);
        double prob_mutation = base_params.m * (1 - prob_death);
        double prob_to_transient = base_params.q * (1 - prob_death);
        double prob_to_resistant = base_params.r * (1 - prob_death);
        double prob_to_sensitive = base_params.l * (1 - prob_death);
        double prob_birth = 1 - base_params.idle - prob_death - prob_mutation -
                           prob_to_transient - prob_to_resistant - prob_to_sensitive;
        
        prob_table[k] = {base_params.idle, prob_birth, prob_death, prob_mutation,
                        prob_to_transient, prob_to_resistant, prob_to_sensitive};
    }
}

Probabilities Treatment::get_base_probabilities(int k) {
    if (prob_table.find(k) == prob_table.end()) {
        extend_probability_table(k);
    }
    return prob_table[k];
}

// ...existing code for update_treatment_state...

void Treatment::update_treatment_state(long long tumor_size, int generation) {
    current_generation = generation;
    
    if (schedule_type == ScheduleType::OFF) {
        treatment_active = false;
        secondary_active = false;
        return;
    }
    
    if (schedule_type == ScheduleType::CONTINUOUS) {
        treatment_active = true;
        secondary_active = secondary_therapy_enabled;
        drug_iterations++;
        drug_iterations2++;
    } else if (schedule_type == ScheduleType::MTD || schedule_type == ScheduleType::ADAPTIVE) {
        if (!treatment_started && tumor_size > treatment_start_size) {
            treatment_started = true;
            treatment_active = true;
            secondary_active = secondary_therapy_enabled;
            drug_iterations = 0;
            drug_iterations2 = 0;
        }
        
        if (treatment_started) {
            if (treatment_active) {
                drug_iterations++;
                drug_iterations2++;
                
                if (drug_iterations >= dose_duration) {
                    drug_iterations = 0;
                    doses_given++;
                    
                    if (schedule_type == ScheduleType::ADAPTIVE && tumor_size < treatment_stop_size) {
                        treatment_active = false;
                        secondary_active = false;
                    }
                }
                
                if (tumor_size > relapse_size) {
                    relapsed = true;
                }
            } else if (schedule_type == ScheduleType::ADAPTIVE && tumor_size > treatment_start_size) {
                treatment_active = true;
                secondary_active = secondary_therapy_enabled;
                drug_iterations = 0;
                drug_iterations2 = 0;
                doses_given++;
            }
        }
    }
}

std::pair<double, double> Treatment::get_drug_concentration() const {
    if (!treatment_active) return {0.0, 0.0};
    
    double conc1 = std::max(0.0, 1.0 - static_cast<double>(drug_iterations) / dose_duration);
    double conc2 = secondary_active ? std::max(0.0, 1.0 - static_cast<double>(drug_iterations2) / dose_duration) : 0.0;
    
    return {conc1, conc2};
}

Probabilities Treatment::calculate_probabilities(int k, const std::string& cell_type) {
    Probabilities probs = get_base_probabilities(k);
    
    // Zero out impossible transitions
    if (cell_type == "S") {
        probs.prob_to_sensitive = 0.0;
    } else if (cell_type == "Q") {
        probs.prob_to_transient = 0.0;
    } else if (cell_type == "R") {
        probs.prob_to_transient = 0.0;
        probs.prob_to_sensitive = 0.0;
        probs.prob_to_resistant = 0.0;
    }
    
    // Rebalance
    probs.prob_birth = 1.0 - probs.prob_idle - probs.prob_death - probs.prob_mutation -
                       probs.prob_to_transient - probs.prob_to_resistant - probs.prob_to_sensitive;
    
    auto [conc1, conc2] = get_drug_concentration();
    
    if (treatment_active) {
        if (cell_type == "S") {
            probs = apply_treatment_to_sensitive(probs, conc1, conc2);
        } else if (cell_type == "Q") {
            probs = apply_treatment_to_transient(probs, conc1, conc2);
        } else if (cell_type == "R") {
            probs = apply_treatment_to_resistant(probs, conc1, conc2);
        }
    }
    
    if (penalty_enabled && !treatment_active && (cell_type == "Q" || cell_type == "R")) {
        probs = apply_penalty(probs);
    }
    
    // Normalize
    if (treatment_active || (penalty_enabled && !treatment_active && (cell_type == "Q" || cell_type == "R"))) {
        double total = probs.prob_idle + probs.prob_birth + probs.prob_death + probs.prob_mutation +
                      probs.prob_to_transient + probs.prob_to_resistant + probs.prob_to_sensitive;
        if (total > 0) {
            probs.prob_idle /= total;
            probs.prob_birth /= total;
            probs.prob_death /= total;
            probs.prob_mutation /= total;
            probs.prob_to_transient /= total;
            probs.prob_to_resistant /= total;
            probs.prob_to_sensitive /= total;
        }
    }
    
    return probs;
}

// ...existing code for treatment application methods...

Probabilities Treatment::apply_treatment_to_sensitive(Probabilities probs, double conc1, double conc2) {
    if (kill_mechanism == KillMechanism::KBE) {
        probs.prob_death = probs.prob_death + (treat_amt - probs.prob_death) * conc1;
        probs.prob_birth = 1 - probs.prob_death - probs.prob_mutation - probs.prob_to_transient -
                          probs.prob_to_resistant - probs.prob_idle - probs.prob_to_sensitive;
    } else { // KOD
        double diff = std::min((probs.prob_birth - probs.prob_death) * 3 * conc1, 0.9 * probs.prob_birth);
        probs.prob_birth -= diff;
        probs.prob_death += diff;
    }
    
    if (secondary_active && conc2 > 0) {
        probs.prob_idle += probs.prob_to_transient * conc2;
        probs.prob_to_transient -= probs.prob_to_transient * conc2;
    }
    
    return probs;
}

Probabilities Treatment::apply_treatment_to_transient(Probabilities probs, double conc1, double conc2) {
    double diff = (probs.prob_birth - probs.prob_death) * conc1;
    probs.prob_birth += diff / pen_amt;
    probs.prob_death -= diff / pen_amt;
    
    if (secondary_active && conc2 > 0) {
        if (secondary_therapy_type == "plast") {
            double boost = 2 * base_params.q * conc2;
            probs.prob_idle = std::max(0.0, probs.prob_idle - boost);
            probs.prob_to_sensitive += boost;
        } else {
            double red = std::min({probs.prob_birth - 0.01, probs.prob_death - 0.01, 0.1});
            probs.prob_birth -= red * conc2;
            probs.prob_death -= red * conc2;
            probs.prob_to_sensitive += 2 * red * conc2;
        }
    }
    
    return probs;
}

Probabilities Treatment::apply_treatment_to_resistant(Probabilities probs, double conc1, double conc2) {
    double diff = (probs.prob_birth - probs.prob_death) * conc1;
    probs.prob_birth += diff / pen_amt;
    probs.prob_death -= diff / pen_amt;
    return probs;
}

Probabilities Treatment::apply_penalty(Probabilities probs) {
    double diff = probs.prob_birth - probs.prob_death;
    probs.prob_birth -= diff / (pen_amt + 1);
    probs.prob_death += diff / (pen_amt + 1);
    return probs;
}
