#include "Clone.h"
#include <algorithm>

Clone::Clone(int clone_id, int parent_id, const std::string& parent_type,
             int generation, long long n_sensitive, long long n_transient,
             long long n_resistant, int n_driver_mutations)
    : clone_id(clone_id), parent_id(parent_id), parent_type(parent_type),
      generation(generation), n_driver_mutations(n_driver_mutations),
      n_sensitive(n_sensitive), n_transient(n_transient), n_resistant(n_resistant) {}

long long Clone::total_cells() const {
    return n_sensitive + n_transient + n_resistant;
}

void Clone::update_counts(long long delta_sensitive, long long delta_transient,
                         long long delta_resistant) {
    n_sensitive += delta_sensitive;
    n_transient += delta_transient;
    n_resistant += delta_resistant;
}

bool Clone::is_extinct() const {
    return total_cells() == 0;
}

CloneData Clone::to_dict() const {
    return {clone_id, parent_id, parent_type, generation, 
            n_driver_mutations, n_sensitive, n_transient, n_resistant};
}

// CloneCollection implementation
CloneCollection::CloneCollection() : next_id(0) {}

Clone* CloneCollection::add_clone(int parent_id, const std::string& parent_type,
                                  int generation, long long n_sensitive,
                                  long long n_transient, long long n_resistant,
                                  int n_driver_mutations) {
    auto clone = std::make_unique<Clone>(next_id, parent_id, parent_type,
                                         generation, n_sensitive, n_transient,
                                         n_resistant, n_driver_mutations);
    Clone* ptr = clone.get();
    clones[next_id] = std::move(clone);
    unique_driver_mutations.insert(n_driver_mutations);
    next_id++;
    return ptr;
}

Clone* CloneCollection::get_clone(int clone_id) {
    auto it = clones.find(clone_id);
    return it != clones.end() ? it->second.get() : nullptr;
}

void CloneCollection::remove_extinct_clones() {
    for (auto it = clones.begin(); it != clones.end();) {
        if (it->second->is_extinct()) {
            it = clones.erase(it);
        } else {
            ++it;
        }
    }
}

const std::unordered_set<int>& CloneCollection::get_unique_driver_mutations() const {
    return unique_driver_mutations;
}

long long CloneCollection::get_total_cells() const {
    long long total = 0;
    for (const auto& [id, clone] : clones) {
        total += clone->total_cells();
    }
    return total;
}

long long CloneCollection::get_total_resistant() const {
    long long total = 0;
    for (const auto& [id, clone] : clones) {
        total += clone->n_resistant;
    }
    return total;
}

CloneCollection::CellCounts CloneCollection::get_total_by_type() const {
    CellCounts counts{0, 0, 0, 0};
    for (const auto& [id, clone] : clones) {
        counts.sensitive += clone->n_sensitive;
        counts.transient += clone->n_transient;
        counts.resistant += clone->n_resistant;
    }
    counts.total = counts.sensitive + counts.transient + counts.resistant;
    return counts;
}

std::vector<CloneData> CloneCollection::to_dataframe_rows() const {
    std::vector<CloneData> rows;
    rows.reserve(clones.size());
    for (const auto& [id, clone] : clones) {
        rows.push_back(clone->to_dict());
    }
    return rows;
}
