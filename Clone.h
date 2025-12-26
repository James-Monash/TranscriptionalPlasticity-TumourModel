#ifndef CLONE_H
#define CLONE_H

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <memory>

struct CloneData {
    int clone_id;
    int parent_id;
    std::string parent_type;
    int generation;
    int n_driver_mutations;
    long long n_sensitive;
    long long n_transient;
    long long n_resistant;
};

class Clone {
public:
    Clone(int clone_id, int parent_id, const std::string& parent_type,
          int generation, long long n_sensitive = 0, long long n_transient = 0,
          long long n_resistant = 0, int n_driver_mutations = 1);
    
    long long total_cells() const;
    void update_counts(long long delta_sensitive, long long delta_transient, 
                      long long delta_resistant);
    bool is_extinct() const;
    CloneData to_dict() const;
    
    // Public members for direct access (similar to Python)
    int clone_id;
    int parent_id;
    std::string parent_type;
    int generation;
    int n_driver_mutations;
    long long n_sensitive;
    long long n_transient;
    long long n_resistant;
};

class CloneCollection {
public:
    CloneCollection();
    
    Clone* add_clone(int parent_id, const std::string& parent_type, int generation,
                    long long n_sensitive = 0, long long n_transient = 0,
                    long long n_resistant = 0, int n_driver_mutations = 1);
    
    Clone* get_clone(int clone_id);
    void remove_extinct_clones();
    const std::unordered_set<int>& get_unique_driver_mutations() const;
    long long get_total_cells() const;
    long long get_total_resistant() const;
    
    struct CellCounts {
        long long sensitive;
        long long transient;
        long long resistant;
        long long total;
    };
    CellCounts get_total_by_type() const;
    
    std::vector<CloneData> to_dataframe_rows() const;
    size_t size() const { return clones.size(); }
    
    // Iterator support
    auto begin() { return clones.begin(); }
    auto end() { return clones.end(); }
    auto begin() const { return clones.begin(); }
    auto end() const { return clones.end(); }
    
private:
    std::unordered_map<int, std::unique_ptr<Clone>> clones;
    int next_id;
    std::unordered_set<int> unique_driver_mutations;
};

#endif // CLONE_H
