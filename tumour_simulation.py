"""
TumourSimulation class: Orchestrates the tumor evolution simulation.

Responsibilities:
- Read and parse JSON configuration
- Initialize clones and treatment protocol
- Execute branching process simulation for specified generations
- Track tumor dynamics and treatment response
- Output results to CSV files
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Optional, Tuple, List
from pathlib import Path

from clone import Clone, CloneCollection
from treatment import Treatment, ScheduleType


class TumourSimulation:
    """
    Main simulation class that coordinates tumor evolution under treatment.
    
    Uses a discrete-time branching process to simulate:
    - Cell birth, death, mutation, and phenotypic transitions
    - Treatment effects on different cell types
    - Clonal dynamics and tumor growth
    """
    
    def __init__(self, config_path: str):
        """
        Initialize simulation from JSON configuration file.
        
        Parameters:
        -----------
        config_path : str
            Path to JSON configuration file
        """
        self.config = self._load_config(config_path)
        self.config_path = config_path
        
        # Extract simulation parameters
        self.generations = self.config['simulation']['generations']
        self.initial_size = self.config['simulation']['initial_size']
        self.output_dir = self.config['simulation'].get('output_dir', './output')
        self.output_prefix = self.config['simulation'].get('output_prefix', 'simulation')
        self.track_detailed_history = self.config['simulation'].get('track_history', False)
        
        # Extract biological parameters
        bio_params = self.config['biological_parameters']
        self.base_params = {
            's': bio_params['s'],
            'u': bio_params['u'],
            'pq': bio_params['pq'],
            'pr': bio_params['pr'],
            'ps': bio_params['ps'],
            'idle': bio_params['idle']
        }
        
        # Initialize components
        self.clones = CloneCollection()
        self.treatment = Treatment(
            config=self.config['treatment'],
            base_params=self.base_params
        )
        
        # Initialize starting clone
        self._initialize_tumor()
        
        # Tracking variables
        self.current_generation = 0
        self.state = 0  # 0: ongoing, 1: extinct, 2: relapsed
        self.history = []
        
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
    
    def _load_config(self, config_path: str) -> Dict:
        """
        Load configuration from JSON file.
        
        Parameters:
        -----------
        config_path : str
            Path to JSON configuration file
            
        Returns:
        --------
        dict
            Configuration dictionary
        """
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    
    def _initialize_tumor(self):
        """Initialize tumor with starting clone of sensitive cells."""
        initial_clone = self.clones.add_clone(
            parent_id=-1,
            parent_type="Sens",
            generation=0,
            n_sensitive=self.initial_size,
            n_quasi=0,
            n_resistant=0
        )
    
    def run(self) -> Tuple[int, pd.DataFrame]:
        """
        Run the simulation for specified number of generations.
        
        Returns:
        --------
        tuple of (int, DataFrame)
            (final_state, results_dataframe)
            final_state: 0=timeout, 1=extinct, 2=relapsed/detected
        """
        print(f"Starting simulation: {self.output_prefix}")
        print(f"Generations: {self.generations}, Initial size: {self.initial_size}")
        start_time = datetime.now()
        
        for generation in range(self.generations):
            self.current_generation = generation
            
            # Remove extinct clones
            self.clones.remove_extinct_clones()
            
            # Check if tumor went extinct
            tumor_size = self.clones.get_total_cells()
            if tumor_size == 0:
                self.state = 1
                print(f"Tumor extinct at generation {generation}")
                break
            
            # Update treatment state
            self.treatment.update_treatment_state(tumor_size, generation)
            
            # Simulate one generation for all clones
            self._simulate_generation()
            
            # Track history if requested
            if self.track_detailed_history and generation % 10 == 0:
                self._record_history()
            
            # Check for relapse/detection
            if self.treatment.relapsed:
                self.state = 2
                print(f"Tumor relapsed at generation {generation}, size: {tumor_size:.2e}")
                break
            
            # Progress updates
            if generation % 100 == 0 and generation > 0:
                counts = self.clones.get_total_by_type()
                print(f"Gen {generation}: Total={counts['total']:.2e}, "
                      f"S={counts['sensitive']:.2e}, Q={counts['quasi']:.2e}, "
                      f"R={counts['resistant']:.2e}, Clones={len(self.clones)}")
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        print(f"Simulation completed in {duration:.2f} seconds")
        
        # Generate results
        results = self._generate_results()
        
        # Save results
        self._save_results(results)
        
        return self.state, results
    
    def _simulate_generation(self):
        """
        Simulate one generation of the branching process for all clones.
        
        For each clone:
        1. Get probabilities based on treatment state
        2. Sample cell fates using multinomial distribution
        3. Update cell counts
        4. Create new clones from mutations
        """
        # Track new clones created this generation (to avoid modifying dict during iteration)
        new_clones = []

        import pprint
        print(f"\n--- Generation {self.current_generation} ---")
        for clone in list(self.clones.clones.values()):
            if clone.is_extinct():
                continue
            print(f"\nClone ID {clone.clone_id} (Gen {clone.generation}): S={clone.n_sensitive}, Q={clone.n_quasi}, R={clone.n_resistant}")
            # Simulate sensitive cells
            if clone.n_sensitive > 0:
                probs_s = self.treatment.calculate_probabilities(clone.generation, "S")
                print("  Sensitive cell probabilities:")
                pprint.pprint(probs_s)
                deltas_s, mutations_s = self._simulate_cell_population(
                    clone, "S", clone.n_sensitive
                )
                print(f"    S deltas: {deltas_s}")
                clone.update_counts(delta_sensitive=deltas_s['net_change'])
                new_clones.extend(mutations_s)
            # Simulate quasi-resistant cells
            if clone.n_quasi > 0:
                probs_q = self.treatment.calculate_probabilities(clone.generation, "Q")
                print("  Quasi cell probabilities:")
                pprint.pprint(probs_q)
                deltas_q, mutations_q = self._simulate_cell_population(
                    clone, "Q", clone.n_quasi
                )
                print(f"    Q deltas: {deltas_q}")
                clone.update_counts(delta_quasi=deltas_q['net_change'])
                new_clones.extend(mutations_q)
            # Simulate resistant cells
            if clone.n_resistant > 0:
                probs_r = self.treatment.calculate_probabilities(clone.generation, "R")
                print("  Resistant cell probabilities:")
                pprint.pprint(probs_r)
                deltas_r, mutations_r = self._simulate_cell_population(
                    clone, "R", clone.n_resistant
                )
                print(f"    R deltas: {deltas_r}")
                clone.update_counts(delta_resistant=deltas_r['net_change'])
                new_clones.extend(mutations_r)
            # Show updated cell counts after all events
            print(f"  Updated: S={clone.n_sensitive}, Q={clone.n_quasi}, R={clone.n_resistant}")
            # Apply state transitions between S, Q, R
            self._apply_state_transitions(clone)
            
        # Add newly created clones
        for new_clone in new_clones:
            self.clones.clones[new_clone.clone_id] = new_clone
    
    def _simulate_cell_population(
        self,
        clone: Clone,
        cell_type: str,
        n_cells: int
    ) -> Tuple[Dict, List[Clone]]:
        """
        Simulate fate of a population of cells of one type.
        
        Parameters:
        -----------
        clone : Clone
            Parent clone
        cell_type : str
            Type of cells ("S", "Q", or "R")
        n_cells : int
            Number of cells to simulate
            
        Returns:
        --------
        tuple of (dict, list)
            (deltas dictionary, list of new mutant clones)
        """
        # Get probabilities for this cell type under current treatment
        probs = self.treatment.calculate_probabilities(clone.generation, cell_type)
        
        # Sample cell fates using multinomial distribution
        # Outcomes: idle, birth, death, mutation, transitions
        fate_probs = [
            probs['prob_idle'],  # idle
            probs['prob_birth'],   # birth
            probs['prob_death'],  # death
            probs['prob_mutation'],   # mutation (creates new clone)
            probs['prob_to_quasi'],   # transition to Q (or revert to S if Q)
            probs['prob_to_resistant']    # transition to R
        ]
        
        # Ensure probabilities sum to 1 (numerical stability)
        prob_sum = sum(fate_probs)
        if prob_sum > 0:
            fate_probs = [p / prob_sum for p in fate_probs]
        
        # Sample fates
        fates = np.random.multinomial(n_cells, fate_probs)
        n_idle, n_birth, n_death, n_mutation, n_transition_q, n_transition_r = fates
        
        # Calculate net change for this cell type
        net_change = n_birth - n_death
        
        # Create new clones from mutations
        new_clones = []
        if n_mutation > 0:
            new_clone = self.clones.add_clone(
                parent_id=clone.clone_id,
                parent_type=cell_type,
                generation=self.current_generation,
                n_sensitive=n_mutation if cell_type == "S" else 0,
                n_quasi=n_mutation if cell_type == "Q" else 0,
                n_resistant=n_mutation if cell_type == "R" else 0
            )
            new_clones.append(new_clone)
            # Mutations leave the parent population
            net_change -= n_mutation
        
        deltas = {
            'net_change': net_change,
            'births': n_birth,
            'deaths': n_death,
            'mutations': n_mutation,
            'transition_q': n_transition_q,
            'transition_r': n_transition_r
        }
        
        return deltas, new_clones
    
    def _apply_state_transitions(self, clone: Clone):
        """
        Apply phenotypic state transitions between S, Q, R within a clone.
        
        This handles transitions like:
        - S -> Q (acquire quasi-resistance)
        - S -> R (acquire full resistance)
        - Q -> S (revert to sensitive)
        - Q -> R (acquire full resistance)
        
        Parameters:
        -----------
        clone : Clone
            Clone to apply transitions to
        """
        # Get probabilities for transitions
        probs_s = self.treatment.calculate_probabilities(clone.generation, "S")
        probs_q = self.treatment.calculate_probabilities(clone.generation, "Q")
        
        # S -> Q transitions
        if clone.n_sensitive > 0:
            n_s_to_q = np.random.binomial(clone.n_sensitive, probs_s['prob_to_quasi'])
            n_s_to_r = np.random.binomial(clone.n_sensitive - n_s_to_q, probs_s['prob_to_resistant'])
            
            clone.n_sensitive -= (n_s_to_q + n_s_to_r)
            clone.n_quasi += n_s_to_q
            clone.n_resistant += n_s_to_r
        
        # Q -> S (reversion) and Q -> R transitions
        if clone.n_quasi > 0:
            # Note: probs_q['prob_to_quasi'] represents reversion rate for quasi cells
            n_q_to_s = np.random.binomial(clone.n_quasi, probs_q['prob_to_quasi'])
            n_q_to_r = np.random.binomial(clone.n_quasi - n_q_to_s, probs_q['prob_to_resistant'])
            
            clone.n_quasi -= (n_q_to_s + n_q_to_r)
            clone.n_sensitive += n_q_to_s
            clone.n_resistant += n_q_to_r
    
    def _record_history(self):
        """Record current state to history."""
        counts = self.clones.get_total_by_type()
        treatment_state = self.treatment.get_state_summary()
        
        history_entry = {
            'generation': self.current_generation,
            'total_cells': counts['total'],
            'sensitive': counts['sensitive'],
            'quasi': counts['quasi'],
            'resistant': counts['resistant'],
            'n_clones': len(self.clones),
            'treatment_active': treatment_state['treatment_active'],
            'doses_given': treatment_state['doses_given']
        }
        self.history.append(history_entry)
    
    def _generate_results(self) -> pd.DataFrame:
        """
        Generate results DataFrame from current clones.
        
        Returns:
        --------
        DataFrame
            Results with clone information
        """
        if len(self.clones) == 0:
            # Extinct tumor
            return pd.DataFrame(columns=['K', 'Id', 'Parent', 'ParentType', 'N', 'Ns', 'Nq', 'Nr', 'T'])
        
        rows = []
        for clone in self.clones:
            row = clone.to_dict()
            row['T'] = self.current_generation
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Sort by clone ID
        df = df.sort_values('Id').reset_index(drop=True)
        
        # Reorder columns to match original format
        column_order = ['K', 'Id', 'Parent', 'ParentType', 'N', 'Ns', 'Nr', 'Nq', 'T']
        df = df[column_order]
        
        return df
    
    def _save_results(self, results: pd.DataFrame):
        """
        Save simulation results to CSV file.
        
        Parameters:
        -----------
        results : DataFrame
            Results to save
        """
        # Generate output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        counts = self.clones.get_total_by_type()
        
        filename = (
            f"{self.output_prefix}_"
            f"s{self.base_params['s']}_"
            f"u{self.base_params['u']}_"
            f"state{self.state}_"
            f"{timestamp}.csv"
        )
        
        filepath = Path(self.output_dir) / filename
        
        # Save main results
        results.to_csv(filepath, index=False)
        print(f"Results saved to: {filepath}")
        
        # Save history if tracked
        if self.history:
            history_filename = f"{self.output_prefix}_history_{timestamp}.csv"
            history_filepath = Path(self.output_dir) / history_filename
            pd.DataFrame(self.history).to_csv(history_filepath, index=False)
            print(f"History saved to: {history_filepath}")
        
        # Save treatment summary
        self._save_treatment_summary(timestamp)
    
    def _save_treatment_summary(self, timestamp: str):
        """
        Save treatment summary information.
        
        Parameters:
        -----------
        timestamp : str
            Timestamp for filename
        """
        treatment_state = self.treatment.get_state_summary()
        counts = self.clones.get_total_by_type()
        
        summary = {
            'config_file': self.config_path,
            'final_generation': self.current_generation,
            'final_state': self.state,
            'final_total_cells': counts['total'],
            'final_sensitive': counts['sensitive'],
            'final_quasi': counts['quasi'],
            'final_resistant': counts['resistant'],
            'final_n_clones': len(self.clones),
            'doses_given': treatment_state['doses_given'],
            'treatment_schedule': self.treatment.schedule_type.value,
            'drug_type': self.treatment.drug_type.value,
            'treatment_active_at_end': treatment_state['treatment_active']
        }
        
        summary_filename = f"{self.output_prefix}_summary_{timestamp}.json"
        summary_filepath = Path(self.output_dir) / summary_filename
        
        with open(summary_filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Summary saved to: {summary_filepath}")
    
    def get_summary(self) -> Dict:
        """
        Get current simulation summary.
        
        Returns:
        --------
        dict
            Summary information
        """
        counts = self.clones.get_total_by_type()
        treatment_state = self.treatment.get_state_summary()
        
        return {
            'generation': self.current_generation,
            'state': self.state,
            'total_cells': counts['total'],
            'sensitive_cells': counts['sensitive'],
            'quasi_cells': counts['quasi'],
            'resistant_cells': counts['resistant'],
            'n_clones': len(self.clones),
            'treatment_active': treatment_state['treatment_active'],
            'doses_given': treatment_state['doses_given']
        }


def run_simulation_from_config(config_path: str) -> Tuple[int, pd.DataFrame]:
    """
    Convenience function to run a simulation from a config file.
    
    Parameters:
    -----------
    config_path : str
        Path to JSON configuration file
        
    Returns:
    --------
    tuple of (int, DataFrame)
        (final_state, results_dataframe)
    """
    simulation = TumourSimulation(config_path)
    return simulation.run()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python tumour_simulation.py <config_file.json>")
        sys.exit(1)
    
    config_file = sys.argv[1]
    state, results = run_simulation_from_config(config_file)
    
    print(f"\nFinal state: {state}")
    print(f"Final clones: {len(results)}")
    print(f"Final tumor size: {results['N'].sum():.2e}")
