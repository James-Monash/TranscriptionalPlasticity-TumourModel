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
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, Optional, Tuple, List
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial

from clone import Clone, CloneCollection
from treatment import Treatment, ScheduleType

# Create NumPy random generator for 64-bit integer support in binomial sampling
# Will be reinitialized in each worker process for multiprocessing
_rng = None
_seed = None

def _get_rng():
    """Get or initialize the RNG for this process."""
    global _rng, _seed
    if _rng is None:
        _rng = np.random.default_rng(_seed)
    return _rng

def _set_seed(seed):
    """Set the global seed for the RNG."""
    global _rng, _seed
    _seed = seed
    _rng = np.random.default_rng(_seed)


class TumourSimulation:
    """
    Main simulation class that coordinates tumor evolution under treatment.
    
    Uses a discrete-time branching process to simulate:
    - Cell birth, death, mutation, and phenotypic transitions
    - Treatment effects on different cell types
    - Clonal dynamics and tumor growth
    """
    
    def __init__(self, config_path: str, config_dict: Optional[Dict] = None, condition_index: Optional[int] = None, replicate_number: Optional[int] = None):
        """
        Initialize simulation from JSON configuration file or dict.
        
        Parameters:
        -----------
        config_path : str
            Path to JSON configuration file
        config_dict : dict, optional
            Pre-loaded configuration dictionary (for multi-condition runs)
        condition_index : int, optional
            Index of condition being run (for multi-condition runs)
        replicate_number : int, optional
            Replicate number for this simulation run
        """
        if config_dict is not None:
            self.config = config_dict
        else:
            self.config = self._load_config(config_path)
        self.config_path = config_path
        self.condition_index = condition_index
        self.replicate_number = replicate_number
        
        # Extract simulation parameters
        self.generations = self.config['simulation'].get('generations', 1000000)  # Default to very large number if not specified
        self.initial_size = self.config['simulation']['initial_size']
        self.initial_transient = self.config['simulation'].get('initial_transient', 0)
        self.initial_resistant = self.config['simulation'].get('initial_resistant', 0)
        self.output_dir = self.config['simulation'].get('output_dir', './output')
        self.output_prefix = self.config['simulation'].get('output_prefix', 'simulation')
        self.seed = self.config['simulation'].get('seed', None)
        # Append condition index and replicate number to output prefix
        if condition_index is not None:
            self.output_prefix = f"{self.output_prefix}_cond{condition_index}"
        if replicate_number is not None:
            self.output_prefix = f"{self.output_prefix}_rep{replicate_number}"
        self.track_detailed_history = self.config['simulation'].get('track_history', False)
        self.enable_live_plot = self.config['simulation'].get('enable_live_plot', False)
        
        # Initialize RNG with seed if provided
        if self.seed is not None:
            # For replicates, create unique seed by adding replicate number
            effective_seed = self.seed
            if replicate_number is not None:
                effective_seed = self.seed + replicate_number
            if condition_index is not None:
                effective_seed = effective_seed + (condition_index * 100000)
            _set_seed(effective_seed)
        else:
            # No seed specified, use default random initialization
            _set_seed(None)
        
        # Get number of replicates (for reporting purposes)
        self.number_of_replicates = self.config['simulation'].get('number_of_replicates', 1)
        
        # Output control parameters
        output_config = self.config['simulation'].get('output', {})
        self.save_individual_csvs = output_config.get('save_individual_csvs', False)
        self.save_summary_json = output_config.get('save_summary_json', False)
        self.save_consolidated_summary = output_config.get('save_consolidated_summary', True)
        self.output_columns = output_config.get('output_columns', None)  # None means all columns
        
        # Extract biological parameters
        bio_params = self.config['biological_parameters']
        self.base_params = {
            's': bio_params['s'],
            'm': bio_params['m'],
            'q': bio_params['q'],
            'r': bio_params['r'],
            'l': bio_params['l'],
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
        self.state = 'ongoing'  # 'ongoing', 'extinct', or 'relapsed'
        self.history = []
        
        # Track mutation statistics
        self.total_cell_events = 0
        self.total_driver_mutations = 0
        
        # Cache for probabilities: {(k, cell_type): probability_dict}
        self.probability_cache = {}
        
        # Track event statistics for validation
        self.event_counters = {
            'idle': 0,
            'birth': 0,
            'death': 0,
            'mutation': 0,
            'S_to_Q': 0,  # Sensitive to transient-resistant
            'S_to_R': 0,  # Sensitive to resistant
            'Q_to_S': 0,  # Transient to sensitive (reversion)
            'Q_to_R': 0,  # Transient to resistant
        }
        # Track which cells could have taken each action
        self.event_opportunities = {
            'idle': 0,
            'birth': 0,
            'death': 0,
            'mutation': 0,
            'S_to_Q': 0,  # Only S cells can do this
            'S_to_R': 0,  # Only S cells can do this
            'Q_to_S': 0,  # Only Q cells can do this
            'Q_to_R': 0,  # Only Q cells can do this
            'any_to_R': 0,  # S or Q cells (not already R)
        }
        
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize plotting variables
        self.plot_data = {
            'generations': [],
            'total_population': [],
            'resistant_population': [],
            'transient_population': []
        }
        self.fig = None
        self.ax = None
        self.total_line = None
        self.resistant_line = None
    
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
        """Initialize tumor with starting clone containing specified cell types."""
        initial_clone = self.clones.add_clone(
            parent_id=-1,
            parent_type="Sens",
            generation=0,
            n_sensitive=self.initial_size,
            n_transient=self.initial_transient,
            n_resistant=self.initial_resistant,
            n_driver_mutations=1
        )
    
    def run(self) -> Tuple[int, pd.DataFrame]:
        """
        Run the simulation for specified number of generations.
        
        Returns:
        --------
        tuple of (str, DataFrame)
            (final_state, results_dataframe)
            final_state: 'ongoing'=timeout, 'extinct'=extinct, 'relapsed'=relapsed/detected
        """
        start_time = datetime.now()
        
        for generation in range(self.generations):
            self.current_generation = generation
            
            # Remove extinct clones
            self.clones.remove_extinct_clones()
            
            # Check if tumor went extinct
            tumor_size = self.clones.get_total_cells()
            
            # Record data for plotting
            if self.enable_live_plot:
                self.plot_data['generations'].append(generation)
                self.plot_data['total_population'].append(tumor_size)
                self.plot_data['resistant_population'].append(self.clones.get_total_resistant())
                counts = self.clones.get_total_by_type()
                self.plot_data['transient_population'].append(counts['transient'])
            
            #print(tumor_size)
            if tumor_size == 0:
                self.state = 'extinct'
                break
            
            # Check if tumor has reached relapse size
            if tumor_size >= self.treatment.relapse_size:
                print(generation)
                self.state = 'relapsed'
                break
            
            # Update treatment state
            prev_treatment_active = self.treatment.treatment_active
            prev_drug_iterations = self.treatment.drug_iterations
            self.treatment.update_treatment_state(tumor_size, generation)
            
            # Clear probability cache if treatment state changed (drug concentration affects probabilities)
            if (self.treatment.treatment_active != prev_treatment_active or 
                self.treatment.drug_iterations != prev_drug_iterations):
                self.probability_cache = {}
            
            # Simulate one generation for all clones
            self._simulate_generation()
            
            # Track history if requested
            if self.track_detailed_history and generation % 10 == 0:
                self._record_history()
            
            # Check for relapse/detection
            if self.treatment.relapsed:
                self.state = 'relapsed'
                break
            
            # Progress updates
            #if generation % 100 == 0 and generation > 0:
                #counts = self.clones.get_total_by_type()
                #mutation_rate = (self.total_driver_mutations / self.total_cell_events * 100) if self.total_cell_events > 0 else 0
                #print(f"Gen {generation}: Total={counts['total']:.2e}, "
                      #f"S={counts['sensitive']:.2e}, Q={counts['transient']:.2e}, "
                      #f"R={counts['resistant']:.2e}, Clones={len(self.clones)}, "
                      #f"Mutations={self.total_driver_mutations:,}/{self.total_cell_events:,} ({mutation_rate:.4f}%)")
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Generate results
        results = self._generate_results() 
        
        # Save results if configured
        if self.save_individual_csvs or self.save_summary_json:
            self._save_results(results)
        
        # Generate plot if enabled
        if self.enable_live_plot:
            self._generate_plot()
        
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
        
        # Get unique driver mutation counts present in current generation
        unique_k_values = self.clones.get_unique_driver_mutations() # O(1) operation; the set is maintained in CloneCollection
        
        # Pre-calculate probabilities for all unique k values and cell types
        # This avoids redundant probability calculations for clones with same k
        for k in unique_k_values:
            for cell_type in ["S", "Q", "R"]:
                cache_key = (k, cell_type)
                if cache_key not in self.probability_cache:
                    self.probability_cache[cache_key] = self.treatment.calculate_probabilities(k, cell_type)

        #print(f"\n--- Generation {self.current_generation} ---")
        # Need list() to create snapshot since new clones are added during iteration
        #print(f"\n--- Generation {self.current_generation} ---")
        # Need list() to create snapshot since new clones are added during iteration
        for clone in list(self.clones.clones.values()):
            #print(f'The total number of cells is {self.clones.get_total_cells()}')
            #print(f"\nClone ID {clone.clone_id} (Gen {clone.generation}): S={clone.n_sensitive}, Q={clone.n_transient}, R={clone.n_resistant}")
            
            # Initialize delta dictionaries
            deltas_s = {'net_change': 0, 'births': 0, 'deaths': 0, 'mutations': 0, 'transition_q': 0, 'transition_r': 0}
            deltas_q = {'net_change': 0, 'births': 0, 'deaths': 0, 'mutations': 0, 'transition_q': 0, 'transition_r': 0}
            deltas_r = {'net_change': 0, 'births': 0, 'deaths': 0, 'mutations': 0, 'transition_q': 0, 'transition_r': 0}
            
            # Get cached probabilities for this clone's driver mutation count
            k = clone.n_driver_mutations
            
            # Simulate sensitive cells
            if clone.n_sensitive > 0:
                probs_s = self.probability_cache.get((k, "S"))
                #print("  Sensitive cell probabilities:")
                #pprint.pprint(probs_s)
                deltas_s, mutations_s = self._simulate_cell_population(
                    clone, "S", clone.n_sensitive, probs_s
                )
                #print(f"    S deltas: {deltas_s}")
                #self._print_event_statistics()
                new_clones.extend(mutations_s)
            
            # Simulate transient-resistant cells
            if clone.n_transient > 0:
                probs_q = self.probability_cache.get((k, "Q"))
                #print("  Transient cell probabilities:")
                #pprint.pprint(probs_q)
                deltas_q, mutations_q = self._simulate_cell_population(
                    clone, "Q", clone.n_transient, probs_q
                )
                #print(f"    Q deltas: {deltas_q}")
                #self._print_event_statistics()
                new_clones.extend(mutations_q)
            
            # Simulate resistant cells
            if clone.n_resistant > 0:
                probs_r = self.probability_cache.get((k, "R"))
                #print("  Resistant cell probabilities:")
                #pprint.pprint(probs_r)
                deltas_r, mutations_r = self._simulate_cell_population(
                    clone, "R", clone.n_resistant, probs_r
                )
                #print(f"    R deltas: {deltas_r}")
                #self._print_event_statistics()
                new_clones.extend(mutations_r)
            
            # Apply all changes to clone counts
            # The net_change already accounts for births, deaths, and mutations
            # Transitions need to be applied separately to move cells between states
            
            # Update counts for each type (births - deaths - mutations that left this type)
            clone.update_counts(
                delta_sensitive=deltas_s['net_change'],
                delta_transient=deltas_q['net_change'],
                delta_resistant=deltas_r['net_change']
            )
            
            # Apply state transitions
            # S -> Q and S -> R transitions
            clone.n_sensitive -= (deltas_s['transition_q'] + deltas_s['transition_r'])
            clone.n_transient += deltas_s['transition_q']
            clone.n_resistant += deltas_s['transition_r']
            
            # Q -> S (reversion) and Q -> R transitions
            # Note: for Q cells, transition_q means reversion to S
            clone.n_transient -= (deltas_q['transition_q'] + deltas_q['transition_r'])
            clone.n_sensitive += deltas_q['transition_q']
            clone.n_resistant += deltas_q['transition_r']
            
            # R cells have no transitions (transition_q and transition_r are always 0)
            
            # Show updated cell counts after all events
            #print(f"  Updated: S={clone.n_sensitive}, Q={clone.n_transient}, R={clone.n_resistant}")
            
        # Add newly created clones
        for new_clone in new_clones:
            self.clones.clones[new_clone.clone_id] = new_clone
    
    def _simulate_cell_population(
        self,
        clone: Clone,
        cell_type: str,
        n_cells: int,
        probs: Dict[str, float]
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
        probs : dict
            Pre-calculated probabilities for this cell type and driver mutation count
            
        Returns:
        --------
        tuple of (dict, list)
            (deltas dictionary, list of new mutant clones)
        """
        # Use provided probabilities instead of recalculating
        
        # Sample cell fates using multinomial distribution
        # Outcomes: idle, birth, death, mutation, transitions
        # Note: transition probabilities depend on cell type:
        #   S cells: can gain Q or R (prob_to_transient, prob_to_resistant)
        #   Q cells: can revert to S or gain R (prob_to_sensitive, prob_to_resistant)
        #   R cells: no transitions (all transition probs are 0)
        
        if cell_type == "S":
            transition_prob_1 = probs['prob_to_transient']  # S -> Q
            transition_prob_2 = probs['prob_to_resistant']  # S -> R
        elif cell_type == "Q":
            transition_prob_1 = probs['prob_to_sensitive']  # Q -> S
            transition_prob_2 = probs['prob_to_resistant']  # Q -> R
        else:  # R cells
            transition_prob_1 = 0  # No transitions
            transition_prob_2 = 0
        
        fate_probs = [
            probs['prob_idle'],     # idle
            probs['prob_birth'],    # birth
            probs['prob_death'],    # death
            probs['prob_mutation'], # mutation (creates new clone)
            transition_prob_1,      # first transition (S->Q or Q->S)
            transition_prob_2       # second transition (S->R or Q->R)
        ]
        
        # Ensure probabilities sum to 1 (numerical stability)
        prob_sum = sum(fate_probs)
        if prob_sum > 0:
            fate_probs = [p / prob_sum for p in fate_probs]
        
        # Sample fates using _rng for 64-bit integer support
        rng = _get_rng()
        fates = rng.multinomial(n_cells, fate_probs)
        # Convert to Python integers immediately to prevent overflow
        n_idle, n_birth, n_death, n_mutation, n_transition_q, n_transition_r = [int(x) for x in fates]
        
        # Track statistics (convert to Python int to avoid overflow)
        self.total_cell_events += int(n_cells)
        self.total_driver_mutations += int(n_mutation)
        
        # Track event counters for validation
        self.event_counters['idle'] += int(n_idle)
        self.event_counters['birth'] += int(n_birth)
        self.event_counters['death'] += int(n_death)
        self.event_counters['mutation'] += int(n_mutation)
        
        # Track opportunities (all cells can idle, birth, die, mutate)
        self.event_opportunities['idle'] += int(n_cells)
        self.event_opportunities['birth'] += int(n_cells)
        self.event_opportunities['death'] += int(n_cells)
        self.event_opportunities['mutation'] += int(n_cells)
        
        # Track state-specific transitions
        if cell_type == "S":
            self.event_counters['S_to_Q'] += int(n_transition_q)
            self.event_counters['S_to_R'] += int(n_transition_r)
            self.event_opportunities['S_to_Q'] += int(n_cells)
            self.event_opportunities['S_to_R'] += int(n_cells)
            self.event_opportunities['any_to_R'] += int(n_cells)
        elif cell_type == "Q":
            self.event_counters['Q_to_S'] += int(n_transition_q)
            self.event_counters['Q_to_R'] += int(n_transition_r)
            self.event_opportunities['Q_to_S'] += int(n_cells)
            self.event_opportunities['Q_to_R'] += int(n_cells)
            self.event_opportunities['any_to_R'] += int(n_cells)
        # R cells have no transition opportunities
        
        # Calculate net change for this cell type
        # Net change = births - deaths
        # Mutations now represent cell birth with mutation, so they add to population
        net_change = n_birth - n_death
        
        # Create new clones from mutations - one clone per mutated cell
        new_clones = []
        if n_mutation > 0:
            for _ in range(n_mutation):
                new_clone = self.clones.add_clone(
                    parent_id=clone.clone_id,
                    parent_type=cell_type,
                    generation=self.current_generation,
                    n_sensitive=1 if cell_type == "S" else 0,
                    n_transient=1 if cell_type == "Q" else 0,
                    n_resistant=1 if cell_type == "R" else 0,
                    n_driver_mutations=clone.n_driver_mutations + 1
                )
                new_clones.append(new_clone)
                #print(f"    NEW CLONE CREATED: Clone {new_clone.clone_id} (parent: {clone.clone_id}, type: {cell_type}, cells: 1, driver mutations: {new_clone.n_driver_mutations})")

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
        - S -> Q (acquire transient-resistance)
        - S -> R (acquire full resistance)
        - Q -> S (revert to sensitive)
        - Q -> R (acquire full resistance)
        
        Parameters:
        -----------
        clone : Clone
            Clone to apply transitions to
        """
        # Get probabilities for transitions
        probs_s = self.treatment.calculate_probabilities(clone.n_driver_mutations, "S")
        probs_q = self.treatment.calculate_probabilities(clone.n_driver_mutations, "Q")
        
        rng = _get_rng()
        
        # S -> Q transitions
        if clone.n_sensitive > 0:
            n_s_to_q = int(rng.binomial(clone.n_sensitive, probs_s['prob_to_transient']))
            n_s_to_r = int(rng.binomial(clone.n_sensitive - n_s_to_q, probs_s['prob_to_resistant']))
            
            clone.n_sensitive -= (n_s_to_q + n_s_to_r)
            clone.n_transient += n_s_to_q
            clone.n_resistant += n_s_to_r
        
        # Q -> S (reversion) and Q -> R transitions
        if clone.n_transient > 0:
            # Note: probs_q['prob_to_sensitive'] represents reversion rate for transient cells
            n_q_to_s = int(rng.binomial(clone.n_transient, probs_q['prob_to_sensitive']))
            n_q_to_r = int(rng.binomial(clone.n_transient - n_q_to_s, probs_q['prob_to_resistant']))
            
            clone.n_transient -= (n_q_to_s + n_q_to_r)
            clone.n_sensitive += n_q_to_s
            clone.n_resistant += n_q_to_r
    
    def _print_event_statistics(self):

        """Print empirical event frequencies vs theoretical probabilities for validation."""
        if self.total_cell_events == 0:
            return
        
        print(f"    Empirical event frequencies (observed/opportunities):")
        
        # Universal events (all cell types)
        for event in ['idle', 'birth', 'death', 'mutation']:
            count = self.event_counters[event]
            opps = self.event_opportunities[event]
            freq = (count / opps * 100) if opps > 0 else 0
            print(f"      {event:12s}: {count:12,}/{opps:12,} = {freq:8.4f}%")
        
        # State-specific transitions
        for event, label in [
            ('S_to_Q', 'S -> Q'),
            ('S_to_R', 'S -> R'),
            ('Q_to_S', 'Q -> S'),
            ('Q_to_R', 'Q -> R')
        ]:
            count = self.event_counters[event]
            opps = self.event_opportunities[event]
            if opps > 0:
                freq = count / opps * 100
                print(f"      {label:12s}: {count:12,}/{opps:12,} = {freq:8.4f}%")
        
        # Combined "any to R" transition
        any_to_r = self.event_counters['S_to_R'] + self.event_counters['Q_to_R']
        any_to_r_opps = self.event_opportunities['any_to_R']
        if any_to_r_opps > 0:
            freq = any_to_r / any_to_r_opps * 100
            print(f"      {'Any -> R':12s}: {any_to_r:12,}/{any_to_r_opps:12,} = {freq:8.4f}%")
    
    def _record_history(self):
        """Record current state to history."""
        counts = self.clones.get_total_by_type()
        treatment_state = self.treatment.get_state_summary()
        
        history_entry = {
            'generation': self.current_generation,
            'total_cells': counts['total'],
            'sensitive': counts['sensitive'],
            'transient': counts['transient'],
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
        Save simulation results to CSV file based on output configuration.
        
        Parameters:
        -----------
        results : DataFrame
            Results to save
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        counts = self.clones.get_total_by_type()
        
        # Save individual CSV if requested
        if self.save_individual_csvs:
            filename = (
                f"{self.output_prefix}_"
                f"s{self.base_params['s']}_"
                f"m{self.base_params['m']}_"
                f"state{self.state}_"
                f"{timestamp}.csv"
            )
            filepath = Path(self.output_dir) / filename
            results.to_csv(filepath, index=False)
            print(f"Individual results saved to: {filepath}")
        
        # Save history if tracked
        if self.history and self.save_individual_csvs:
            history_filename = f"{self.output_prefix}_history_{timestamp}.csv"
            history_filepath = Path(self.output_dir) / history_filename
            pd.DataFrame(self.history).to_csv(history_filepath, index=False)
            print(f"History saved to: {history_filepath}")
        
        # Save treatment summary JSON if requested
        if self.save_summary_json:
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
            'final_transient': counts['transient'],
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
            'transient_cells': counts['transient'],
            'resistant_cells': counts['resistant'],
            'n_clones': len(self.clones),
            'treatment_active': treatment_state['treatment_active'],
            'doses_given': treatment_state['doses_given']
        }
    
    def _generate_plot(self):
        """Generate and display the population plot at the end of simulation."""
        if not self.plot_data['generations']:
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.plot_data['generations'], self.plot_data['total_population'], 
                'b-', label='Total population', linewidth=2)
        plt.plot(self.plot_data['generations'], self.plot_data['transient_population'], 
                'orange', label='Transient-resistant cells', linewidth=2)
        plt.plot(self.plot_data['generations'], self.plot_data['resistant_population'], 
                'r-', label='Resistant cells', linewidth=2)
        plt.xlabel('Generation', fontsize=12)
        plt.ylabel('Population size', fontsize=12)
        plt.title('Tumor Population Dynamics', fontsize=14)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def get_final_summary_row(self) -> Dict:
        """
        Get a summary row for this simulation suitable for aggregation.
        
        Returns:
        --------
        dict
            Dictionary with key metrics from this simulation run
        """
        counts = self.clones.get_total_by_type()
        treatment_state = self.treatment.get_state_summary()
        
        summary = {
            'condition_index': self.condition_index if self.condition_index is not None else 0,
            'replicate_number': self.replicate_number if self.replicate_number is not None else 0,
            'final_generation': self.current_generation,
            'final_state': self.state,
            'initial_size': self.initial_size,
            
            # Biological parameters
            's': self.base_params['s'],
            'm': self.base_params['m'],
            'q': self.base_params['q'],
            'r': self.base_params['r'],
            'l': self.base_params['l'],
            'idle': self.base_params['idle'],
            
            # Final cell counts
            'final_total_cells': counts['total'],
            'final_sensitive': counts['sensitive'],
            'final_transient': counts['transient'],
            'final_resistant': counts['resistant'],
            'final_n_clones': len(self.clones),
            
            # Fractions
            'fraction_sensitive': counts['sensitive'] / counts['total'] if counts['total'] > 0 else 0,
            'fraction_transient': counts['transient'] / counts['total'] if counts['total'] > 0 else 0,
            'fraction_resistant': counts['resistant'] / counts['total'] if counts['total'] > 0 else 0,
            
            # Treatment info
            'treatment_schedule': self.treatment.schedule_type.value,
            'drug_type': self.treatment.drug_type.value,
            'doses_given': treatment_state['doses_given'],
            'treatment_active_at_end': treatment_state['treatment_active'],
            
            # Mutation statistics
            'total_cell_events': self.total_cell_events,
            'total_driver_mutations': self.total_driver_mutations,
            'mutation_rate': self.total_driver_mutations / self.total_cell_events if self.total_cell_events > 0 else 0
        }
        
        return summary


def _init_worker(base_seed=None):
    """Initialize worker process with fresh RNG using process-safe seeding."""
    global _rng, _seed
    import os
    # Use process ID to create a unique but reproducible seed for this worker
    # This ensures each worker has independent random streams
    pid = os.getpid()
    if base_seed is not None:
        # If base seed provided, combine it with PID for reproducible but independent streams
        _seed = base_seed + pid
    else:
        # No seed specified, use PID only
        _seed = pid
    _rng = np.random.default_rng(_seed)


def _run_single_replicate(args):
    """
    Run a single replicate simulation. Used for multiprocessing.
    
    Parameters:
    -----------
    args : tuple
        (config_path, config_dict, condition_index, replicate_number)
        
    Returns:
    --------
    tuple
        (condition_index, replicate_number, state, dataframe, summary_dict)
    """
    config_path, config_dict, condition_index, replicate_number = args
    
    try:
        simulation = TumourSimulation(
            config_path=config_path,
            config_dict=config_dict,
            condition_index=condition_index,
            replicate_number=replicate_number
        )
        state, df = simulation.run()
        summary = simulation.get_final_summary_row()
        return (condition_index, replicate_number, state, df, summary)
    except Exception as e:
        print(f"Error in condition {condition_index}, replicate {replicate_number}: {e}")
        import traceback
        traceback.print_exc()
        return (condition_index, replicate_number, 'error', pd.DataFrame(), {})


def _run_condition_until_target_sequential(config_path: str, sim_config: Dict, condition_index: Optional[int], 
                                           target_successful: int) -> List[Tuple[str, pd.DataFrame]]:
    """
    Run simulations for a single condition until target number of successful runs achieved.
    SEQUENTIAL VERSION (no multiprocessing) for debugging.
    
    Parameters:
    -----------
    config_path : str
        Path to config file
    sim_config : dict
        Configuration for this condition
    condition_index : int or None
        Index of this condition
    target_successful : int
        Target number of successful (non-extinct) simulations
        
    Returns:
    --------
    list of tuple
        List of (state, dataframe) for successful simulations
    list of dict
        List of summary dictionaries
    dict
        Statistics dictionary with attempt counts
    """
    successful_results = []
    successful_summaries = []
    trial_number = 0
    total_attempts = 0
    
    while len(successful_results) < target_successful:
        # Run one simulation at a time
        args = (config_path, sim_config, condition_index, trial_number)
        cond_idx, rep_num, state, df, summary = _run_single_replicate(args)
        
        total_attempts += 1
        trial_number += 1
        
        # Check if successful (non-extinct)
        if state != 'extinct' and state != 'error':
            successful_results.append((state, df))
            successful_summaries.append(summary)
        
        # Progress update every 10 attempts
        if total_attempts % 10 == 0 or len(successful_results) >= target_successful:
            print(f"  Progress: {len(successful_results)}/{target_successful} successful "
                  f"({total_attempts} total attempts, "
                  f"{total_attempts - len(successful_results)} extinct)")
    
    stats = {
        'successful': len(successful_results),
        'total_attempts': total_attempts,
        'extinct': total_attempts - len(successful_results),
        'extinction_rate': (total_attempts - len(successful_results)) / total_attempts if total_attempts > 0 else 0
    }
    
    return successful_results, successful_summaries, stats


def _run_condition_until_target(config_path: str, sim_config: Dict, condition_index: Optional[int], 
                                target_successful: int, num_workers: int) -> List[Tuple[str, pd.DataFrame]]:
    """
    Run simulations for a single condition until target number of successful runs achieved.
    
    Parameters:
    -----------
    config_path : str
        Path to config file
    sim_config : dict
        Configuration for this condition
    condition_index : int or None
        Index of this condition
    target_successful : int
        Target number of successful (non-extinct) simulations
    num_workers : int
        Number of parallel workers
        
    Returns:
    --------
    list of tuple
        List of (state, dataframe) for successful simulations
    dict
        Statistics dictionary with attempt counts
    """
    successful_results = []
    successful_summaries = []
    trial_number = 0
    total_attempts = 0
    total_extinct = 0
    
    # Use a pool that persists across batches with initializer
    # Get seed from config if available
    base_seed = sim_config.get('simulation', {}).get('seed', None)
    with Pool(processes=num_workers, initializer=_init_worker, initargs=(base_seed,)) as pool:
        while len(successful_results) < target_successful:
            # Run batches to account for extinctions
            # Estimate batch size based on success rate so far
            if total_attempts > 0 and len(successful_results) > 0:
                success_rate = len(successful_results) / total_attempts
                # Estimate how many more attempts we need, with buffer
                estimated_needed = int((target_successful - len(successful_results)) / max(success_rate, 0.1)) + 5
                batch_size = min(estimated_needed, num_workers * 3)
            else:
                # Initial batch - assume we might need more than target due to extinctions
                batch_size = min(target_successful * 2, num_workers * 3)
            
            # Don't create more tasks than we need
            remaining_needed = target_successful - len(successful_results)
            batch_size = min(batch_size, remaining_needed + int(remaining_needed * 0.1))  # Add 10% buffer for potential extinctions
            
            # Create batch of tasks
            batch_tasks = []
            for _ in range(batch_size):
                batch_tasks.append((config_path, sim_config, condition_index, trial_number))
                trial_number += 1
            
            # Run batch in parallel
            if num_workers > 1:
                batch_results = pool.map(_run_single_replicate, batch_tasks)
            else:
                batch_results = [_run_single_replicate(task) for task in batch_tasks]
            
            # Count outcomes in this batch
            batch_extinct = sum(1 for _, _, state, _, _ in batch_results if state == 'extinct')
            batch_success = sum(1 for _, _, state, _, _ in batch_results if state != 'extinct' and state != 'error')
            batch_error = sum(1 for _, _, state, _, _ in batch_results if state == 'error')
            
            # Collect successful results (non-extinct)
            batch_collected = 0
            for cond_idx, rep_num, state, df, summary in batch_results:
                if state != 'extinct' and state != 'error':
                    successful_results.append((state, df))
                    successful_summaries.append(summary)
                    batch_collected += 1
                    if len(successful_results) >= target_successful:
                        break
            
            # Only count simulations we actually needed/used
            # = successful ones collected + extinct/error ones (which we had to run to find successful ones)
            total_attempts += batch_collected + batch_extinct + batch_error
            total_extinct += batch_extinct
    
    # Trim to exact target number
    successful_results = successful_results[:target_successful]
    successful_summaries = successful_summaries[:target_successful]
    
    stats = {
        'successful': len(successful_results),
        'total_attempts': total_attempts,
        'extinct': total_extinct,
        'extinction_rate': total_extinct / total_attempts if total_attempts > 0 else 0
    }
    
    return successful_results, successful_summaries, stats


def _run_single_job(args):
    """
    Run a single simulation job (one replicate of one condition).
    Must be at module level for pickling in multiprocessing.
    
    Parameters:
    -----------
    args : tuple
        (config_path, condition_idx, replicate_idx, sim_config)
    
    Returns:
    --------
    tuple
        (condition_idx, replicate_idx, state, dataframe, summary_dict)
    """
    config_path, condition_idx, replicate_idx, sim_config = args
    
    try:
        # Run single simulation
        sim = TumourSimulation(config_path, config_dict=sim_config, 
                              condition_index=condition_idx, 
                              replicate_number=replicate_idx)
        state, results = sim.run()
        summary = sim.get_final_summary_row()
        
        return condition_idx, replicate_idx, state, results, summary
    except Exception as e:
        print(f"Error in condition {condition_idx}, replicate {replicate_idx}: {e}")
        import traceback
        traceback.print_exc()
        return condition_idx, replicate_idx, 'error', pd.DataFrame(), {}


def run_simulation_from_config(config_path: str, num_workers: Optional[int] = None) -> List[Tuple[int, pd.DataFrame]]:
    """
    Convenience function to run simulation(s) from a config file.
    Supports both single and multiple initial conditions, with replicates.
    Uses multiprocessing to run replicates in parallel until target number of 
    SUCCESSFUL (non-extinct) simulations is achieved.
    
    Parameters:
    -----------
    config_path : str
        Path to JSON configuration file
    num_workers : int, optional
        Number of parallel workers (default: all CPU cores)
    use_multiprocessing : bool
        If True (default), use multiprocessing. If False, run sequentially for debugging.
        
    Returns:
    --------
    list of tuple of (str, DataFrame)
        List of (final_state, results_dataframe) for each successful replicate
    """
    # Load config to check structure
    with open(config_path, 'r') as f:
        full_config = json.load(f)
    
    # Determine number of workers
    if num_workers is None:
        num_workers = cpu_count()
    
    all_results = []
    all_summaries = []
    all_stats = []
    
    # Determine output directory and whether to save consolidated summary
    output_dir = Path(full_config.get('simulation', {}).get('output_dir', './output')) if 'simulation' in full_config else Path('./output')
    output_prefix = full_config.get('simulation', {}).get('output_prefix', 'simulation') if 'simulation' in full_config else 'simulation'
    save_consolidated = full_config.get('simulation', {}).get('output', {}).get('save_consolidated_summary', True) if 'simulation' in full_config else True
    output_columns = full_config.get('simulation', {}).get('output', {}).get('output_columns', None) if 'simulation' in full_config else None
    
    # Check if config specifies use_multiprocessing at global level 
    use_multiprocessing = full_config.get('use_multiprocessing', True)  # Default to True
    
    # Check if we should account for extinctions when batching
    account_for_extinctions = full_config.get('account_for_extinctions', True)  # Default to True for backward compatibility
    
    # Flatten ALL simulations (conditions + replicates) into a single job list
    # This maximizes parallelization regardless of whether we have single/multi conditions
    all_jobs_specs = []  # List of (condition_idx, target_successful, sim_config)
    
    if 'simulations' in full_config:
        # Multiple initial conditions
        print(f"Found {len(full_config['simulations'])} initial conditions to simulate")
        for condition_idx, sim_config in enumerate(full_config['simulations']):
            target_successful = sim_config['simulation'].get('number_of_replicates', 1)
            all_jobs_specs.append((condition_idx, target_successful, sim_config))
    else:
        # Single initial condition
        target_successful = full_config['simulation'].get('number_of_replicates', 1)
        all_jobs_specs.append((None, target_successful, full_config))
    
    print(f"Multiprocessing: {'ENABLED' if use_multiprocessing else 'DISABLED'}")
    print(f"Workers: {num_workers}")
    print(f"Account for extinctions: {'YES' if account_for_extinctions else 'NO'}")
    print("="*80)
    
    overall_start = datetime.now()
    
    if use_multiprocessing:
        # MULTIPROCESSING PATH: Run all jobs with intelligent batching to handle extinctions
        print(f"Running simulations in parallel with intelligent batching to handle extinctions")
        print(f"Maximizing worker utilization with {num_workers} workers\n")
        
        # Get base seed if available
        base_seed = None
        if 'simulations' in full_config:
            # For multi-condition, seed might be in first condition
            base_seed = full_config['simulations'][0].get('simulation', {}).get('seed', None)
        else:
            base_seed = full_config.get('simulation', {}).get('seed', None)
        
        # Track overall progress
        condition_successful = {idx: 0 for idx, _, _ in all_jobs_specs}
        condition_target = {idx: target for idx, target, _ in all_jobs_specs}
        condition_extinct = {idx: 0 for idx, _, _ in all_jobs_specs}
        condition_total_attempts = {idx: 0 for idx, _, _ in all_jobs_specs}
        
        # Start worker pool once for all conditions
        with Pool(processes=num_workers, initializer=_init_worker, initargs=(base_seed,)) as pool:
            replicate_counter = {idx: 0 for idx, _, _ in all_jobs_specs}
            last_milestone = 0  # Track last milestone for logging every 10 successful simulations
            
            # Keep running batches until all conditions have enough successful results
            while any(condition_successful[idx] < condition_target[idx] for idx, _, _ in all_jobs_specs):
                # Build batch of jobs for conditions that still need more successful results
                batch_jobs = []
                for condition_idx, target, sim_config in all_jobs_specs:
                    # Stop adding jobs if batch is already full
                    if len(batch_jobs) >= num_workers:
                        break
                    
                    # How many more successful results does this condition need?
                    needed = target - condition_successful[condition_idx]
                    if needed <= 0:
                        continue
                    
                    # Estimate extinction rate for this condition
                    attempts = condition_total_attempts[condition_idx]
                    extinct = condition_extinct[condition_idx]
                    
                    if account_for_extinctions:
                        # Add buffer based on observed extinction rate
                        if attempts > 0:
                            extinction_rate = extinct / attempts
                            # Add buffer based on extinction rate (run extra jobs to account for extinctions)
                            buffer_multiplier = 1.5 if extinction_rate > 0.5 else 1.2 if extinction_rate > 0.2 else 1.1
                            jobs_to_run = max(1, int(needed * buffer_multiplier))
                        else:
                            # First batch - run just what we need (or a bit more for extinction buffer)
                            jobs_to_run = max(1, min(needed * 2, needed + 5))
                    else:
                        # Don't account for extinctions - only run exact number needed
                        jobs_to_run = needed
                    
                    # Don't exceed remaining batch capacity
                    remaining_capacity = num_workers - len(batch_jobs)
                    jobs_to_run = min(jobs_to_run, remaining_capacity)
                    
                    # Create jobs for this condition
                    for _ in range(jobs_to_run):
                        replicate_idx = replicate_counter[condition_idx]
                        replicate_counter[condition_idx] += 1
                        batch_jobs.append((config_path, condition_idx, replicate_idx, sim_config))
                
                if not batch_jobs:
                    break
                
                # Run batch in parallel
                print(f"Running batch of {len(batch_jobs)} simulations...")
                batch_results = pool.map(_run_single_job, batch_jobs)
                
                # Process results
                for cond_idx, rep_idx, state, results_df, summary in batch_results:
                    condition_total_attempts[cond_idx] += 1
                    
                    if state not in ['extinct', 'error']:
                        # Only store if we haven't exceeded target for this condition
                        if condition_successful[cond_idx] < condition_target[cond_idx]:
                            all_results.append((state, results_df))
                            all_summaries.append(summary)
                            condition_successful[cond_idx] += 1
                    elif state == 'extinct':
                        condition_extinct[cond_idx] += 1
                
                # Progress update
                total_successful = sum(condition_successful.values())
                total_target = sum(condition_target.values())
                total_attempts = sum(condition_total_attempts.values())
                total_extinct = sum(condition_extinct.values())
                
                # Regular progress update after each batch
                print(f"  Progress: {total_successful}/{total_target} successful ({total_attempts} total attempts, {total_extinct} extinct)")
                        
        # Create stats for each condition
        for condition_idx, target, _ in all_jobs_specs:
            all_stats.append({
                'condition_index': condition_idx,
                'successful': condition_successful[condition_idx],
                'total_attempts': condition_total_attempts[condition_idx],
                'extinct': condition_extinct[condition_idx],
                'extinction_rate': (condition_extinct[condition_idx] / condition_total_attempts[condition_idx] 
                                   if condition_total_attempts[condition_idx] > 0 else 0)
            })
        
        overall_elapsed = (datetime.now() - overall_start).total_seconds()
        print(f"\nAll simulations complete (elapsed: {overall_elapsed:.2f}s)")
        print(f"Total successful: {len(all_results)}, Total extinct: {sum(condition_extinct.values())}, Total attempts: {sum(condition_total_attempts.values())}")
    
    else:
        # SEQUENTIAL PATH: Run all jobs one by one
        print(f"Running in SEQUENTIAL mode (no multiprocessing)\n")
        
        for condition_idx, target_successful, sim_config in all_jobs_specs:
            print(f"\nCondition {condition_idx if condition_idx is not None else 'Single'}: Running until {target_successful} successful simulation(s)")
            print("="*80)
            
            condition_start = datetime.now()
            results, summaries, stats = _run_condition_until_target_sequential(
                config_path, sim_config, condition_idx, target_successful
            )
            all_results.extend(results)
            all_summaries.extend(summaries)
            all_stats.append({
                'condition_index': condition_idx,
                **stats
            })
            
            condition_elapsed = (datetime.now() - condition_start).total_seconds()
            print(f"\nCondition complete (elapsed: {condition_elapsed:.2f}s):")
            print(f"  Successful: {stats['successful']}")
            print(f"  Total attempts: {stats['total_attempts']}")
            print(f"  Extinct: {stats['extinct']}")
            print(f"  Extinction rate: {stats['extinction_rate']:.2%}")
        
        overall_elapsed = (datetime.now() - overall_start).total_seconds()
        print(f"\nAll simulations complete (elapsed: {overall_elapsed:.2f}s)")
    
    # Save consolidated summary if requested
    if save_consolidated and all_summaries:
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_filename = f"{output_prefix}_consolidated_summary_{timestamp}.csv"
        summary_filepath = output_dir / summary_filename
        
        summary_df = pd.DataFrame(all_summaries)
        
        # Filter columns if output_columns is specified
        if output_columns is not None:
            # Keep only specified columns that exist in the dataframe
            available_columns = [col for col in output_columns if col in summary_df.columns]
            if available_columns:
                summary_df = summary_df[available_columns]
            else:
                print(f"  Warning: None of the specified output_columns exist in summary data")
        
        summary_df.to_csv(summary_filepath, index=False)
        print(f"\nConsolidated summary saved to: {summary_filepath}")
    
    # Store stats and summaries for later access
    run_simulation_from_config.last_run_stats = all_stats
    run_simulation_from_config.last_run_summaries = all_summaries
    
    return all_results


if __name__ == "__main__":

    import sys

    if len(sys.argv) < 2:
        print("Usage: python tumour_simulation.py <config_file.json> [num_workers]")
        sys.exit(1)
    
    config_file = sys.argv[1]
    num_workers = int(sys.argv[2]) if len(sys.argv) > 2 else None

    # Record overall timing
    overall_start = datetime.now()
    
    # Run simulations with multiprocessing support
    all_results = run_simulation_from_config(config_file, num_workers=num_workers)
    
    overall_end = datetime.now()
    overall_elapsed = overall_end - overall_start
    
    # Get statistics from the run
    stats_list = getattr(run_simulation_from_config, 'last_run_stats', [])
    
    # Print summary for all successful replicates
    print(f"\n\n{'='*80}")
    print("SUMMARY OF ALL SUCCESSFUL REPLICATES")
    print(f"{'='*80}")
    
    number_relapsed = 0
    number_ongoing = 0
    
    for idx, (state, results) in enumerate(all_results, start=1):
        if state == 'relapsed':
            number_relapsed += 1
        elif state == 'ongoing':
            number_ongoing += 1
        
        # Only print details for first few and last few to avoid clutter
        if idx <= 5 or idx > len(all_results) - 5:
            print(f"\nSuccessful Replicate {idx}:")
            print(f"  Final state: {state}")
            if len(results) > 0:
                print(f"  Final clones: {len(results)}")
                print(f"  Final tumor size: {results['N'].sum():.2e}")
        elif idx == 6:
            print(f"\n... ({len(all_results) - 10} more replicates) ...")
    
    total_successful = len(all_results)
    total_attempts = sum(s['total_attempts'] for s in stats_list)
    total_extinct = sum(s['extinct'] for s in stats_list)
    
    print(f"\n{'='*80}")
    print("OVERALL STATISTICS")
    print(f"{'='*80}")
    print(f"Successful replicates (returned): {total_successful}")
    print(f"  - Relapsed: {number_relapsed}")
    print(f"  - Ongoing (timeout): {number_ongoing}")
    print(f"\nTotal attempts made: {total_attempts}")
    print(f"Extinct tumors (discarded): {total_extinct}")
    if total_attempts > 0:
        print(f"Extinction rate: {total_extinct / total_attempts:.4f} ({total_extinct / total_attempts * 100:.2f}%)")
    print(f"\nTotal elapsed time: {overall_elapsed}")
    print(f"{'='*80}")
    
    # Print per-condition statistics if multiple conditions
    if len(stats_list) > 1:
        print(f"\n{'='*80}")
        print("PER-CONDITION STATISTICS")
        print(f"{'='*80}")
        for stat in stats_list:
            cond_idx = stat['condition_index']
            print(f"\nCondition {cond_idx + 1 if cond_idx is not None else 'N/A'}:")
            print(f"  Successful: {stat['successful']}")
            print(f"  Total attempts: {stat['total_attempts']}")
            print(f"  Extinct: {stat['extinct']}")
            print(f"  Extinction rate: {stat['extinction_rate']:.2%}")
        print(f"{'='*80}")
    