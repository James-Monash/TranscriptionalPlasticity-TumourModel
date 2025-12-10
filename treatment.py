"""
Treatment class: Manages drug treatment schedules and calculates branching probabilities.

Handles:
- Treatment scheduling (MTD, ADT, custom schedules)
- Drug concentration over time
- Probability calculations for each cell type based on treatment state
- Multiple drug types (absolute, proportional, plasticity therapies)
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from enum import Enum


class DrugType(Enum):
    """Enumeration of drug types."""
    ABSOLUTE = "abs"
    PROPORTIONAL = "prop"


class ScheduleType(Enum):
    """Enumeration of treatment schedule types."""
    CONTINUOUS = "continuous"
    MTD = "mtd"  # Maximum Tolerated Dose
    ADAPTIVE = "adaptive"  # Adaptive Therapy
    FIXED_DOSES = "fixed_doses"
    CUSTOM = "custom"
    THRESHOLD = "threshold_triggered"
    INTERMITTENT = "intermittent"
    OFF = "off"  # No treatment


class Treatment:
    """
    Manages treatment protocol and calculates cell fate probabilities.
    
    Responsibilities:
    - Track treatment schedule and current state
    - Calculate drug concentrations over time
    - Compute branching probabilities for S, Q, R cells under treatment
    - Handle primary and secondary (plasticity) therapies
    """
    
    def __init__(
        self,
        config: Dict,
        base_params: Dict
    ):
        """
        Initialize treatment from configuration.
        
        Parameters:
        -----------
        config : dict
            Treatment configuration from JSON
        base_params : dict
            Base biological parameters (s, m, q, r, l, idle)
        """
        # Treatment schedule parameters
        self.schedule_type = ScheduleType(config.get('schedule_type', 'off'))
        self.drug_type = DrugType(config.get('drug_type', 'abs'))
        
        # Primary therapy parameters
        self.treat_amt = config.get('treat_amt', 0.8)
        self.pen_amt = config.get('pen_amt', 4.0)
        self.dose_duration = config.get('dose_duration', 24)
        
        # Secondary therapy (plasticity) parameters
        self.secondary_therapy_enabled = config.get('secondary_therapy', False)
        self.secondary_therapy_type = config.get('secondary_therapy_type', 'plast')
        
        # Adaptive therapy thresholds
        self.treatment_start_size = config.get('treatment_start_size', 1e9)
        self.treatment_stop_size = config.get('treatment_stop_size', 1e9)
        self.relapse_size = config.get('relapse_size', 4e9)
        
        # Custom schedule
        self.custom_schedule = config.get('custom_schedule', [])
        
        # Penalty (fitness cost without treatment)
        self.penalty_enabled = config.get('penalty', False)
        
        # Base parameters 
        self.base_s = base_params['s'] # selective advantage 
        self.base_m = base_params['m'] # mutation rate
        self.base_q = base_params['q'] # rate of transient resistance gain 
        self.base_r = base_params['r'] # rate of permanent resistance gain
        self.base_l = base_params['l'] # rate of transient resistance loss
        self.base_idle = base_params['idle'] # baseline quiescence rate
        
        # State variables
        self.current_generation = 0
        self.drug_iterations = 0
        self.drug_iterations2 = 0
        self.doses_given = 0
        self.treatment_active = False
        self.secondary_active = False
        self.treatment_started = False
        self.relapsed = False
        
        # Precompute probability table
        self._compute_probability_table()
    
    def _compute_probability_table(self):
        """
        Compute base probabilities for different generations (k values).
        
        This creates a lookup table of probabilities based on the number
        of driver mutations (k) a clone has acquired.
        """
        k_list = list(range(1, 50)) # The first cell is considered to have 1 driver mutation
        
        # Use dict instead of DataFrame for O(1) lookups
        self.prob_table = {}
        for k in k_list:
            # Death probability decreases with mutations (selective advantage)
            prob_death = ((1 - self.base_idle) / 2) * (1 - self.base_s) ** k
            
            # Mutation and transition probabilities (scaled by survival)
            prob_mutation = self.base_m * (1 - prob_death) # probability of developing a new driver mutation
            prob_to_quasi = self.base_q * (1 - prob_death) # probability of transient resistance gain
            prob_to_resistant = self.base_r * (1 - prob_death) # probability of permanent resistance gain
            prob_to_sensitive = self.base_l * (1 - prob_death) # probability of transient resistance loss
            
            # Birth probability is remainder
            prob_birth = 1 - self.base_idle - prob_death - prob_mutation - prob_to_quasi - prob_to_resistant - prob_to_sensitive
            
            self.prob_table[k] = {
                'prob_idle': self.base_idle,
                'prob_birth': prob_birth,
                'prob_death': prob_death,
                'prob_mutation': prob_mutation,
                'prob_to_quasi': prob_to_quasi,
                'prob_to_resistant': prob_to_resistant,
                'prob_to_sensitive': prob_to_sensitive
            }

    def _extend_probability_table(self, new_k: int):
        """
        Extend the probability table up to new_k (inclusive).
        """
        current_max_k = max(self.prob_table.keys())
        for k in range(int(current_max_k) + 1, new_k + 1):
            prob_death = ((1 - self.base_idle) / 2) * (1 - self.base_s) ** k
            prob_mutation = self.base_m * (1 - prob_death)
            prob_to_quasi = self.base_q * (1 - prob_death)
            prob_to_resistant = self.base_r * (1 - prob_death)
            prob_to_sensitive = self.base_l * (1 - prob_death)
            prob_birth = 1 - self.base_idle - prob_death - prob_mutation - prob_to_quasi - prob_to_resistant - prob_to_sensitive
            self.prob_table[k] = {
                'prob_idle': self.base_idle,
                'prob_birth': prob_birth,
                'prob_death': prob_death,
                'prob_mutation': prob_mutation,
                'prob_to_quasi': prob_to_quasi,
                'prob_to_resistant': prob_to_resistant,
                'prob_to_sensitive': prob_to_sensitive
            }

    def get_base_probabilities(self, k: int) -> Dict[str, float]:
        """
        Get base probabilities for a clone with k driver mutations.
        Extends the probability table if k is greater than current table size.
        
        Parameters:
        -----------
        k : int
            Number of driver mutations
            
        Returns:
        --------
        dict
            Dictionary of base probabilities
        """
        if k not in self.prob_table:
            self._extend_probability_table(k)

        # Direct dict lookup - O(1) instead of O(n) with DataFrame
        # Return copy since caller may modify it
        probs = self.prob_table[k]
        return {
            'prob_idle': probs['prob_idle'],
            'prob_birth': probs['prob_birth'],
            'prob_death': probs['prob_death'],
            'prob_mutation': probs['prob_mutation'],
            'prob_to_quasi': probs['prob_to_quasi'],
            'prob_to_resistant': probs['prob_to_resistant'],
            'prob_to_sensitive': probs['prob_to_sensitive']
        }
    
    def update_treatment_state(self, tumor_size: int, generation: int):
        """
        Update treatment state based on tumor size and schedule.
        
        Parameters:
        -----------
        tumor_size : int
            Current total tumor cell count
        generation : int
            Current generation number
        """
        self.current_generation = generation
        
        if self.schedule_type == ScheduleType.OFF:
            self.treatment_active = False
            self.secondary_active = False
            return
        
        if self.schedule_type == ScheduleType.CONTINUOUS:
            self.treatment_active = True
            self.secondary_active = self.secondary_therapy_enabled
            self.drug_iterations += 1
            self.drug_iterations2 += 1
            
        elif self.schedule_type == ScheduleType.MTD:
            # Start treatment when tumor reaches threshold
            if not self.treatment_started and tumor_size > self.treatment_start_size:
                self.treatment_started = True
                self.treatment_active = True
                self.secondary_active = self.secondary_therapy_enabled
                self.drug_iterations = 0
                self.drug_iterations2 = 0
            
            if self.treatment_started:
                self.drug_iterations += 1
                self.drug_iterations2 += 1
                
                # Cycle doses
                if self.drug_iterations >= self.dose_duration:
                    self.drug_iterations = 0
                    self.doses_given += 1
                
                if self.drug_iterations2 >= self.dose_duration:
                    self.drug_iterations2 = 0
                
                # Check for relapse
                if tumor_size > self.relapse_size:
                    self.relapsed = True
        
        elif self.schedule_type == ScheduleType.ADAPTIVE:
            # Adaptive therapy: on/off based on tumor size
            if not self.treatment_started and tumor_size > self.treatment_start_size:
                self.treatment_started = True
                self.treatment_active = True
                self.secondary_active = self.secondary_therapy_enabled
                self.drug_iterations = 0
                self.drug_iterations2 = 0
            
            if self.treatment_started:
                if self.treatment_active:
                    self.drug_iterations += 1
                    self.drug_iterations2 += 1
                    
                    # Complete dose cycle
                    if self.drug_iterations >= self.dose_duration:
                        self.drug_iterations = 0
                        self.doses_given += 1
                        
                        # Turn off if tumor shrunk below threshold
                        if tumor_size < self.treatment_stop_size:
                            self.treatment_active = False
                            self.secondary_active = False
                        elif tumor_size > self.relapse_size:
                            self.relapsed = True
                    
                    if self.drug_iterations2 >= self.dose_duration:
                        self.drug_iterations2 = 0
                else:
                    # Treatment off, check if we need to resume
                    if tumor_size > self.treatment_start_size:
                        self.treatment_active = True
                        self.secondary_active = self.secondary_therapy_enabled
                        self.drug_iterations = 0
                        self.drug_iterations2 = 0
                        self.doses_given += 1
    
    def get_drug_concentration(self) -> Tuple[float, float]:
        """
        Calculate current drug concentrations.
        
        Returns:
        --------
        tuple of (float, float)
            (primary_concentration, secondary_concentration)
            Values range from 0.0 to 1.0
        """
        if not self.treatment_active:
            return (0.0, 0.0)
        
        # Linear decay over dose duration
        conc1 = max(0.0, 1.0 - self.drug_iterations / self.dose_duration)
        conc2 = max(0.0, 1.0 - self.drug_iterations2 / self.dose_duration)
        
        if not self.secondary_active:
            conc2 = 0.0
        
        return (conc1, conc2)
    
    def calculate_probabilities(
        self,
        k: int,
        cell_type: str
    ) -> Dict[str, float]:
        """
        Calculate branching probabilities for a cell type under current treatment.
        
        The set of available actions for each cell at a fixed time step is dependent 
        on the current state of the cell:
        - If a cell is currently sensitive, it cannot take the quasi resistant to sensitive option
        - If a cell is currently quasi resistant, it cannot take the sensitive to quasi resistant option
        - If a cell currently has permanent resistance, it cannot take the sensitive to quasi 
          resistant or the quasi resistant to sensitive option
        
        Parameters:
        -----------
        k : int
            Number of driver mutations in clone
        cell_type : str
            Cell type: "S" (sensitive), "Q" (quasi), or "R" (resistant)
            
        Returns:
        --------
        dict
            Dictionary with probabilities: prob_idle, prob_birth, prob_death, prob_mutation,
            prob_to_quasi, prob_to_resistant, prob_to_sensitive
        """
        # Get base probabilities
        probs = self.get_base_probabilities(k)
        
        # Zero out impossible transitions based on current cell type
        if cell_type == "S":
            # Sensitive cells can't lose quasi-resistance (they don't have it)
            probs['prob_to_sensitive'] = 0.0
        elif cell_type == "Q":
            # Quasi cells can't gain quasi-resistance (they already have it)
            probs['prob_to_quasi'] = 0.0
        elif cell_type == "R":
            # Resistant cells can't gain or lose quasi-resistance or gain more resistance (permanent resistance)
            probs['prob_to_quasi'] = 0.0
            probs['prob_to_sensitive'] = 0.0
            probs['prob_to_resistant'] = 0.0
        
        # Rebalance birth probability after zeroing impossible transitions
        # Birth probability is the remainder after all other events
        probs['prob_birth'] = 1.0 - probs['prob_idle'] - probs['prob_death'] - probs['prob_mutation'] - probs['prob_to_quasi'] - probs['prob_to_resistant'] - probs['prob_to_sensitive']
        
        # Get drug concentrations
        conc1, conc2 = self.get_drug_concentration()

        if self.treatment_active:
        
            # Apply treatment effects based on cell type
            if cell_type == "S":
                probs = self._apply_treatment_to_sensitive(probs, conc1, conc2)
            elif cell_type == "Q":
                probs = self._apply_treatment_to_quasi(probs, conc1, conc2)
            elif cell_type == "R":
                probs = self._apply_treatment_to_resistant(probs, conc1, conc2)
        
        # Apply penalty if enabled and no treatment
        if self.penalty_enabled and not self.treatment_active:
            if cell_type in ["Q", "R"]:
                probs = self._apply_penalty(probs, cell_type, conc1)
   
        # Only normalize if treatment was applied (base probs already normalized)
        if self.treatment_active or (self.penalty_enabled and not self.treatment_active and cell_type in ["Q", "R"]):
            total = sum(probs.values())
            if total > 0:
                for key in probs:
                    probs[key] /= total
        
        return probs
    
    def _apply_treatment_to_sensitive(
        self,
        probs: Dict[str, float],
        conc1: float,
        conc2: float
    ) -> Dict[str, float]:
        """Apply primary and secondary therapy effects to sensitive cells."""
        
        # Primary therapy effect
        if self.drug_type == DrugType.ABSOLUTE:
            # Absolute drug: death rate moves toward treat_amt
            probs['prob_death'] = probs['prob_death'] + (self.treat_amt - probs['prob_death']) * conc1
            probs['prob_birth'] = 1 - probs['prob_death'] - probs['prob_mutation'] - probs['prob_to_quasi'] - probs['prob_to_resistant'] - probs['prob_idle'] - probs['prob_to_sensitive']
        
        elif self.drug_type == DrugType.PROPORTIONAL:
            # Proportional drug: death increases based on growth rate
            diff = min(((probs['prob_birth'] - probs['prob_death']) * 3 * conc1), 0.9 * probs['prob_birth'])
            probs['prob_birth'] = probs['prob_birth'] - diff
            probs['prob_death'] = probs['prob_death'] + diff
        
        # Secondary therapy (plasticity modulation)
        if self.secondary_active and conc2 > 0:
            # Increase quiescence, decrease resistance acquisition
            probs['prob_idle'] = probs['prob_idle'] + probs['prob_to_quasi'] * conc2
            probs['prob_to_quasi'] = probs['prob_to_quasi'] - probs['prob_to_quasi'] * conc2
        
        return probs
    
    def _apply_treatment_to_quasi(
        self,
        probs: Dict[str, float],
        conc1: float,
        conc2: float
    ) -> Dict[str, float]:
        """Apply primary and secondary therapy effects to quasi-resistant cells."""
        
        # Primary therapy: partial protection via penalty amount
        diff = (probs['prob_birth'] - probs['prob_death']) * conc1
        probs['prob_birth'] = probs['prob_birth'] + diff / self.pen_amt
        probs['prob_death'] = probs['prob_death'] - diff / self.pen_amt
        
        # Secondary therapy: promote reversion to sensitive
        if self.secondary_active and conc2 > 0:
            if self.secondary_therapy_type == "plast":
                # Type A: force out of quiescence, increase reversion to sensitive
                # Note: prob_to_quasi is already 0 for Q cells, so we work with prob_to_sensitive instead
                prob_to_sensitive_boost = 2 * self.base_q * conc2
                probs['prob_idle'] = max(0, probs['prob_idle'] - prob_to_sensitive_boost)
                probs['prob_to_sensitive'] = probs['prob_to_sensitive'] + prob_to_sensitive_boost
            else:
                # Type B: induce dormancy and reversion
                red = min(probs['prob_birth'] - 0.01, probs['prob_death'] - 0.01, 0.1)
                probs['prob_birth'] = probs['prob_birth'] - red * conc2
                probs['prob_death'] = probs['prob_death'] - red * conc2
                probs['prob_to_sensitive'] = probs['prob_to_sensitive'] + 2 * red * conc2
        
        return probs
    
    def _apply_treatment_to_resistant(
        self,
        probs: Dict[str, float],
        conc1: float,
        conc2: float
    ) -> Dict[str, float]:
        """Apply primary therapy effects to resistant cells."""
        
        # Primary therapy: partial protection via penalty amount
        diff = (probs['prob_birth'] - probs['prob_death']) * conc1
        probs['prob_birth'] = probs['prob_birth'] + diff / self.pen_amt
        probs['prob_death'] = probs['prob_death'] - diff / self.pen_amt
        
        # Secondary therapy has no effect on resistant cells
        
        return probs
    
    def _apply_penalty(
        self,
        probs: Dict[str, float],
        cell_type: str,
        conc1: float
    ) -> Dict[str, float]:
        """Apply fitness penalty to resistant/quasi cells without treatment."""
        diff = (probs['prob_birth'] - probs['prob_death']) * conc1
        probs['prob_birth'] = probs['prob_birth'] - diff / (self.pen_amt + 1)
        probs['prob_death'] = probs['prob_death'] + diff / (self.pen_amt + 1)
        return probs
    
    def get_state_summary(self) -> Dict:
        """
        Get summary of current treatment state.
        
        Returns:
        --------
        dict
            Treatment state information
        """
        conc1, conc2 = self.get_drug_concentration()
        return {
            'generation': self.current_generation,
            'treatment_active': self.treatment_active,
            'secondary_active': self.secondary_active,
            'doses_given': self.doses_given,
            'drug_iterations': self.drug_iterations,
            'concentration_primary': conc1,
            'concentration_secondary': conc2,
            'relapsed': self.relapsed
        }
