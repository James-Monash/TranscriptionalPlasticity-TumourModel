"""
Treatment class: Manages drug treatment schedules and calculates branching probabilities.

Handles:
- Treatment scheduling (MTD, ADT, custom schedules)
- Drug concentration over time
- Probability calculations for each cell type based on treatment state
- Multiple drug types (absolute, proportional, plasticity therapies)
"""

import numpy as np
import pandas
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
            Base biological parameters (s, u, pq, pr, ps, idle)
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
        self.base_s = base_params['s']
        self.base_u = base_params['u']
        self.base_pq = base_params['pq']
        self.base_pr = base_params['pr']
        self.base_ps = base_params['ps']
        self.base_idle = base_params['idle']
        
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
        k_list = list(range(0, 50))
        
        rows = []
        for k in k_list:
            # Death probability increases with mutations
            prob_death = ((1 - self.base_idle) / 2) * (1 - self.base_s) ** k
            prob_no_death = 1 - self.base_idle - prob_death
            
            # Birth and transition probabilities
            prob_birth_pure = prob_no_death * (1 - self.base_u)
            prob_mutation = prob_no_death * self.base_u
            prob_to_quasi = prob_no_death * self.base_pq
            prob_to_resistant = prob_no_death * self.base_pr
            prob_to_sensitive = prob_no_death * self.base_ps
            
            rows.append({
                'k': k,
                'prob_idle': self.base_idle,
                'prob_birth': prob_birth_pure,
                'prob_death': prob_death,
                'prob_mutation': prob_mutation,
                'prob_to_quasi': prob_to_quasi,
                'prob_to_resistant': prob_to_resistant,
                'prob_to_sensitive': prob_to_sensitive
            })
        
        self.prob_table = pandas.DataFrame(rows)

    def get_base_probabilities(self, k: int) -> Dict[str, float]:
        """
        Get base probabilities for a clone with k driver mutations.
        
        Parameters:
        -----------
        k : int
            Number of driver mutations
            
        Returns:
        --------
        dict
            Dictionary of base probabilities
        """
        if k >= len(self.prob_table):
            k = len(self.prob_table) - 1
        
        row = self.prob_table.loc[k]
        return {
            'prob_idle': row['prob_idle'],
            'prob_birth': row['prob_birth'],
            'prob_death': row['prob_death'],
            'prob_mutation': row['prob_mutation'],
            'prob_to_quasi': row['prob_to_quasi'],
            'prob_to_resistant': row['prob_to_resistant'],
            'prob_to_sensitive': row['prob_to_sensitive']
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
        
        Parameters:
        -----------
        k : int
            Number of driver mutations in clone
        cell_type : str
            Cell type: "S" (sensitive), "Q" (quasi), or "R" (resistant)
            
        Returns:
        --------
        dict
            Dictionary with probabilities: pnd, pb, pdd, pm, pq, pr, ps
        """
        # Get base probabilities
        probs = self.get_base_probabilities(k)
        
        # Get drug concentrations
        conc1, conc2 = self.get_drug_concentration()
        
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
        
        return probs
    
    def _apply_treatment_to_sensitive(
        self,
        probs: Dict[str, float],
        conc1: float,
        conc2: float
    ) -> Dict[str, float]:
        """Apply primary and secondary therapy effects to sensitive cells."""
        if not self.treatment_active:
            return probs
        
        # Primary therapy effect
        if self.drug_type == DrugType.ABSOLUTE:
            # Absolute drug: death rate moves toward treat_amt
            probs['prob_death'] = probs['prob_death'] + (self.treat_amt - probs['prob_death']) * conc1
            probs['prob_birth'] = 1 - probs['prob_death'] - probs['prob_mutation'] - probs['prob_to_quasi'] - probs['prob_to_resistant'] - probs['prob_idle']
        
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
        if not self.treatment_active:
            return probs
        
        # Primary therapy: partial protection via penalty amount
        diff = (probs['prob_birth'] - probs['prob_death']) * conc1
        probs['prob_birth'] = probs['prob_birth'] + diff / self.pen_amt
        probs['prob_death'] = probs['prob_death'] - diff / self.pen_amt
        
        # Secondary therapy: promote reversion to sensitive
        if self.secondary_active and conc2 > 0:
            if self.secondary_therapy_type == "plast":
                # Type A: force out of quiescence, increase reversion
                probs['prob_idle'] = probs['prob_idle'] - 2 * probs['prob_to_quasi'] * conc2
                probs['prob_to_quasi'] = probs['prob_to_quasi'] + 2 * probs['prob_to_quasi'] * conc2
            else:
                # Type B: induce dormancy and reversion
                red = min(probs['prob_birth'] - 0.01, probs['prob_death'] - 0.01, 0.1)
                probs['prob_birth'] = probs['prob_birth'] - red * conc2
                probs['prob_death'] = probs['prob_death'] - red * conc2
                probs['prob_to_quasi'] = probs['prob_to_quasi'] + 2 * red * conc2
        
        return probs
    
    def _apply_treatment_to_resistant(
        self,
        probs: Dict[str, float],
        conc1: float,
        conc2: float
    ) -> Dict[str, float]:
        """Apply primary therapy effects to resistant cells."""
        if not self.treatment_active:
            return probs
        
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
