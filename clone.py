"""
Clone class: Stores information on cell populations with identical driver mutations.

Tracks:
- Number of sensitive (S), transient-resistant (Q), and resistant (R) cells
- Clone lineage information (parent, generation)
- Cell type and mutation history
"""

import numpy as np
from typing import Dict, Optional, List


class Clone:
    """
    Represents a clonal population within a tumor.
    
    Each clone has a specific set of driver mutations and contains cells that can be:
    - Sensitive (S): Drug-sensitive
    - Transient-resistant (Q): Phenotypically plastic, partially resistant
    - Resistant (R): Fully resistant to primary therapy
    """
    
    def __init__(
        self,
        clone_id: int,
        parent_id: int,
        parent_type: str,
        generation: int,
        n_sensitive: int = 0,
        n_transient: int = 0,
        n_resistant: int = 0,
        n_driver_mutations: int = 1
    ):
        """
        Initialize a clone.
        
        Parameters:
        -----------
        clone_id : int
            Unique identifier for this clone
        parent_id : int
            ID of parent clone (-1 for initial clone)
        parent_type : str
            Cell type of parent ("S", "Q", "R", or "Sens" for initial)
        generation : int
            Generation (timestep) when clone emerged
        n_sensitive : int
            Initial number of sensitive cells
        n_transient : int
            Initial number of transient-resistant cells
        n_resistant : int
            Initial number of resistant cells
        """
        self.clone_id = clone_id
        self.parent_id = parent_id
        self.parent_type = parent_type
        self.generation = generation
        self.n_driver_mutations = n_driver_mutations
        # Cell counts by type
        self.n_sensitive = n_sensitive
        self.n_transient = n_transient
        self.n_resistant = n_resistant
        # Tracking history 
        self.history = {
            'n_sensitive': [],
            'n_transient': [],
            'n_resistant': [],
            'total': [],
            'generations': []
        }
        
    @property
    def total_cells(self) -> int:
        """Total number of cells in this clone."""
        return self.n_sensitive + self.n_transient + self.n_resistant
    
    def update_counts(
        self,
        delta_sensitive: int = 0,
        delta_transient: int = 0,
        delta_resistant: int = 0
    ):
        """
        Update cell counts based on births, deaths, and transitions.
        
        Parameters:
        -----------
        delta_sensitive : int
            Net change in sensitive cells (births - deaths + transitions)
        delta_transient : int
            Net change in transient-resistant cells
        delta_resistant : int
            Net change in resistant cells
        """
        self.n_sensitive += delta_sensitive
        self.n_transient += delta_transient
        self.n_resistant += delta_resistant
            
    def record_history(self, generation: int):
        """
        Record current state to history.
        
        Parameters:
        -----------
        generation : int
            Current generation number
        """
        self.history['n_sensitive'].append(self.n_sensitive)
        self.history['n_transient'].append(self.n_transient)
        self.history['n_resistant'].append(self.n_resistant)
        self.history['total'].append(self.total_cells)
        self.history['generations'].append(generation)
    
    def is_extinct(self) -> bool:
        """Check if clone has gone extinct."""
        return self.total_cells == 0
    
    def to_dict(self) -> Dict:
        """
        Convert clone to dictionary format for output.
        
        Returns:
        --------
        dict
            Clone information as dictionary
        """
        return {
            'Id': self.clone_id,
            'Parent': self.parent_id,
            'ParentType': self.parent_type,
            'K': self.generation,
            'N': self.total_cells,
            'Ns': self.n_sensitive,
            'Nq': self.n_transient,
            'Nr': self.n_resistant
        }
    
    def __repr__(self) -> str:
        return (f"Clone(id={self.clone_id}, generation={self.generation}, "
                f"S={self.n_sensitive}, Q={self.n_transient}, R={self.n_resistant}, "
                f"total={self.total_cells})")


class CloneCollection:
    """
    Manages a collection of clones in the tumor.
    """
    
    def __init__(self):
        """Initialize empty clone collection."""
        self.clones: Dict[int, Clone] = {}
        self.next_id = 0
        # Track unique driver mutation counts (once added, never removed)
        self.unique_driver_mutations: set = set()
    
    def add_clone(
        self,
        parent_id: int,
        parent_type: str,
        generation: int,
        n_sensitive: int = 0,
        n_transient: int = 0,
        n_resistant: int = 0,
        n_driver_mutations: int = 1
    ) -> Clone:
        """
        Add a new clone to the collection.
        
        Parameters:
        -----------
        parent_id : int
            ID of parent clone
        parent_type : str
            Cell type of parent
        generation : int
            Generation when clone emerged
        n_sensitive : int
            Initial sensitive cells
        n_transient : int
            Initial transient-resistant cells
        n_resistant : int
            Initial resistant cells
        n_driver_mutations : int
            Number of driver mutations in this clone
            
        Returns:
        --------
        Clone
            The newly created clone
        """
        new_clone = Clone(
            clone_id=self.next_id,
            parent_id=parent_id,
            parent_type=parent_type,
            generation=generation,
            n_sensitive=n_sensitive,
            n_transient=n_transient,
            n_resistant=n_resistant,
            n_driver_mutations=n_driver_mutations
        )
        self.clones[self.next_id] = new_clone
        self.unique_driver_mutations.add(n_driver_mutations)
        self.next_id += 1
        return new_clone
    
    def get_clone(self, clone_id: int) -> Optional[Clone]:
        """
        Get clone by ID.
        
        Parameters:
        -----------
        clone_id : int
            Clone identifier
            
        Returns:
        --------
        Clone or None
            The clone if found, None otherwise
        """
        return self.clones.get(clone_id)
    
    def remove_extinct_clones(self):
        """Remove all extinct clones from collection."""
        extinct_ids = [
            clone_id for clone_id, clone in self.clones.items()
            if clone.is_extinct()
        ]
        for clone_id in extinct_ids:
            del self.clones[clone_id]
    
    def get_unique_driver_mutations(self) -> set:
        """Get set of unique driver mutation counts across all clones."""
        return self.unique_driver_mutations
    
    def get_total_cells(self) -> int:
        """Get total number of cells across all clones."""
        return sum(clone.total_cells for clone in self.clones.values())
    
    def get_total_resistant(self) -> int:
        """Get total number of resistant cells across all clones."""
        return sum(clone.n_resistant for clone in self.clones.values())
    
    def get_total_by_type(self) -> Dict[str, int]:
        """
        Get total cells by type across all clones.
        
        Returns:
        --------
        dict
            Dictionary with 'sensitive', 'transient', 'resistant', 'total' counts
        """
        return {
            'sensitive': sum(clone.n_sensitive for clone in self.clones.values()),
            'transient': sum(clone.n_transient for clone in self.clones.values()),
            'resistant': sum(clone.n_resistant for clone in self.clones.values()),
            'total': self.get_total_cells()
        }
    
    def to_dataframe_rows(self) -> List[Dict]:
        """
        Convert all clones to list of dictionaries for DataFrame creation.
        
        Returns:
        --------
        list of dict
            List of clone dictionaries
        """
        return [clone.to_dict() for clone in self.clones.values()]
    
    def __len__(self) -> int:
        return len(self.clones)
    
    def __iter__(self):
        return iter(self.clones.values())
    
    def __repr__(self) -> str:
        total = self.get_total_cells()
        return f"CloneCollection(n_clones={len(self)}, total_cells={total})"
