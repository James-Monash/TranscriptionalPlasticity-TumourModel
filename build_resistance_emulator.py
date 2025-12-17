"""
Resistance Distribution Emulator Builder

This module builds an emulator that predicts the distribution of resistance proportions
for different biological parameter combinations. Uses:
- Latin Hypercube Sampling for initial parameter space exploration
- Burr XII distribution fitting for each parameter combination
- Adaptive refinement to sample where uncertainty is highest
- Interpolation using Gaussian Process for unsampled points

The emulator maps 6D parameter space to Burr XII distribution parameters.
"""

import sys
import json
import numpy as np
import pandas as pd
import pickle
import time
from pathlib import Path
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count
from scipy.stats import burr12, qmc
from scipy.spatial.distance import cdist
from scipy.interpolate import LinearNDInterpolator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
import warnings

from tumour_simulation import run_simulation_from_config
from analyze_resistance_distribution import calculate_resistance_proportions

warnings.filterwarnings('ignore')


class ParameterSpace:
    """
    Manages the 6D parameter space transformation and sampling.
    
    Parameters are transformed to log scale (except idle) and normalized to [0,1].
    """
    
    def __init__(self):
        """Initialize parameter bounds."""
        # Original parameter ranges
        self.param_bounds = {
            's': (1e-3, 1e-1),      # Selective advantage
            'm': (1e-8, 1e-5),      # Mutation rate
            'q': (1e-11, 1e-2),     # Transient transition rate
            'r': (1e-11, 1e-2),     # Resistant transition rate
            'l': (1e-11, 1e-2),     # Reversion rate
            'idle': (0.0, 0.5)      # Idle probability
        }
        
        # Parameter names in order
        self.param_names = ['s', 'm', 'q', 'r', 'l', 'idle']
        
        # Which parameters use log scale
        self.log_params = ['s', 'm', 'q', 'r', 'l']
        
    def to_unit_hypercube(self, params_dict):
        """
        Transform biological parameters to [0,1] hypercube.
        
        Parameters:
        -----------
        params_dict : dict
            Dictionary with keys: s, m, q, r, l, idle
            
        Returns:
        --------
        numpy.array
            6D point in unit hypercube
        """
        normalized = np.zeros(6)
        
        for i, param in enumerate(self.param_names):
            value = params_dict[param]
            lower, upper = self.param_bounds[param]
            
            if param in self.log_params:
                # Log scale transformation
                log_value = np.log10(value)
                log_lower = np.log10(lower)
                log_upper = np.log10(upper)
                normalized[i] = (log_value - log_lower) / (log_upper - log_lower)
            else:
                # Linear scale
                normalized[i] = (value - lower) / (upper - lower)
        
        return normalized
    
    def from_unit_hypercube(self, normalized_point):
        """
        Transform point from [0,1] hypercube to biological parameters.
        
        Parameters:
        -----------
        normalized_point : array-like
            6D point in unit hypercube
            
        Returns:
        --------
        dict
            Dictionary with biological parameters
        """
        params_dict = {}
        
        for i, param in enumerate(self.param_names):
            norm_value = normalized_point[i]
            lower, upper = self.param_bounds[param]
            
            if param in self.log_params:
                # Inverse log scale transformation
                log_lower = np.log10(lower)
                log_upper = np.log10(upper)
                log_value = norm_value * (log_upper - log_lower) + log_lower
                params_dict[param] = 10 ** log_value
            else:
                # Linear scale
                params_dict[param] = norm_value * (upper - lower) + lower
        
        return params_dict
    
    def latin_hypercube_sample(self, n_samples, seed=None):
        """
        Generate Latin Hypercube samples in unit hypercube.
        
        Parameters:
        -----------
        n_samples : int
            Number of samples to generate
        seed : int, optional
            Random seed
            
        Returns:
        --------
        numpy.array
            Array of shape (n_samples, 6)
        """
        sampler = qmc.LatinHypercube(d=6, seed=seed)
        samples = sampler.random(n=n_samples)
        return samples


class TimingEstimator:
    """
    Estimates simulation runtime based on s value.
    Adapts estimates based on actual observed batch times.
    """
    
    def __init__(self):
        """Initialize with empirical timing data."""
        # Empirical data: (s_value, seconds_per_replicate)
        # NOTE: These are for INDIVIDUAL simulations, not batched
        self.timing_data = np.array([
            [0.1, 1.812258],
            [0.01, 7.294177],
            [0.001, 63.127795]
        ])
        
        # Track actual batch performance for better estimation
        self.batch_history = []  # List of (batch_size, avg_s, actual_time_per_point)
        self.batch_overhead_factor = 1.5  # Initial overhead estimate for batched execution
        
    def estimate_time(self, s_value, n_replicates=1):
        """
        Estimate time for simulations with given s value.
        
        Parameters:
        -----------
        s_value : float
            Selective advantage
        n_replicates : int
            Number of replicates
            
        Returns:
        --------
        float
            Estimated time in seconds
        """
        # Log-log interpolation (time scales roughly as 1/s)
        log_s = np.log10(s_value)
        log_s_data = np.log10(self.timing_data[:, 0])
        log_time_data = np.log10(self.timing_data[:, 1])
        
        # Linear interpolation in log-log space
        log_time = np.interp(log_s, log_s_data[::-1], log_time_data[::-1])
        time_per_replicate = 10 ** log_time
        
        # Apply batch overhead factor
        return time_per_replicate * n_replicates * self.batch_overhead_factor
    
    def estimate_batch_time(self, s_values, n_replicates):
        """
        Estimate total time for a batch of simulations.
        In batched execution, time is dominated by slowest conditions.
        
        Parameters:
        -----------
        s_values : list of float
            Selective advantage values for each condition
        n_replicates : int
            Number of replicates per condition
            
        Returns:
        --------
        float
            Estimated total batch time in seconds
        """
        # Estimate time for each condition
        individual_times = [self.estimate_time(s, n_replicates) for s in s_values]
        
        # Batch time is roughly: max_time + overhead for managing multiple conditions
        # Not the sum, because they run in parallel
        max_time = max(individual_times) if individual_times else 0
        avg_time = np.mean(individual_times) if individual_times else 0
        
        # Total time is between max (perfect parallelization) and sum (sequential)
        # Use weighted average favoring max with some serial component
        batch_size = len(s_values)
        parallelization_efficiency = max(0.5, 1.0 / np.sqrt(batch_size))  # Efficiency decreases with batch size
        
        estimated_time = max_time + (avg_time * batch_size - max_time) * (1 - parallelization_efficiency)
        
        return estimated_time
    
    def update_from_batch(self, s_values, n_replicates, actual_time):
        """
        Update estimator based on actual batch performance.
        
        Parameters:
        -----------
        s_values : list of float
            Selective advantage values that were run
        n_replicates : int
            Number of replicates per condition
        actual_time : float
            Actual time taken for the batch
        """
        avg_s = np.mean(s_values)
        batch_size = len(s_values)
        time_per_point = actual_time / batch_size
        
        self.batch_history.append({
            'batch_size': batch_size,
            'avg_s': avg_s,
            'actual_time': actual_time,
            'time_per_point': time_per_point,
            'n_replicates': n_replicates
        })
        
        # Update overhead factor based on recent history (last 5 batches)
        if len(self.batch_history) >= 2:
            recent = self.batch_history[-5:]
            
            # Calculate observed overhead relative to base estimates
            overhead_factors = []
            for batch in recent:
                base_estimate = 0
                log_s = np.log10(batch['avg_s'])
                log_s_data = np.log10(self.timing_data[:, 0])
                log_time_data = np.log10(self.timing_data[:, 1])
                log_time = np.interp(log_s, log_s_data[::-1], log_time_data[::-1])
                base_time_per_rep = 10 ** log_time
                
                expected_base = base_time_per_rep * batch['n_replicates']
                if expected_base > 0:
                    overhead_factors.append(batch['time_per_point'] / expected_base)
            
            if overhead_factors:
                # Exponential moving average for smoother updates
                new_factor = np.mean(overhead_factors)
                self.batch_overhead_factor = 0.7 * self.batch_overhead_factor + 0.3 * new_factor
                print(f"  Updated timing overhead factor: {self.batch_overhead_factor:.2f}x")


def create_config_file(params_dict, n_replicates, output_dir, config_name):
    """
    Create a simulation configuration file.
    
    Parameters:
    -----------
    params_dict : dict
        Biological parameters (s, m, q, r, l, idle)
    n_replicates : int
        Number of replicates
    output_dir : str
        Output directory
    config_name : str
        Configuration file name
        
    Returns:
    --------
    str
        Path to created config file
    """
    config = {
        "simulation": {
            "initial_size": 1000,
            "output_dir": str(output_dir),
            "output_prefix": config_name,
            "number_of_replicates": n_replicates,
            "output": {
                "save_consolidated_summary": False
            }
        },
        "biological_parameters": {
            "s": params_dict['s'],
            "m": params_dict['m'],
            "q": params_dict['q'],
            "r": params_dict['r'],
            "l": params_dict['l'],
            "idle": params_dict['idle']
        },
        "treatment": {
            "schedule_type": "off"
        },
        "use_multiprocessing": True
    }
    
    config_path = Path(output_dir) / f"{config_name}.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return str(config_path)


def create_batch_config_file(params_list, n_replicates, output_dir, config_name):
    """
    Create a multi-condition simulation configuration file for batch processing.
    
    Parameters:
    -----------
    params_list : list of dict
        List of biological parameter dictionaries (s, m, q, r, l, idle)
    n_replicates : int
        Number of replicates per condition
    output_dir : str
        Output directory
    config_name : str
        Configuration file name
        
    Returns:
    --------
    str
        Path to created config file
    """
    # Create simulations list with all conditions
    simulations = []
    for params_dict in params_list:
        simulations.append({
            "biological_parameters": {
                "s": params_dict['s'],
                "m": params_dict['m'],
                "q": params_dict['q'],
                "r": params_dict['r'],
                "l": params_dict['l'],
                "idle": params_dict['idle']
            },
            "simulation": {
                "initial_size": 1000,
                "number_of_replicates": n_replicates,
                "output": {
                    "save_consolidated_summary": False
                }
            },
            "treatment": {
                "schedule_type": "off"
            }
        })
    
    config = {
        "simulation": {
            "output_dir": str(output_dir),
            "output_prefix": config_name
        },
        "simulations": simulations,
        "use_multiprocessing": True,
        "account_for_extinctions": True
    }
    
    config_path = Path(output_dir) / f"{config_name}.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return str(config_path)


def fit_burr_distribution(resistance_proportions):
    """
    Fit Burr XII distribution to resistance proportions.
    
    Parameters:
    -----------
    resistance_proportions : array-like
        Resistance proportions from simulations
        
    Returns:
    --------
    dict
        Fitted Burr XII parameters and quality metrics
    """
    try:
        # Fit Burr XII distribution
        params = burr12.fit(resistance_proportions, floc=0)
        c, d, loc, scale = params
        
        # Calculate goodness-of-fit
        from scipy.stats import kstest
        ks_stat, ks_pval = kstest(resistance_proportions, 
                                   lambda x: burr12.cdf(x, c, d, loc, scale))
        
        # Calculate log-likelihood
        log_likelihood = np.sum(burr12.logpdf(resistance_proportions, c, d, loc, scale))
        
        return {
            'c': c,
            'd': d,
            'loc': loc,
            'scale': scale,
            'ks_statistic': ks_stat,
            'ks_pvalue': ks_pval,
            'log_likelihood': log_likelihood,
            'n_samples': len(resistance_proportions),
            'success': True
        }
    except Exception as e:
        print(f"Warning: Failed to fit Burr XII: {e}")
        return {
            'c': np.nan,
            'd': np.nan,
            'loc': np.nan,
            'scale': np.nan,
            'ks_statistic': np.nan,
            'ks_pvalue': np.nan,
            'log_likelihood': np.nan,
            'n_samples': len(resistance_proportions),
            'success': False
        }


def run_batch_simulations(point_ids, normalized_points, param_space, n_replicates, output_dir, batch_name, num_workers=None):
    """
    Run simulations for a batch of parameter points using multi-condition config.
    
    Parameters:
    -----------
    point_ids : list
        List of point IDs
    normalized_points : list of arrays
        List of normalized parameter points
    param_space : ParameterSpace
        Parameter space object
    n_replicates : int
        Number of replicates per point
    output_dir : str
        Output directory
    batch_name : str
        Name for this batch
    num_workers : int, optional
        Number of workers for multiprocessing
        
    Returns:
    --------
    tuple of (list of dict, float, list of float)
        (Results for each point, actual_batch_time, s_values)
    """
    # Convert all points to biological parameters
    params_list = [param_space.from_unit_hypercube(point) for point in normalized_points]
    s_values = [p['s'] for p in params_list]
    
    # Create batch config file
    config_name = f"emulator_batch_{batch_name}"
    config_path = create_batch_config_file(params_list, n_replicates, output_dir, config_name)
    
    print(f"  Created batch config with {len(params_list)} conditions")
    print(f"  Config file: {config_path}")
    print(f"  s values range: [{min(s_values):.2e}, {max(s_values):.2e}]")
    
    try:
        # Run all simulations with multiprocessing
        start_time = time.time()
        if num_workers is None:
            num_workers = cpu_count()
        
        results_list = run_simulation_from_config(config_path, num_workers=num_workers)
        elapsed_time = time.time() - start_time
        
        print(f"  Batch completed in {elapsed_time:.1f}s ({elapsed_time/3600:.2f} hours)")
        print(f"  Average time per condition: {elapsed_time/len(params_list):.1f}s")
        
        # Process results for each condition
        batch_results = []
        for i, (point_id, normalized_point, params_dict) in enumerate(zip(point_ids, normalized_points, params_list)):
            # Extract results for this condition (each condition returns n_replicates results)
            # Results are ordered by condition index
            condition_start = i * n_replicates
            condition_end = (i + 1) * n_replicates
            condition_results = results_list[condition_start:condition_end] if condition_start < len(results_list) else []
            
            try:
                # Calculate resistance proportions
                resistance_props = calculate_resistance_proportions(condition_results)
                
                # Fit Burr XII distribution
                fit_result = fit_burr_distribution(resistance_props)
                
                batch_results.append({
                    'point_id': point_id,
                    'normalized_point': normalized_point,
                    'params': params_dict,
                    'resistance_proportions': resistance_props,
                    'burr_params': fit_result,
                    'n_replicates': n_replicates,
                    'n_successful': len(resistance_props),
                    'elapsed_time': elapsed_time / len(params_list),  # Approximate per-point time
                    'success': True
                })
            except Exception as e:
                print(f"  Error processing point {point_id}: {e}")
                batch_results.append({
                    'point_id': point_id,
                    'normalized_point': normalized_point,
                    'params': params_dict,
                    'resistance_proportions': np.array([]),
                    'burr_params': {
                        'c': np.nan, 'd': np.nan, 'loc': np.nan, 'scale': np.nan,
                        'success': False
                    },
                    'n_replicates': n_replicates,
                    'n_successful': 0,
                    'elapsed_time': 0,
                    'success': False,
                    'error': str(e)
                })
        
        # Clean up config file
        Path(config_path).unlink(missing_ok=True)
        
        return batch_results, elapsed_time, s_values
        
    except Exception as e:
        print(f"  Error running batch: {e}")
        # Clean up config file
        Path(config_path).unlink(missing_ok=True)
        
        # Return failed results for all points
        failed_results = [{
            'point_id': point_id,
            'normalized_point': normalized_point,
            'params': param_space.from_unit_hypercube(normalized_point),
            'resistance_proportions': np.array([]),
            'burr_params': {
                'c': np.nan, 'd': np.nan, 'loc': np.nan, 'scale': np.nan,
                'success': False
            },
            'n_replicates': n_replicates,
            'n_successful': 0,
            'elapsed_time': 0,
            'success': False,
            'error': str(e)
        } for point_id, normalized_point in zip(point_ids, normalized_points)]
        
        return failed_results, 0, s_values


class BurrEmulator:
    """
    Emulator for predicting Burr XII distribution parameters across parameter space.
    Uses Gaussian Process interpolation.
    """
    
    def __init__(self, param_space):
        """
        Initialize emulator.
        
        Parameters:
        -----------
        param_space : ParameterSpace
            Parameter space transformer
        """
        self.param_space = param_space
        self.training_points = []
        self.training_burr_params = []
        
        # Gaussian Process models for each Burr parameter
        self.gp_models = {
            'c': None,
            'd': None,
            'loc': None,
            'scale': None
        }
        
    def add_training_point(self, normalized_point, burr_params):
        """
        Add a training point to the emulator.
        
        Parameters:
        -----------
        normalized_point : array-like
            Point in unit hypercube
        burr_params : dict
            Fitted Burr XII parameters (c, d, loc, scale)
        """
        self.training_points.append(normalized_point)
        self.training_burr_params.append(burr_params)
        
    def train(self, kernel_scale=0.1):
        """
        Train Gaussian Process models for interpolation.
        
        Parameters:
        -----------
        kernel_scale : float
            Length scale for RBF kernel
        """
        if len(self.training_points) == 0:
            raise ValueError("No training points added")
        
        X = np.array(self.training_points)
        
        print(f"\nTraining emulator with {len(X)} points...")
        
        # Train GP for each Burr parameter
        for param_name in ['c', 'd', 'loc', 'scale']:
            # Extract target values (log-transform for stability)
            y = np.array([bp[param_name] for bp in self.training_burr_params])
            
            # Remove any NaN values
            valid_mask = ~np.isnan(y)
            if not valid_mask.all():
                print(f"  Warning: {(~valid_mask).sum()} NaN values in {param_name}")
            
            X_valid = X[valid_mask]
            y_valid = y[valid_mask]
            
            # Log-transform for better GP behavior (except loc which can be negative)
            if param_name != 'loc':
                y_valid = np.log(y_valid + 1e-10)
            
            # Train GP
            kernel = ConstantKernel(1.0) * RBF(length_scale=kernel_scale)
            gp = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=5,
                alpha=1e-6,
                normalize_y=True
            )
            gp.fit(X_valid, y_valid)
            
            self.gp_models[param_name] = gp
            print(f"  Trained GP for {param_name}")
        
        print("Emulator training complete!")
        
    def predict(self, normalized_point, return_std=False):
        """
        Predict Burr XII parameters for a given point.
        
        Parameters:
        -----------
        normalized_point : array-like
            Point in unit hypercube
        return_std : bool
            Whether to return prediction uncertainty
            
        Returns:
        --------
        dict
            Predicted Burr XII parameters
        dict (optional)
            Standard deviations if return_std=True
        """
        if self.gp_models['c'] is None:
            raise ValueError("Emulator not trained yet")
        
        X_query = np.array(normalized_point).reshape(1, -1)
        
        predictions = {}
        stds = {}
        
        for param_name in ['c', 'd', 'loc', 'scale']:
            if return_std:
                y_pred, y_std = self.gp_models[param_name].predict(X_query, return_std=True)
            else:
                y_pred = self.gp_models[param_name].predict(X_query)
                y_std = None
            
            # Inverse transform (except loc)
            if param_name != 'loc':
                predictions[param_name] = np.exp(y_pred[0])
            else:
                predictions[param_name] = y_pred[0]
            
            if return_std:
                stds[param_name] = y_std[0]
        
        if return_std:
            return predictions, stds
        else:
            return predictions
    
    def predict_uncertainty(self, normalized_points):
        """
        Predict uncertainty (variance) for multiple points.
        
        Parameters:
        -----------
        normalized_points : array-like
            Points in unit hypercube, shape (n_points, 6)
            
        Returns:
        --------
        numpy.array
            Total variance for each point (sum across Burr parameters)
        """
        X_query = np.array(normalized_points)
        
        total_variance = np.zeros(len(X_query))
        
        for param_name in ['c', 'd', 'loc', 'scale']:
            _, y_std = self.gp_models[param_name].predict(X_query, return_std=True)
            total_variance += y_std ** 2
        
        return total_variance
    
    def save(self, filepath):
        """Save emulator to file."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'training_points': self.training_points,
                'training_burr_params': self.training_burr_params,
                'gp_models': self.gp_models,
                'param_space': self.param_space
            }, f)
        print(f"Emulator saved to: {filepath}")
    
    @staticmethod
    def load(filepath):
        """Load emulator from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        emulator = BurrEmulator(data['param_space'])
        emulator.training_points = data['training_points']
        emulator.training_burr_params = data['training_burr_params']
        emulator.gp_models = data['gp_models']
        
        print(f"Emulator loaded from: {filepath}")
        return emulator


def adaptive_refinement(emulator, param_space, n_new_points, existing_points):
    """
    Select new points to sample based on prediction uncertainty.
    
    Parameters:
    -----------
    emulator : BurrEmulator
        Trained emulator
    param_space : ParameterSpace
        Parameter space
    n_new_points : int
        Number of new points to select
    existing_points : array-like
        Already sampled points in unit hypercube
        
    Returns:
    --------
    numpy.array
        New points to sample, shape (n_new_points, 6)
    """
    print(f"\nAdaptive refinement: selecting {n_new_points} new points...")
    
    # Generate candidate points using dense LH sampling
    n_candidates = n_new_points * 100
    candidates = param_space.latin_hypercube_sample(n_candidates)
    
    # Remove candidates too close to existing points
    min_distance = 0.1  # Minimum distance in normalized space
    if len(existing_points) > 0:
        distances = cdist(candidates, existing_points)
        min_dist_to_existing = distances.min(axis=1)
        valid_candidates = candidates[min_dist_to_existing > min_distance]
    else:
        valid_candidates = candidates
    
    if len(valid_candidates) < n_new_points:
        print(f"  Warning: Only {len(valid_candidates)} valid candidates found")
        valid_candidates = candidates  # Use all candidates
    
    # Calculate uncertainty for candidates
    uncertainties = emulator.predict_uncertainty(valid_candidates)
    
    # Select points with highest uncertainty
    top_indices = np.argsort(uncertainties)[-n_new_points:]
    new_points = valid_candidates[top_indices]
    
    print(f"  Selected {len(new_points)} points with mean uncertainty: {uncertainties[top_indices].mean():.4f}")
    
    return new_points


def build_emulator(max_time_seconds, output_dir='./emulator_output', n_replicates=100, 
                   initial_lhs_points=20, refinement_points_per_round=10):
    """
    Build resistance distribution emulator with time budget.
    
    Parameters:
    -----------
    max_time_seconds : float
        Maximum time budget in seconds
    output_dir : str
        Output directory for results
    n_replicates : int
        Number of replicates per parameter point
    initial_lhs_points : int
        Number of initial LHS points
    refinement_points_per_round : int
        Number of points to add per refinement round
        
    Returns:
    --------
    BurrEmulator
        Trained emulator
    """
    start_time = time.time()
    end_time = start_time + max_time_seconds
    
    # Setup
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    param_space = ParameterSpace()
    timing_estimator = TimingEstimator()
    emulator = BurrEmulator(param_space)
    
    # Results storage
    all_results = []
    
    print("="*80)
    print("RESISTANCE DISTRIBUTION EMULATOR BUILDER")
    print("="*80)
    print(f"Time budget: {max_time_seconds/3600:.2f} hours ({max_time_seconds} seconds)")
    print(f"Output directory: {output_dir}")
    print(f"Replicates per point: {n_replicates}")
    print(f"Initial LHS points: {initial_lhs_points}")
    print(f"Refinement points per round: {refinement_points_per_round}")
    print("="*80)
    
    # Phase 1: Initial LHS sampling
    print("\nPHASE 1: Initial Latin Hypercube Sampling")
    print("="*80)
    
    lhs_samples = param_space.latin_hypercube_sample(initial_lhs_points, seed=42)
    
    # Estimate time for initial points using batch estimation
    s_values_all = [param_space.from_unit_hypercube(sample)['s'] for sample in lhs_samples]
    total_estimated_time = timing_estimator.estimate_batch_time(s_values_all, n_replicates)
    
    print(f"Estimated time for {initial_lhs_points} LHS points: {total_estimated_time/3600:.2f} hours")
    
    if total_estimated_time > max_time_seconds * 0.7:
        # Reduce number of points if estimate is too high
        scale_factor = (max_time_seconds * 0.5) / total_estimated_time
        new_n_points = max(5, int(initial_lhs_points * scale_factor))
        print(f"WARNING: Reducing initial points to {new_n_points} to fit time budget")
        lhs_samples = lhs_samples[:new_n_points]
    
    # Determine optimal batch size based on time budget and diversity
    # Smaller batches = faster feedback, but more overhead
    # Larger batches = better parallelization, but slower feedback
    max_batch_size = max(3, min(10, len(lhs_samples) // 2))  # Start with smaller batches
    
    print(f"\nRunning {len(lhs_samples)} LHS points in batches of ~{max_batch_size}...")
    
    # Determine number of workers
    num_workers = cpu_count()
    print(f"Using {num_workers} workers for multiprocessing")
    
    # Split into batches
    point_id = 0
    batch_num = 0
    for batch_start in range(0, len(lhs_samples), max_batch_size):
        batch_end = min(batch_start + max_batch_size, len(lhs_samples))
        batch_samples = lhs_samples[batch_start:batch_end]
        batch_point_ids = list(range(point_id, point_id + len(batch_samples)))
        
        if time.time() > end_time:
            print(f"\nTime budget exhausted after {batch_num} batches")
            break
        
        print(f"\n--- Batch {batch_num + 1} ({len(batch_samples)} points) ---")
        
        # Run batch
        batch_results, actual_time, s_vals = run_batch_simulations(
            point_ids=batch_point_ids,
            normalized_points=batch_samples,
            param_space=param_space,
            n_replicates=n_replicates,
            output_dir=output_dir,
            batch_name=f"lhs_batch{batch_num}",
            num_workers=num_workers
        )
        
        # Update timing estimator with actual performance
        timing_estimator.update_from_batch(s_vals, n_replicates, actual_time)
        
        # Process results
        for result in batch_results:
            if result['success'] and result['burr_params']['success']:
                emulator.add_training_point(result['normalized_point'], result['burr_params'])
                all_results.append(result)
                print(f"\nPoint {result['point_id']}: ✓ Success")
                print(f"  s={result['params']['s']:.2e}, m={result['params']['m']:.2e}")
                print(f"  Burr params: c={result['burr_params']['c']:.4f}, d={result['burr_params']['d']:.4f}")
            else:
                print(f"\nPoint {result['point_id']}: ✗ Failed")
        
        point_id += len(batch_samples)
        batch_num += 1
        
        # Save intermediate results after each batch
        save_results(all_results, output_path / 'intermediate_results.csv')
    
    # Train initial emulator
    if len(emulator.training_points) >= 3:
        emulator.train()
    else:
        print("\nWARNING: Not enough successful points to train emulator")
        save_results(all_results, output_path / 'final_results.csv')
        return emulator
    
    # Phase 2: Adaptive Refinement
    print("\n" + "="*80)
    print("PHASE 2: Adaptive Refinement")
    print("="*80)
    
    round_num = 1
    while time.time() < end_time and len(emulator.training_points) < 200:
        time_remaining = end_time - time.time()
        if time_remaining < 60:
            print(f"\nLess than 1 minute remaining, stopping refinement")
            break
        
        print(f"\nRefinement Round {round_num}")
        print("-" * 40)
        
        # Select new points based on uncertainty
        existing_points = np.array(emulator.training_points)
        new_points = adaptive_refinement(
            emulator, param_space, 
            n_new_points=refinement_points_per_round,
            existing_points=existing_points
        )
        
        # Run simulations for new points in batch
        new_point_ids = list(range(point_id, point_id + len(new_points)))
        
        print(f"Running {len(new_points)} refinement points in batch...")
        
        # Run batch
        batch_results, actual_time, s_vals = run_batch_simulations(
            point_ids=new_point_ids,
            normalized_points=new_points,
            param_space=param_space,
            n_replicates=n_replicates,
            output_dir=output_dir,
            batch_name=f"refinement_round{round_num}",
            num_workers=num_workers
        )
        
        # Update timing estimator
        timing_estimator.update_from_batch(s_vals, n_replicates, actual_time)
        
        # Process results
        points_completed = 0
        for result in batch_results:
            if result['success'] and result['burr_params']['success']:
                emulator.add_training_point(result['normalized_point'], result['burr_params'])
                all_results.append(result)
                points_completed += 1
                print(f"\nPoint {result['point_id']}: ✓ Success")
                print(f"  s={result['params']['s']:.2e}")
            else:
                print(f"\nPoint {result['point_id']}: ✗ Failed")
        
        point_id += len(new_points)
        
        # Save intermediate results after each refinement round
        save_results(all_results, output_path / 'intermediate_results.csv')
        
        # Retrain emulator with new points
        if points_completed > 0:
            emulator.train()
            print(f"\nEmulator retrained with {len(emulator.training_points)} total points")
        
        round_num += 1
    
    # Save final results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    # Save all simulation results
    results_path = output_path / 'final_results.csv'
    save_results(all_results, results_path)
    
    # Save emulator
    emulator_path = output_path / 'resistance_emulator.pkl'
    emulator.save(emulator_path)
    
    # Save summary
    summary = {
        'total_points_sampled': len(all_results),
        'successful_points': len(emulator.training_points),
        'total_time_seconds': time.time() - start_time,
        'output_directory': str(output_path),
        'n_replicates': n_replicates,
        'timestamp': datetime.now().isoformat()
    }
    
    summary_path = output_path / 'emulator_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Total points sampled: {len(all_results)}")
    print(f"Successful fits: {len(emulator.training_points)}")
    print(f"Total elapsed time: {(time.time() - start_time)/3600:.2f} hours")
    print(f"Results saved to: {results_path}")
    print(f"Emulator saved to: {emulator_path}")
    print("="*80)
    
    return emulator


def save_results(results_list, filepath):
    """
    Save all simulation results to CSV.
    
    Parameters:
    -----------
    results_list : list
        List of result dictionaries
    filepath : Path or str
        Output file path
    """
    rows = []
    for result in results_list:
        row = {
            'point_id': result['point_id'],
            's': result['params']['s'],
            'm': result['params']['m'],
            'q': result['params']['q'],
            'r': result['params']['r'],
            'l': result['params']['l'],
            'idle': result['params']['idle'],
            'n_replicates': result['n_replicates'],
            'n_successful': result['n_successful'],
            'burr_c': result['burr_params']['c'],
            'burr_d': result['burr_params']['d'],
            'burr_loc': result['burr_params']['loc'],
            'burr_scale': result['burr_params']['scale'],
            'ks_pvalue': result['burr_params']['ks_pvalue'],
            'elapsed_time': result['elapsed_time'],
            'success': result['success']
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)
    print(f"Results saved to: {filepath}")


def predict_from_emulator(emulator_path, params_dict):
    """
    Use saved emulator to predict resistance distribution for given parameters.
    
    Parameters:
    -----------
    emulator_path : str
        Path to saved emulator file
    params_dict : dict
        Biological parameters (s, m, q, r, l, idle)
        
    Returns:
    --------
    dict
        Predicted Burr XII parameters and distribution object
    """
    # Load emulator
    emulator = BurrEmulator.load(emulator_path)
    
    # Transform parameters to unit hypercube
    normalized_point = emulator.param_space.to_unit_hypercube(params_dict)
    
    # Predict Burr parameters
    burr_params, stds = emulator.predict(normalized_point, return_std=True)
    
    print("\nPredicted Burr XII Distribution Parameters:")
    print("="*50)
    for param, value in burr_params.items():
        print(f"  {param}: {value:.6f} (±{stds[param]:.6f})")
    print("="*50)
    
    # Create distribution object
    dist = burr12(c=burr_params['c'], d=burr_params['d'], 
                  loc=burr_params['loc'], scale=burr_params['scale'])
    
    # Calculate summary statistics
    mean = dist.mean()
    median = dist.median()
    std = dist.std()
    
    print("\nPredicted Resistance Proportion Distribution:")
    print(f"  Mean: {mean:.6f}")
    print(f"  Median: {median:.6f}")
    print(f"  Std Dev: {std:.6f}")
    print(f"  95% CI: [{dist.ppf(0.025):.6f}, {dist.ppf(0.975):.6f}]")
    
    return {
        'burr_params': burr_params,
        'uncertainties': stds,
        'distribution': dist,
        'mean': mean,
        'median': median,
        'std': std
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python build_resistance_emulator.py <max_time_hours>")
        print("\nExample:")
        print("  python build_resistance_emulator.py 10  # Run for 10 hours")
        print("\nOptional: To use existing emulator for prediction:")
        print("  python build_resistance_emulator.py --predict emulator_output/resistance_emulator.pkl")
        sys.exit(1)
    
    if sys.argv[1] == "--predict":
        # Prediction mode
        if len(sys.argv) < 3:
            print("Error: Please provide path to emulator file")
            sys.exit(1)
        
        emulator_path = sys.argv[2]
        
        # Example prediction (user can modify these parameters)
        example_params = {
            's': 0.01,
            'm': 1e-6,
            'q': 1e-6,
            'r': 1e-8,
            'l': 1e-5,
            'idle': 0.1
        }
        
        print("\nExample prediction with parameters:")
        for param, value in example_params.items():
            print(f"  {param}: {value}")
        
        result = predict_from_emulator(emulator_path, example_params)
        
    else:
        # Building mode
        max_time_hours = float(sys.argv[1])
        max_time_seconds = max_time_hours * 3600
        
        # Build emulator
        emulator = build_emulator(
            max_time_seconds=max_time_seconds,
            output_dir='./emulator_output',
            n_replicates=100,
            initial_lhs_points=20,
            refinement_points_per_round=5
        )
