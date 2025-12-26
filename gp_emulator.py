"""
Gaussian Process Emulator for Tumour Simulation
Called from C++ bootstrap_manager to train and validate GP models
"""

import json
import numpy as np
import pickle
import sys
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

def validate_parameters(params):
    """
    Validate that parameters are within valid ranges and satisfy constraints
    Returns: (is_valid, error_message)
    """
    # Check parameter ranges
    if not (0.001 <= params['s'] <= 0.1):
        return False, f"Selection coefficient s={params['s']} outside valid range [0.001, 0.1]"
    
    if not (1e-8 <= params['m'] <= 1e-5):
        return False, f"Mutation rate m={params['m']} outside valid range [1e-8, 1e-5]"
    
    if not (1e-11 <= params['q'] <= 1e-2):
        return False, f"Transition rate q={params['q']} outside valid range [1e-11, 1e-2]"
    
    if not (1e-11 <= params['r'] <= 1e-2):
        return False, f"Transition rate r={params['r']} outside valid range [1e-11, 1e-2]"
    
    if not (1e-11 <= params['l'] <= 1e-2):
        return False, f"Transition rate l={params['l']} outside valid range [1e-11, 1e-2]"
    
    if not (0.0 <= params['idle'] <= 0.5):
        return False, f"Idle probability idle={params['idle']} outside valid range [0.0, 0.5]"
    
    # Check birth > death constraint
    prob_death = ((1 - params['idle']) / 2.0) * (1 - params['s'])
    prob_mutation = params['m'] * (1 - prob_death)
    prob_to_transient = params['q'] * (1 - prob_death)
    prob_to_resistant = params['r'] * (1 - prob_death)
    prob_to_sensitive = params['l'] * (1 - prob_death)
    prob_birth = 1 - params['idle'] - prob_death - prob_mutation - \
                 prob_to_transient - prob_to_resistant - prob_to_sensitive
    
    if prob_birth <= prob_death:
        return False, (f"Birth probability ({prob_birth:.6f}) must be greater than "
                      f"death probability ({prob_death:.6f}). "
                      f"This parameter combination is biologically invalid.")
    
    return True, ""

def load_training_data(filename='emulator_training_data.json'):
    """Load training data from JSON file"""
    with open(filename, 'r') as f:
        data = json.load(f)
    
    X = []  # Parameters
    y_tr_50 = []  # Total resistant 50th percentile
    y_tr_75 = []  # Total resistant 75th percentile
    y_tr_90 = []  # Total resistant 90th percentile
    y_tf_50 = []  # Transient fraction 50th percentile
    y_tf_75 = []  # Transient fraction 75th percentile
    y_tf_90 = []  # Transient fraction 90th percentile
    sample_ids = []
    
    for sample in data['samples']:
        params = sample['parameters']
        results = sample['results']
        
        # Skip samples with no results (from slow regions)
        if results['bootstrap_samples'] == 0:
            continue
        
        sample_ids.append(params['sample_id'])
        
        # Input: log-transform parameters for better GP performance
        X.append([
            np.log10(params['s']),
            np.log10(params['m']),
            np.log10(params['q']),
            np.log10(params['r']),
            np.log10(params['l']),
            params['idle']
        ])
        
        # Outputs: use midpoint of confidence intervals
        tr = results['total_resistant']
        y_tr_50.append((tr['50th']['lower'] + tr['50th']['upper']) / 2.0)
        y_tr_75.append((tr['75th']['lower'] + tr['75th']['upper']) / 2.0)
        y_tr_90.append((tr['90th']['lower'] + tr['90th']['upper']) / 2.0)
        
        tf = results['transient_fraction']
        y_tf_50.append((tf['50th']['lower'] + tf['50th']['upper']) / 2.0)
        y_tf_75.append((tf['75th']['lower'] + tf['75th']['upper']) / 2.0)
        y_tf_90.append((tf['90th']['lower'] + tf['90th']['upper']) / 2.0)
    
    X = np.array(X)
    y_tr_50 = np.array(y_tr_50)
    y_tr_75 = np.array(y_tr_75)
    y_tr_90 = np.array(y_tr_90)
    y_tf_50 = np.array(y_tf_50)
    y_tf_75 = np.array(y_tf_75)
    y_tf_90 = np.array(y_tf_90)
    
    return X, y_tr_50, y_tr_75, y_tr_90, y_tf_50, y_tf_75, y_tf_90, sample_ids, data['metadata']

def train_gp_emulator(X, y, name):
    """Train a Gaussian Process emulator"""
    print(f"  Training GP for {name}... ", end='', flush=True)
    
    # Standardize inputs
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    
    # Standardize outputs
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
    
    # Define kernel: RBF with white noise
    kernel = ConstantKernel(1.0) * RBF(length_scale=np.ones(X.shape[1])) + WhiteKernel(noise_level=0.01)
    
    # Train GP
    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=10,
        alpha=1e-6,
        normalize_y=False
    )
    
    gp.fit(X_scaled, y_scaled)
    
    print(f"Done (log-likelihood: {gp.log_marginal_likelihood_value_:.2f})")
    
    return gp, scaler_X, scaler_y

def cross_validate_emulator(X, y_dict, sample_ids, n_folds=5):
    """
    Perform k-fold cross-validation on all outputs
    Returns: list of dicts with validation results for each sample
    """
    print(f"\n  Performing {n_folds}-fold cross-validation...")
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    cv_results = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"    Fold {fold_idx + 1}/{n_folds}...", end='', flush=True)
        
        X_train, X_test = X[train_idx], X[test_idx]
        
        # Train GPs on training fold
        fold_gps = {}
        for output_name, y in y_dict.items():
            y_train = y[train_idx]
            gp, scaler_X, scaler_y = train_gp_emulator(X_train, y_train, f"{output_name}_fold{fold_idx}")
            fold_gps[output_name] = (gp, scaler_X, scaler_y)
        
        # Predict on test fold
        for i, test_sample_idx in enumerate(test_idx):
            X_test_single = X_test[i:i+1]
            sample_id = sample_ids[test_sample_idx]
            
            result = {'sample_id': sample_id}
            
            # Get predictions for each output
            for output_name, y in y_dict.items():
                gp, scaler_X, scaler_y = fold_gps[output_name]
                
                # Scale input
                X_scaled = scaler_X.transform(X_test_single)
                
                # Predict (scaled)
                y_pred_scaled, y_std_scaled = gp.predict(X_scaled, return_std=True)
                
                # Inverse transform
                y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()[0]
                y_std = y_std_scaled[0] * scaler_y.scale_[0]
                
                # Observed value
                y_obs = y[test_sample_idx]
                
                # 95% CI
                ci_lower = y_pred - 1.96 * y_std
                ci_upper = y_pred + 1.96 * y_std
                
                # Check if observed is within CI
                within_ci = (y_obs >= ci_lower) and (y_obs <= ci_upper)
                
                # Distance from CI
                if within_ci:
                    distance = 0.0
                else:
                    distance = min(abs(y_obs - ci_lower), abs(y_obs - ci_upper))
                
                result[output_name] = {
                    'observed': y_obs,
                    'predicted': y_pred,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'within_ci': within_ci,
                    'distance': distance
                }
            
            cv_results.append(result)
        
        print(" Done")
    
    return cv_results

def calculate_coverage(cv_results):
    """Calculate coverage statistics from cross-validation results"""
    output_names = ['total_resistant_50th', 'total_resistant_75th', 'total_resistant_90th',
                   'transient_fraction_50th', 'transient_fraction_75th', 'transient_fraction_90th']
    
    coverage_stats = {}
    
    for output_name in output_names:
        within_count = sum(1 for r in cv_results if r[output_name]['within_ci'])
        total_count = len(cv_results)
        coverage = within_count / total_count if total_count > 0 else 0.0
        
        # Calculate average distance for samples outside CI
        distances = [r[output_name]['distance'] for r in cv_results if not r[output_name]['within_ci']]
        avg_distance = np.mean(distances) if distances else 0.0
        
        coverage_stats[output_name] = {
            'coverage': coverage,
            'within_ci': within_count,
            'total': total_count,
            'avg_distance_outside': avg_distance
        }
    
    # Overall coverage (all outputs must be within CI)
    all_within = sum(1 for r in cv_results if all(r[name]['within_ci'] for name in output_names))
    overall_coverage = all_within / len(cv_results) if cv_results else 0.0
    
    coverage_stats['overall'] = {
        'coverage': overall_coverage,
        'within_ci': all_within,
        'total': len(cv_results)
    }
    
    return coverage_stats

def identify_problem_regions(cv_results, X, sample_ids):
    """
    Identify parameter regions where emulator performs poorly
    Returns: list of sample indices that need more neighbors
    """
    output_names = ['total_resistant_50th', 'total_resistant_75th', 'total_resistant_90th',
                   'transient_fraction_50th', 'transient_fraction_75th', 'transient_fraction_90th']
    
    problem_samples = []
    
    for result in cv_results:
        # Check if any output is outside CI
        any_outside = any(not result[name]['within_ci'] for name in output_names)
        
        if any_outside:
            # Calculate combined distance (weighted by which outputs failed)
            total_distance = sum(result[name]['distance'] for name in output_names 
                               if not result[name]['within_ci'])
            
            sample_id = result['sample_id']
            
            # Find the sample index
            sample_idx = sample_ids.index(sample_id)
            
            problem_samples.append({
                'sample_id': sample_id,
                'sample_idx': sample_idx,
                'parameters': X[sample_idx].tolist(),
                'total_distance': total_distance
            })
    
    # Sort by distance (worst first)
    problem_samples.sort(key=lambda x: x['total_distance'], reverse=True)
    
    return problem_samples

def predict(emulator_dict, params):
    """
    Make predictions for given parameters
    
    params: dict with keys 's', 'm', 'q', 'r', 'l', 'idle'
    Returns: dict with predictions and uncertainties, or error dict
    """
    # Validate parameters first
    is_valid, error_msg = validate_parameters(params)
    if not is_valid:
        return {
            'error': True,
            'message': error_msg,
            'params': params
        }
    
    # Transform parameters to log-space (except idle)
    X_test = np.array([[
        np.log10(params['s']),
        np.log10(params['m']),
        np.log10(params['q']),
        np.log10(params['r']),
        np.log10(params['l']),
        params['idle']
    ]])
    
    results = {'error': False}
    
    for output_name in ['total_resistant_50th', 'total_resistant_75th', 'total_resistant_90th',
                        'transient_fraction_50th', 'transient_fraction_75th', 'transient_fraction_90th']:
        gp, scaler_X, scaler_y = emulator_dict[output_name]
        
        # Scale input
        X_test_scaled = scaler_X.transform(X_test)
        
        # Predict (scaled)
        y_pred_scaled, y_std_scaled = gp.predict(X_test_scaled, return_std=True)
        
        # Inverse transform
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()[0]
        y_std = y_std_scaled[0] * scaler_y.scale_[0]
        
        results[output_name] = {
            'mean': y_pred,
            'std': y_std,
            'lower_95': y_pred - 1.96 * y_std,
            'upper_95': y_pred + 1.96 * y_std
        }
    
    return results

def main():
    if len(sys.argv) < 2:
        print("Usage: python gp_emulator.py <command> [args]")
        print("Commands:")
        print("  train                  - Train GPs and save model")
        print("  validate              - Train GPs and perform cross-validation")
        print("  predict <params.json> - Make predictions for parameters in JSON file")
        sys.exit(1)
    
    command = sys.argv[1]
    
    print("=== GP Emulator ===")
    print(f"Command: {command}\n")
    
    if command == "predict":
        if len(sys.argv) < 3:
            print("Error: predict command requires parameter file")
            print("Usage: python gp_emulator.py predict <params.json>")
            sys.exit(1)
        
        # Load emulator
        try:
            with open('gp_emulator_model.pkl', 'rb') as f:
                emulator = pickle.load(f)
        except FileNotFoundError:
            print("Error: No trained emulator found. Run 'train' command first.")
            sys.exit(1)
        
        # Load parameters
        param_file = sys.argv[2]
        try:
            with open(param_file, 'r') as f:
                params = json.load(f)
        except FileNotFoundError:
            print(f"Error: Parameter file '{param_file}' not found")
            sys.exit(1)
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in '{param_file}'")
            sys.exit(1)
        
        # Make prediction
        results = predict(emulator, params)
        
        if results.get('error', False):
            print("\n❌ INVALID PARAMETERS")
            print("=" * 60)
            print(results['message'])
            print("\nProvided parameters:")
            for key, value in results['params'].items():
                print(f"  {key}: {value}")
            print("\nValid ranges:")
            print("  s (selection):     [0.001, 0.1]")
            print("  m (mutation):      [1e-8, 1e-5]")
            print("  q (S->Q):          [1e-11, 1e-2]")
            print("  r (S->R):          [1e-11, 1e-2]")
            print("  l (Q->S):          [1e-11, 1e-2]")
            print("  idle:              [0.0, 0.5]")
            print("\nConstraint: Birth probability must be > Death probability")
            sys.exit(1)
        
        # Display results
        print("✓ Parameters validated successfully")
        print("\nInput parameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        
        print("\n" + "=" * 60)
        print("PREDICTIONS")
        print("=" * 60)
        
        print("\nTotal Resistant:")
        print("  50th percentile:")
        print(f"    Mean: {results['total_resistant_50th']['mean']:.4f}")
        print(f"    95% CI: [{results['total_resistant_50th']['lower_95']:.4f}, "
              f"{results['total_resistant_50th']['upper_95']:.4f}]")
        print("  75th percentile:")
        print(f"    Mean: {results['total_resistant_75th']['mean']:.4f}")
        print(f"    95% CI: [{results['total_resistant_75th']['lower_95']:.4f}, "
              f"{results['total_resistant_75th']['upper_95']:.4f}]")
        print("  90th percentile:")
        print(f"    Mean: {results['total_resistant_90th']['mean']:.4f}")
        print(f"    95% CI: [{results['total_resistant_90th']['lower_95']:.4f}, "
              f"{results['total_resistant_90th']['upper_95']:.4f}]")
        
        print("\nTransient Fraction (of resistant cells):")
        print("  50th percentile:")
        print(f"    Mean: {results['transient_fraction_50th']['mean']:.4f}")
        print(f"    95% CI: [{results['transient_fraction_50th']['lower_95']:.4f}, "
              f"{results['transient_fraction_50th']['upper_95']:.4f}]")
        print("  75th percentile:")
        print(f"    Mean: {results['transient_fraction_75th']['mean']:.4f}")
        print(f"    95% CI: [{results['transient_fraction_75th']['lower_95']:.4f}, "
              f"{results['transient_fraction_75th']['upper_95']:.4f}]")
        print("  90th percentile:")
        print(f"    Mean: {results['transient_fraction_90th']['mean']:.4f}")
        print(f"    95% CI: [{results['transient_fraction_90th']['lower_95']:.4f}, "
              f"{results['transient_fraction_90th']['upper_95']:.4f}]")
        
        # Save results
        output_file = param_file.replace('.json', '_predictions.json')
        with open(output_file, 'w') as f:
            json.dump({
                'input_parameters': params,
                'predictions': results
            }, f, indent=2)
        print(f"\nResults saved to: {output_file}")
        
        return
    
    # Load data for train/validate commands
    print("Loading training data...")
    X, y_tr_50, y_tr_75, y_tr_90, y_tf_50, y_tf_75, y_tf_90, sample_ids, metadata = load_training_data()
    print(f"Loaded {len(X)} valid training samples\n")
    
    if len(X) < 5:
        print("Error: Not enough training samples (need at least 5)")
        sys.exit(1)
    
    if command == "train":
        # Train GPs for all outputs
        print("Training GPs...")
        emulators = {}
        
        y_dict = {
            'total_resistant_50th': y_tr_50,
            'total_resistant_75th': y_tr_75,
            'total_resistant_90th': y_tr_90,
            'transient_fraction_50th': y_tf_50,
            'transient_fraction_75th': y_tf_75,
            'transient_fraction_90th': y_tf_90
        }
        
        for name, y in y_dict.items():
            emulators[name] = train_gp_emulator(X, y, name)
        
        # Save metadata
        emulators['metadata'] = metadata
        emulators['sample_ids'] = sample_ids
        
        # Save to file
        with open('gp_emulator_model.pkl', 'wb') as f:
            pickle.dump(emulators, f)
        
        print("\nModel saved to gp_emulator_model.pkl")
        print("The emulator predicts 6 outputs:")
        print("  - Total Resistant: 50th, 75th, 90th percentiles")
        print("  - Transient Fraction: 50th, 75th, 90th percentiles")
        
    elif command == "validate":
        # Train and cross-validate
        y_dict = {
            'total_resistant_50th': y_tr_50,
            'total_resistant_75th': y_tr_75,
            'total_resistant_90th': y_tr_90,
            'transient_fraction_50th': y_tf_50,
            'transient_fraction_75th': y_tf_75,
            'transient_fraction_90th': y_tf_90
        }
        
        cv_results = cross_validate_emulator(X, y_dict, sample_ids)
        
        # Calculate coverage
        print("\n  Calculating coverage statistics...")
        coverage_stats = calculate_coverage(cv_results)
        
        # Identify problem regions
        print("  Identifying problem regions...")
        problem_regions = identify_problem_regions(cv_results, X, sample_ids)
        
        # Save results
        output = {
            'cv_results': cv_results,
            'coverage_stats': coverage_stats,
            'problem_regions': problem_regions
        }
        
        with open('gp_validation_results.json', 'w') as f:
            json.dump(output, f, indent=2)
        
        print("\n=== Validation Results ===")
        print(f"Overall coverage: {coverage_stats['overall']['coverage']*100:.1f}% "
              f"({coverage_stats['overall']['within_ci']}/{coverage_stats['overall']['total']})")
        
        print("\nTotal Resistant Coverage:")
        for perc in ['50th', '75th', '90th']:
            name = f'total_resistant_{perc}'
            stats = coverage_stats[name]
            print(f"  {perc}: {stats['coverage']*100:.1f}% ({stats['within_ci']}/{stats['total']})")
        
        print("\nTransient Fraction Coverage:")
        for perc in ['50th', '75th', '90th']:
            name = f'transient_fraction_{perc}'
            stats = coverage_stats[name]
            print(f"  {perc}: {stats['coverage']*100:.1f}% ({stats['within_ci']}/{stats['total']})")
        
        print(f"\nProblem regions: {len(problem_regions)}")
        print("Results saved to gp_validation_results.json")
        
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)

if __name__ == '__main__':
    main()
