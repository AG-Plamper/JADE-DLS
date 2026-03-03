# -*- coding: utf-8 -*-
"""
regularized.py
==============
NNLS and regularized inverse Laplace transform fitting for DLS correlation
data. Extracts continuous decay-time distributions, detects peaks and
computes per-peak hydrodynamic radii and shape statistics.

Functions
---------
nnls(df, name, nnls_params, plot_number)
    Plain NNLS fit of g²(τ) to a discrete set of exponential basis
    functions; returns peak positions and amplitudes.

nnls_all(dataframes_dict, nnls_params)
    Apply nnls() across all files and collect results.

calculate_decay_rates(df, tau_columns)
    Convert decay-time columns to decay rates Γ = 1/τ.

nnls_reg(df, name, nnls_reg_params, plot_number)
    Tikhonov-regularized NNLS fit with automatic or fixed regularization
    parameter α; detects peaks, computes area fractions, centroids, FWHM,
    skewness and excess kurtosis per peak; 4-panel diagnostic plot.

nnls_reg_all(dataframes_dict, nnls_reg_params)
    Apply nnls_reg() across all files and collect results.

nnls_reg_simple(df, name, nnls_reg_params)
    Simplified variant of nnls_reg without plotting, used for α-grid search.

analyze_random_datasets_grid(df_dict, num_datasets, base_nnls_params, ...)
    Grid search over regularization parameter α on random data subsets
    to find the optimal α for a given dataset.

tau_to_hydrodynamic_radius(tau, q, temperature, viscosity)
    Convert decay time τ to hydrodynamic radius Rh via Stokes-Einstein.

find_dataset_key(filename, full_results)
    Fuzzy match a filename to a key in the full_results dict.

plot_distributions(full_results, nnls_reg_params, df_basedata_mod, ...)
    Plot regularized decay-time distributions for selected angles,
    with optional conversion to Rh axis.

Dependencies: numpy, pandas, scipy, matplotlib
"""
from scipy.optimize import least_squares, minimize
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import sparse
from scipy import stats
from matplotlib import cm
import random

#code for simple NNLS-Fit to the data
def nnls(df, name, nnls_params, plot_number):
    decay_times = nnls_params['decay_times']
    prominence = nnls_params['prominence']
    distance = nnls_params['distance']
    
    #create the vectors
    tau = df['t [s]'].to_numpy()
    D = df['g(2)-1'].to_numpy()
    
    #create grid of tau and decay time combinations
    decay_times_N, tau_M = np.meshgrid(decay_times, tau)
    
    #create matrix A from the mesh
    T = np.exp(-tau_M / decay_times_N)
    
    #define the residual function
    def residuals(f, T, D):
        return (T @ f)**2 - D
    
    #initial guess for f
    f0 = np.ones(T.shape[1])
    
    #perform the non-negative least squares optimization
    bounds = (0, np.inf)  #lower bound to 0, upper bound to infinity
    result = least_squares(residuals, f0, args=(T, D), bounds=bounds)
    
    #get the optimized f
    f_optimized = result.x
    
    #calculate the optimized function values
    optimized_values = (T @ f_optimized)**2
    residuals_values = optimized_values - D
    
    #find peaks in the tau distribution
    peaks, _ = find_peaks(f_optimized, prominence=prominence, distance=distance)
    
    #get peak amplitudes/intensities
    peak_amplitudes = f_optimized[peaks]
    
    #normalization to the total sum
    normalized_amplitudes_sum = peak_amplitudes / np.sum(peak_amplitudes)

    #R²
    ss_res = np.sum(residuals_values**2)
    ss_tot = np.sum((D - np.mean(D))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    #4-panel layout: data+fit | distribution | residuals | Q-Q
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 6))
    fig.suptitle(f'[{plot_number}]: NNLS — {name}', fontsize=12)

    # Panel 1: data + fit
    ax1.semilogx(tau, D, 'o', alpha=0.6, markersize=4, label='Data')
    ax1.semilogx(tau, optimized_values, 'r-', linewidth=2, label='Fit')
    ax1.set_xlabel(r'lag time τ [s]')
    ax1.set_ylabel(r'$g^{(2)}(τ) - 1$')
    ax1.set_title('Data & Fit')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.text(0.95, 0.95, f'R² = {r_squared:.4f}',
             transform=ax1.transAxes, va='top', ha='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Panel 2: decay time distribution
    ax2.semilogx(decay_times, f_optimized, 'bo-', markersize=3, label='f(τ)')
    ax2.plot(decay_times[peaks], f_optimized[peaks], 'rx', markersize=10, label='Peaks')
    ax2.set_xlabel(r'decay time τ [s]')
    ax2.set_ylabel('f(τ) [a.u.]')
    ax2.set_title('τ Distribution')
    ax2.legend()
    ax2.grid(True, which="both", ls="--", alpha=0.3)

    # Panel 3: residuals
    ax3.plot(residuals_values)
    ax3.axhline(0, color='r', linestyle='--', linewidth=1)
    ax3.set_xlabel('Sample Index')
    ax3.set_ylabel('Residuals')
    ax3.set_title('Residuals')
    ax3.grid(True, alpha=0.3)

    # Panel 4: Q-Q
    from scipy import stats as _stats
    _stats.probplot(residuals_values, dist="norm", plot=ax4)
    ax4.set_title('Q-Q Plot')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    
    #prepare results for this dataframe
    results = {'filename': name}
    
    results['R_squared'] = r_squared
    for i, peak_index in enumerate(peaks):
        percentage = normalized_amplitudes_sum[i] * 100
        results[f'tau_{i+1}'] = decay_times[peak_index]
        results[f'intensity_{i+1}'] = f_optimized[peak_index]
        results[f'normalized_sum_percent_{i+1}'] = percentage
    return results

#process all dataframes in the dictionary
def nnls_all(dataframes_dict, nnls_params):
    all_results = []
    plot_number = 1
    for name, df in dataframes_dict.items():
        print(f"Processing {name}...")
        results = nnls(df, name, nnls_params, plot_number)
        all_results.append(results)
        plot_number += 1
    nnls_df = pd.DataFrame(all_results)
    return nnls_df

#calculation of decay rates from tau
def calculate_decay_rates(df, tau_columns):
    for tau_col in tau_columns:
        if tau_col in df.columns:
            gamma_col = tau_col.replace('tau', 'gamma')
            df[gamma_col] = 1 / df[tau_col]
        else:
            print(f"Warning: Column '{tau_col}' not found. Skipping.")
    return df

#regularized fit
def nnls_reg(df, name, nnls_reg_params, plot_number):
    decay_times = nnls_reg_params['decay_times']
    prominence = nnls_reg_params.get('prominence', 0.01)  #default to a very low prominence
    distance = nnls_reg_params.get('distance', 1)  #default to minimum distance
    alpha = nnls_reg_params.get('alpha', 0.01) #default, really low alpha
    normalize = nnls_reg_params.get('normalize', False) #default, no normalization
    sparsity_penalty = nnls_reg_params.get('sparsity_penalty', 0.0) #default no penalty
    peak_method = nnls_reg_params.get('peak_method', 'maximum')  #'maximum' or 'centroid'
        
    #the vectors
    tau = df['t [s]'].to_numpy()
    D = df['g(2)-1'].to_numpy()
    
    #create grid of tau and decay time combinations
    decay_times_N, tau_M = np.meshgrid(decay_times, tau)
    
    #matrix A from the mesh - this represents exp(-t/τ) for all combinations
    T = np.exp(-tau_M / decay_times_N)
    
    #create Tikhonov regularization matrix (2nd derivative)
    def create_tikhonov_matrix(n):
        #create a sparse matrix for computational efficiency
        D2 = sparse.diags([1, -2, 1], [-1, 0, 1], shape=(n-2, n)).toarray()
        return D2
    
    #get the length of decay_times
    n = len(decay_times)
    #create the regularization matrix (second derivative)
    D2 = create_tikhonov_matrix(n)

    def residuals_regularized(f, T, D, D2, alpha):
        #apply normalization constraint if enabled
        if normalize:
            #normalize f to sum to 1.0
            f = f / np.sum(f) if np.sum(f) > 0 else f
        
        # Model: g(2)(τ) = (∑_i A_i * exp(-τ/τ_i))^2
        model_output = (T @ f)**2
        
        #data fidelity residuals
        fit_residuals = model_output - D
        
        #initialize residuals list with fit residuals
        all_residuals = [fit_residuals]
        
        #smoothness regularization term
        smoothness_penalty = alpha * (D2 @ f)
        all_residuals.append(smoothness_penalty)
        
        #apply sparsity penalty (L1 norm) if enabled
        if sparsity_penalty > 0:
            l1_penalty = sparsity_penalty * f
            all_residuals.append(l1_penalty)
            
        return np.concatenate(all_residuals)
    
    # optimization approach based on normalization constraint
    if normalize:
        def objective_function(f):
            #make a copy to avoid modifying the original
            f_copy = np.copy(f)
            #ensure non-negativity
            f_copy = np.maximum(f_copy, 0)
            #normalize to sum to 1
            f_copy = f_copy / np.sum(f_copy) if np.sum(f_copy) > 0 else f_copy
            #compute all residuals
            residuals = residuals_regularized(f_copy, T, D, D2, alpha)
            #return sum of squared residuals
            return np.sum(residuals**2)
        
        #initial guess normalized to sum to 1
        f0 = np.ones(T.shape[1]) / T.shape[1]
        
        #constraint: sum of f equals 1
        constraint = {'type': 'eq', 'fun': lambda f: np.sum(f) - 1.0}
        
        #bounds: all elements of f must be non-negative
        bounds = [(0, None) for _ in range(len(f0))]
        
        #perform the optimization
        result = minimize(
            objective_function,
            f0,
            method='SLSQP',  # Sequential Least Squares Programming
            bounds=bounds,
            constraints=constraint,
            options={'ftol': 1e-8, 'disp': False, 'maxiter': 1000}
        )
        f_optimized = result.x
        
        #ensure the result is normalized
        f_optimized = f_optimized / np.sum(f_optimized) if np.sum(f_optimized) > 0 else f_optimized
        
    else:
        #without normalization, use simpler approach
        #initial guess for f - uniform distribution
        f0 = np.ones(T.shape[1])
        
        #perform the non-negative least squares optimization with regularization
        bounds = (0, np.inf)  # Non-negativity constraint
        result = least_squares(
            lambda f: residuals_regularized(f, T, D, D2, alpha), 
            f0, 
            bounds=bounds,
            method='trf',  # Trust Region Reflective algorithm
            ftol=1e-8,     # function tolerance for convergence
            xtol=1e-8      # parameter tolerance for convergence
        )
        f_optimized = result.x
    
    #calculate the optimized function values
    optimized_values = (T @ f_optimized)**2
    residuals_values = optimized_values - D
    
    #find peaks
    peaks, peak_properties = find_peaks(f_optimized, prominence=prominence, distance=distance, width=0)
     
    #get peak amplitudes/intensities
    peak_amplitudes = f_optimized[peaks]
    
    #enhanced peak analysis — centroid, skewness, kurtosis, FWHM, area per peak
    peak_stats = {}
    if len(peaks) > 0:
        for i, peak_idx in enumerate(peaks):
            #get left and right indices from peak_properties
            try:
                #get the width points (half-max)
                left_idx = int(peak_properties['left_ips'][i])
                right_idx = int(peak_properties['right_ips'][i])
                
                #extend to the base of the peak for area calculation
                #threshold is 1% of peak height
                threshold = 0.01 * f_optimized[peak_idx]
                
                #extend left
                left_base_idx = left_idx
                while left_base_idx > 0 and f_optimized[left_base_idx] > threshold:
                    left_base_idx -= 1
                
                #extend right
                right_base_idx = right_idx
                while right_base_idx < len(f_optimized) - 1 and f_optimized[right_base_idx] > threshold:
                    right_base_idx += 1
                
                #make sure indices are within bounds
                left_base_idx = max(0, left_base_idx)
                right_base_idx = min(len(f_optimized) - 1, right_base_idx)
                
            except (IndexError, KeyError):
                # ! Fallback method if peak_properties doesn't have the expected attributes !
                #find local minima around the peak
                left_base_idx = max(0, peak_idx - 1)
                while left_base_idx > 0 and f_optimized[left_base_idx] > f_optimized[left_base_idx - 1]:
                    left_base_idx -= 1
                
                right_base_idx = min(len(f_optimized) - 1, peak_idx + 1)
                while right_base_idx < len(f_optimized) - 1 and f_optimized[right_base_idx] > f_optimized[right_base_idx + 1]:
                    right_base_idx += 1
            
            #extract the peak segment
            peak_segment = f_optimized[left_base_idx:right_base_idx+1]
            peak_x = decay_times[left_base_idx:right_base_idx+1]
            
            #calculate peak area using trapezoidal rule
            peak_area = np.trapz(peak_segment, peak_x)
            
            #calculate peak width at half maximum (FWHM)
            half_max = peak_amplitudes[i] / 2
            above_half_max = peak_segment >= half_max
            if np.sum(above_half_max) > 0:
                idx_above = np.where(above_half_max)[0]
                left_idx = idx_above[0]
                right_idx = idx_above[-1]
                if right_idx > left_idx and right_idx < len(peak_x) and left_idx < len(peak_x):
                    fwhm = peak_x[right_idx] - peak_x[left_idx]
                else:
                    fwhm = 0
            else:
                fwhm = 0
            
            #weighted τ-space moments — f(τ) as probability distribution over τ
            #note: skewness sign is opposite to cumulant skewness (τ vs Γ space)
            w = peak_segment / np.sum(peak_segment) if np.sum(peak_segment) > 0 else np.ones(len(peak_segment)) / len(peak_segment)
            weighted_mean = np.sum(peak_x * w)
            variance      = np.sum(w * (peak_x - weighted_mean)**2)
            std_dev       = np.sqrt(variance) if variance > 0 else 0
            if std_dev > 0 and len(peak_segment) >= 3:
                peak_skewness = np.sum(w * (peak_x - weighted_mean)**3) / std_dev**3
            else:
                peak_skewness = 0
            if std_dev > 0 and len(peak_segment) >= 4:
                peak_kurtosis = np.sum(w * (peak_x - weighted_mean)**4) / std_dev**4 - 3
            else:
                peak_kurtosis = 0

            peak_stats[f'peak_{i+1}'] = {
                'position'  : decay_times[peak_idx],
                'amplitude' : peak_amplitudes[i],
                'area'      : peak_area,
                'fwhm'      : fwhm,
                'centroid'  : weighted_mean,
                'std_dev'   : std_dev,
                'skewness'  : peak_skewness,   # τ-space: positive = tail toward larger particles
                'kurtosis'  : peak_kurtosis,   # excess kurtosis: >0 = sharp, <0 = broad
                'left_idx'  : left_base_idx,
                'right_idx' : right_base_idx
            }
    
    #area-based normalization over detected peaks only
    if len(peaks) > 0 and peak_stats:
        total_detected_area = np.sum([peak_stats[f'peak_{i+1}']['area'] for i in range(len(peaks))])
        if total_detected_area > 0:
            normalized_area_percent = [
                peak_stats[f'peak_{i+1}']['area'] / total_detected_area * 100
                for i in range(len(peaks))
            ]
        else:
            normalized_area_percent = [100.0 / len(peaks)] * len(peaks)
    else:
        normalized_area_percent = []

    #R²
    ss_res = np.sum(residuals_values**2)
    ss_tot = np.sum((D - np.mean(D))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    #4-panel layout: data+fit | distribution | residuals | Q-Q
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 6))
    fig.suptitle(f'[{plot_number}]: Regularized — {name}', fontsize=12)

    # Panel 1: data + fit
    ax1.semilogx(tau, D, 'o', alpha=0.6, markersize=4, label='Data')
    ax1.semilogx(tau, optimized_values, 'r-', linewidth=2, label='Fit')
    ax1.set_xlabel(r'lag time τ [s]')
    ax1.set_ylabel(r'$g^{(2)}(τ) - 1$')
    ax1.set_title('Data & Fit')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.text(0.95, 0.95, f'R² = {r_squared:.4f}',
             transform=ax1.transAxes, va='top', ha='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Panel 2: tau distribution with colored peak areas
    ax2.semilogx(decay_times, f_optimized, 'bo-', markersize=3, label='f(τ)')
    colors = ['red', 'green', 'orange', 'purple', 'cyan', 'magenta']
    if len(peaks) > 0:
        for i, peak_idx in enumerate(peaks):
            color = colors[i % len(colors)]
            peak_info = peak_stats[f'peak_{i+1}']
            left_idx = peak_info['left_idx']
            right_idx = peak_info['right_idx']
            x_fill = decay_times[left_idx:right_idx+1]
            y_fill = f_optimized[left_idx:right_idx+1]
            ax2.fill_between(x_fill, 0, y_fill, alpha=0.3, color=color, label=f'Area {i+1}')
            tau_display = peak_info['centroid'] if peak_method == 'centroid' else decay_times[peak_idx]
            ax2.plot(tau_display, f_optimized[peak_idx], 'x', color=color, markersize=10)
            annotation_text = (f'τ={tau_display:.2e}\n'
                               f'Area%={normalized_area_percent[i]:.1f}%\n'
                               f'FWHM={peak_info["fwhm"]:.2e}\n'
                               f'Skew={peak_info["skewness"]:.2f}\n'
                               f'Kurt={peak_info["kurtosis"]:.2f}')
            ax2.annotate(annotation_text,
                        xy=(tau_display, f_optimized[peak_idx]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, color='black', backgroundcolor='white',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    ax2.set_xlabel(r'decay time τ [s]')
    ax2.set_ylabel('f(τ) [a.u.]')
    ax2.set_title('τ Distribution')
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, which="both", ls="--", alpha=0.3)

    # Panel 3: residuals
    ax3.plot(residuals_values)
    ax3.axhline(0, color='r', linestyle='--', linewidth=1)
    ax3.set_xlabel('Sample Index')
    ax3.set_ylabel('Residuals')
    ax3.set_title('Residuals')
    ax3.grid(True, alpha=0.3)

    # Panel 4: Q-Q
    stats.probplot(residuals_values, dist="norm", plot=ax4)
    ax4.set_title('Q-Q Plot')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    
    #prepare results for this dataframe
    results = {'filename': name, 'R_squared': r_squared}
    for i, peak_index in enumerate(peaks):
        #tau: maximum or centroid depending on peak_method
        if peak_method == 'centroid' and f'peak_{i+1}' in peak_stats:
            results[f'tau_{i+1}'] = peak_stats[f'peak_{i+1}']['centroid']
        else:
            results[f'tau_{i+1}'] = decay_times[peak_index]
        results[f'intensity_{i+1}']            = f_optimized[peak_index]
        results[f'normalized_area_percent_{i+1}'] = normalized_area_percent[i] if normalized_area_percent else np.nan

        #per-peak statistics (always computed)
        if f'peak_{i+1}' in peak_stats:
            peak_info = peak_stats[f'peak_{i+1}']
            results[f'area_{i+1}']     = peak_info['area']
            results[f'fwhm_{i+1}']     = peak_info['fwhm']
            results[f'centroid_{i+1}'] = peak_info['centroid']
            results[f'std_dev_{i+1}']  = peak_info['std_dev']
            results[f'skewness_{i+1}'] = peak_info['skewness']
            results[f'kurtosis_{i+1}'] = peak_info['kurtosis']   

    return results, f_optimized, optimized_values, residuals_values, peaks, peak_stats
    
def nnls_reg_all(dataframes_dict, nnls_reg_params):
    all_results = []
    plot_number = 1
    full_results = {}
    
    for name, df in dataframes_dict.items():
        print(f"Processing {name}...")
        try:
            results, f_optimized, optimized_values, residuals_values, peaks, peak_stats = nnls_reg(
                df, name, nnls_reg_params, plot_number)
            
            all_results.append(results)
            full_results[name] = {
                'f_optimized': f_optimized,
                'optimized_values': optimized_values,
                'residuals_values': residuals_values,
                'peaks': peaks,
                'peak_stats': peak_stats
            }
            plot_number += 1
                
        except Exception as e:
            print(f"Error processing {name}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    #summary DataFrame
    nnls_reg_df = pd.DataFrame(all_results) if all_results else pd.DataFrame()
    
    return nnls_reg_df, full_results

#simple variant of nnls_reg for alpha-comparison
def nnls_reg_simple(df, name, nnls_reg_params):
    decay_times = nnls_reg_params['decay_times']
    prominence = nnls_reg_params.get('prominence', 0.01)
    distance = nnls_reg_params.get('distance', 1)
    alpha = nnls_reg_params.get('alpha', 0.01)
    
    #create the vectors
    tau = df['t [s]'].to_numpy()
    D = df['g(2)-1'].to_numpy()
    
    #create grid of tau and decay time combinations
    decay_times_N, tau_M = np.meshgrid(decay_times, tau)
    
    #create matrix A from the mesh
    T = np.exp(-tau_M / decay_times_N)
    
    #create Tikhonov regularization matrix (second derivative)
    def create_tikhonov_matrix(n):
        D2 = sparse.diags([1, -2, 1], [-1, 0, 1], shape=(n-2, n)).toarray()
        return D2
    
    #get the length of decay_times
    n = len(decay_times)
    #create the regularization matrix
    D2 = create_tikhonov_matrix(n)
    
    #define the regularized objective function
    def residuals_regularized(f, T, D, D2, alpha):
        #g(2)(τ)-1 = (∑_i A_i * exp(-τ/τ_i))^2
        model_output = (T @ f)**2
        
        #data fidelity residuals
        fit_residuals = model_output - D
        
        #smoothness
        smoothness_penalty = alpha * (D2 @ f)
        return np.concatenate([fit_residuals, smoothness_penalty])
    
    #initial guess (uniform distribution)
    f0 = np.ones(T.shape[1])
    
    #perform nnls
    bounds = (0, np.inf)  # Non-negativity Constraint !
    result = least_squares(
        lambda f: residuals_regularized(f, T, D, D2, alpha), 
        f0, 
        bounds=bounds,
        method='trf',
        ftol=1e-8,
        xtol=1e-8
    )
    f_optimized = result.x
    
    #calculate the optimized function
    optimized_values = (T @ f_optimized)**2
    residuals_values = optimized_values - D
    
    #find peaks
    peaks, _ = find_peaks(f_optimized, prominence=prominence, distance=distance, width=0)
    
    #prepare results for this dataframe
    results = {'filename': name}
    for i, peak_index in enumerate(peaks):
        results[f'tau_{i+1}'] = decay_times[peak_index]
        results[f'intensity_{i+1}'] = f_optimized[peak_index]
    
    return results, f_optimized, optimized_values, residuals_values, peaks

#comparing alpha-values
def analyze_random_datasets_grid(df_dict, num_datasets, base_nnls_params, nnls_reg_simple_function, alpha_range=(0.01, 1), num_alphas=5, seed=None, figsize=None):
    #set random seed
    if seed is not None:
        random.seed(seed)
    
    #get all keys from the dictionary
    all_keys = list(df_dict.keys())
    
    #ensure we don't try to select more datasets than available
    num_to_select = min(num_datasets, len(all_keys))
    
    #randomly select keys
    chosen_keys = random.sample(all_keys, num_to_select)
    print(f"Selected datasets: {', '.join(chosen_keys)}")
    
    #determine grid dimensions
    cols = min(3, num_to_select)  # Max 3 columns
    rows = (num_to_select + cols - 1) // cols  # Ceiling division
    
    #calculate figure size if not provided
    if figsize is None:
        figsize = (6*cols, 5*rows)
    
    #create figure
    fig = plt.figure(figsize=figsize)
    
    #get shared parameters
    alpha_min, alpha_max = alpha_range
    alpha_values = np.logspace(np.log10(alpha_min), np.log10(alpha_max), num_alphas)
    decay_times = base_nnls_params['decay_times']
    
    #color map for different alpha values
    colors = plt.cm.viridis(np.linspace(0, 1, num_alphas))
    
    #dictionary to store peak statistics
    peak_stats = {}
    
    #plots for each dataset
    for i, key in enumerate(chosen_keys):
        ax = fig.add_subplot(rows, cols, i+1, projection='3d')
        
        df = df_dict[key]
        print(f"Analyzing dataset: {key}")
        peak_stats[key] = {}
        
        #compute and plot distributions for each alpha
        for j, alpha in enumerate(alpha_values):
            current_params = base_nnls_params.copy()
            current_params['alpha'] = alpha
            
            _, f_optimized, _, _, _ = nnls_reg_simple_function(df, key, current_params)
            #get peaks
            _, _, _, _, peaks = nnls_reg_simple_function(df, key, current_params)
            
            #store peak statistics
            peak_decay_times = decay_times[peaks]
            peak_stats[key][alpha] = {
                'num_peaks': len(peaks),
                'peak_decay_times': peak_decay_times
            }
            
            #print peak statistics
            print(f"  Alpha: {alpha:.3e}, Number of peaks: {len(peaks)}")
            if len(peaks) > 0:
                print(f"  Peak decay times: {', '.join([f'{t:.3e}' for t in peak_decay_times])}")
            
            #plot the distribution
            ax.plot(np.log10(decay_times), 
                    np.ones_like(decay_times) * np.log10(alpha), 
                    f_optimized, 
                    color=colors[j], 
                    linewidth=1.5)
            
            #highlight peaks
            ax.scatter(np.log10(decay_times[peaks]), 
                      np.ones_like(peaks) * np.log10(alpha),
                      f_optimized[peaks],
                      color='red', s=30, marker='o')
        
        #set labels and title
        ax.set_xlabel('log10(Decay Time [s])', fontsize=9)
        ax.set_ylabel('log10(α)', fontsize=9)
        ax.set_zlabel('Amplitude', fontsize=9)
        ax.set_title(f'{key}', fontsize=11)
        ax.tick_params(axis='both', which='major', labelsize=8)
        
        #view angle
        ax.view_init(elev=30, azim=45)
    
    #legend for all subplots
    legend_labels = [f'α = {alpha:.3f}' for alpha in alpha_values]
    legend_handles = [plt.Line2D([0], [0], color=colors[i]) for i in range(len(alpha_values))]
    #add marker for peaks in legend
    legend_handles.append(plt.Line2D([0], [0], marker='o', color='red', linestyle='None', 
                                    markersize=5, label='Peaks'))
    legend_labels.append('Peaks')
    
    fig.legend(legend_handles, legend_labels, loc='upper center', 
               bbox_to_anchor=(0.5, 0.98), ncol=min(5, num_alphas+1), fontsize=10)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust to make room for the legend
    return fig, chosen_keys

#following section is for comparing the results
#calculate Rh for plot
def tau_to_hydrodynamic_radius(tau, q, temperature, viscosity):
    kB = 1.380649e-23  # Boltzmann constant (J/K)
    q_m = q * 1e9  # nm^-1 to m^-1
    D = 1 / (tau * q_m**2)  # m^2/s
    rh_nm = kB * temperature / (6 * np.pi * viscosity * 1e-3 * D) *10**9  #viscosity conversion cp to Pa·s
    return rh_nm

#dataset matching
def find_dataset_key(filename, full_results):
    filename_base = filename.replace('.ASC', '').replace('.asc', '')

    for key in full_results.keys():
        if filename_base == key or filename == key:
            return key
        if filename_base in key or key in filename_base:
            return key
    
    return None

#plot distributions for comparison
def plot_distributions(full_results, nnls_reg_params, df_basedata_mod, 
                      angles=None, measurement_mode='first',
                      convert_to_radius=True, xlim=None, 
                      figsize=(12, 8), title="Distribution Comparison", filenames=None):
    decay_times = np.array(nnls_reg_params['decay_times'])
    
    #direct filename mode: bypass angle/measurement_mode logic
    if filenames is not None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        colors = cm.tab10(np.linspace(0, 1, max(len(filenames), 1)))
        for i, fname in enumerate(filenames):
            dataset_key = find_dataset_key(fname, full_results)
            if dataset_key is None or dataset_key not in full_results:
                print(f"No results found for '{fname}', skipping.")
                continue
            rows = df_basedata_mod[df_basedata_mod['filename'] == fname]
            if rows.empty:
                print(f"No metadata found for '{fname}', skipping.")
                continue
            row = rows.iloc[0]
            f_opt = full_results[dataset_key]['f_optimized']
            if convert_to_radius:
                x_vals = tau_to_hydrodynamic_radius(decay_times, row['q'],
                                                    row['temperature [K]'],
                                                    row['viscosity [cp]'])
                x_label = 'R$_h$ [nm]'
            else:
                x_vals = decay_times
                x_label = 'Decay Time (s)'
            ax.semilogx(x_vals, f_opt, 'o-', color=colors[i],
                        label=fname, linewidth=2, markersize=4, alpha=0.8)
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel('Intensity [a.u.]', fontsize=12)
        ax.set_title(title, fontsize=14)
        if xlim is not None:
            ax.set_xlim(*xlim)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        plt.tight_layout()
        plt.show()
        return fig, ax
    
    #select angles
    if angles is None:
        angles = sorted(df_basedata_mod['angle [°]'].unique())
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    colors = cm.tab10(np.linspace(0, 1, len(angles)))
    
    plot_count = 0
    
    for angle in angles:
        angle_data = df_basedata_mod[df_basedata_mod['angle [°]'] == angle]
        
        if len(angle_data) == 0:
            print(f"No data for angle {angle}°")
            continue
        
        if measurement_mode == 'average':
            #average all measurements at this angle
            distributions = []
            ref_params = None
            
            for _, row in angle_data.iterrows():
                dataset_key = find_dataset_key(row['filename'], full_results)
                if dataset_key and dataset_key in full_results:
                    f_opt = full_results[dataset_key]['f_optimized']
                    distributions.append(f_opt)
                    if ref_params is None:
                        ref_params = row
            
            if len(distributions) == 0:
                print(f"No valid results for angle {angle}°")
                continue
            
            #average the distributions
            f_optimized = np.mean(distributions, axis=0)
            label = f"{angle}° avg (n={len(distributions)})"
            params = ref_params
            
        elif measurement_mode == 'first':
            #take first measurement at this angle
            params = angle_data.iloc[0]
            dataset_key = find_dataset_key(params['filename'], full_results)
            
            if dataset_key is None or dataset_key not in full_results:
                print(f"No results for {params['filename']}")
                continue
            
            f_optimized = full_results[dataset_key]['f_optimized']
            label = f"{angle}°"
            
        elif measurement_mode == 'all':
            #plot all measurements at this angle
            for i, (_, row) in enumerate(angle_data.iterrows()):
                dataset_key = find_dataset_key(row['filename'], full_results)
                
                if dataset_key is None or dataset_key not in full_results:
                    continue
                
                f_opt = full_results[dataset_key]['f_optimized']
                
                #process this individual measurement
                
                if convert_to_radius:
                    x_vals = tau_to_hydrodynamic_radius(decay_times, row['q'], 
                                                       row['temperature [K]'], row['viscosity [cp]'])
                    x_label = 'R$_h$ [nm]'
                else:
                    x_vals = decay_times
                    x_label = 'Decay Time (s)'
                
                color = colors[plot_count % len(colors)]
                alpha = 0.6 if len(angle_data) > 1 else 0.8
                label_individual = f"{angle}° #{i+1}"
                
                ax.semilogx(x_vals, f_opt, 'o-', color=color, alpha=alpha,
                           label=label_individual, linewidth=2, markersize=3)
            
            plot_count += 1
            continue  #skip the common plotting code below
        
        #common processing for 'average' and 'first' modes
        if convert_to_radius:
            q_value = params.get('q', params.get('q^2', None))
            if q_value is None:
                print(f"Warning: No q-value found for angle {angle}°")
                continue
            
            x_values = tau_to_hydrodynamic_radius(decay_times, q_value, 
                                                 params['temperature [K]'], 
                                                 params['viscosity [cp]'])
            x_label = 'R$_h$ [nm]'
        else:
            x_values = decay_times
            x_label = 'Decay Time (s)'
        
        #plot
        color = colors[plot_count]
        ax.semilogx(x_values, f_optimized, 'o-', color=color, 
                   label=label, linewidth=2, markersize=4, alpha=0.8)
        
        plot_count += 1
    
    #finalize plot
    ax.set_xlabel(x_label, fontsize=12)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.set_ylabel('Intensity [a.u.]', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    #plt.yticks([]) 
    plt.tight_layout()
    plt.show()
    
    return fig, ax