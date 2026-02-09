# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 20:50:20 2025

@author: vinci

changelog 1.1:
    - added nnls_reg_simple, analyze_random_datasets_grid
    - changes to nnls_reg and nnls_reg_all
    
definitions for regularized analysis in DLS-Supreme
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


def calculate_peak_centroid(decay_times, distribution, peak_index, peak_properties=None, width_factor=3.0):
    """
    Calculate the centroid (center of mass) of a peak in log-space

    Args:
        decay_times: Array of decay time values (x-axis)
        distribution: Array of distribution values (y-axis, intensities)
        peak_index: Index of the peak maximum
        peak_properties: Optional dict from find_peaks with 'widths' info
        width_factor: How many widths around peak to include (default: 3.0 for ~full peak)

    Returns:
        Centroid tau value
    """
    # Determine the range around the peak to consider
    n_points = len(decay_times)

    if peak_properties and 'widths' in peak_properties and len(peak_properties['widths']) > 0:
        # Use peak width from find_peaks
        width_idx = np.where(peak_properties == peak_index)[0]
        if len(width_idx) > 0:
            width = peak_properties['widths'][width_idx[0]]
            half_width = int(width * width_factor / 2)
        else:
            # Fallback: use 10% of total range
            half_width = max(1, n_points // 10)
    else:
        # Fallback: use 10% of total range
        half_width = max(1, n_points // 10)

    # Define range around peak
    left = max(0, peak_index - half_width)
    right = min(n_points, peak_index + half_width + 1)

    # Extract local region
    local_times = decay_times[left:right]
    local_dist = distribution[left:right]

    # Calculate centroid in log-space (since decay_times are log-spaced)
    # Centroid = sum(x * f(x)) / sum(f(x))
    log_times = np.log10(local_times)
    total_intensity = np.sum(local_dist)

    if total_intensity > 0:
        log_centroid = np.sum(log_times * local_dist) / total_intensity
        centroid = 10 ** log_centroid
    else:
        # Fallback to maximum if something goes wrong
        centroid = decay_times[peak_index]

    return centroid


#code for simple NNLS-Fit to the data
def nnls(df, name, nnls_params, plot_number):
    decay_times = nnls_params['decay_times']
    prominence = nnls_params['prominence']
    distance = nnls_params['distance']
    
    #create the vectors
    tau = df['t (s)'].to_numpy()
    D = df['g(2)'].to_numpy()
    
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
    
    #find peaks in the tau distribution (with width info for centroid calculation)
    peaks, peak_properties = find_peaks(f_optimized, prominence=prominence, distance=distance, width=0)

    #get peak amplitudes/intensities
    peak_amplitudes = f_optimized[peaks]

    #normalization to the total sum
    normalized_amplitudes_sum = peak_amplitudes / np.sum(peak_amplitudes)

    # Check if centroid mode is enabled
    use_centroid = nnls_params.get('use_centroid', False)
    
    #create plot for this dataframe
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'[{plot_number}]: Analysis for {name}', fontsize=16)
    
    # First subplot: Data and optimized function
    ax1.semilogx(tau, D, 'ro', label='Data (D)')
    ax1.semilogx(tau, optimized_values, 'g-', label='Optimized Function (f_optimized)')
    ax1.set_xlabel('lag time [s]')
    ax1.set_ylabel('g(2)-1')
    ax1.set_title('Optimization Results')
    ax1.legend()
    ax1.grid(True)
    
    # Second subplot: Tau distribution
    ax2.semilogx(decay_times, f_optimized, 'bo-', label='Tau Distribution (f_optimized)')
    ax2.set_xlabel('Tau (Decay Times)')
    ax2.set_ylabel('Intensity (f_optimized)')
    ax2.set_title('Tau Distribution')
    ax2.legend()
    ax2.grid(True, which="both", ls="--")
    
    # Third subplot: Residuals
    ax3.plot(residuals_values)
    ax3.set_xlabel('Sample Index')
    ax3.set_ylabel('Residual Value')
    ax3.set_title('Residuals')
    ax3.axhline(0, color='r', linestyle='--')
    ax3.legend(['Residuals'])
    ax3.grid(True)
    
    #mark peaks in the tau distribution plot
    ax2.plot(decay_times[peaks], f_optimized[peaks], 'rx', markersize=10, label='Peaks')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    #prepare results for this dataframe
    results = {'filename': name}
    for i, peak_index in enumerate(peaks):
        percentage = normalized_amplitudes_sum[i] * 100

        # Calculate tau value: use centroid or maximum
        if use_centroid:
            tau_value = calculate_peak_centroid(decay_times, f_optimized, peak_index, peak_properties)
        else:
            tau_value = decay_times[peak_index]

        results[f'tau_{i+1}'] = tau_value
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
        if tau_col in df.columns:  # Check if the tau column exists
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
    enforce_unimodality = nnls_reg_params.get('enforce_unimodality', False) #default, not enforced
        
    #the vectors
    tau = df['t (s)'].to_numpy()
    D = df['g(2)'].to_numpy()
    
    #create grid of tau and decay time combinations
    decay_times_N, tau_M = np.meshgrid(decay_times, tau)
    
    #matrix A from the mesh - this represents exp(-t/τ) for all combinations
    T = np.exp(-tau_M / decay_times_N)
    
    # Create Tikhonov regularization matrix (2nd derivative)
    def create_tikhonov_matrix(n):
        #create a sparse matrix for computational efficiency
        D2 = sparse.diags([1, -2, 1], [-1, 0, 1], shape=(n-2, n)).toarray()
        return D2
    
    #unimodality constraint matrix
    def create_unimodality_matrix(n):
        # Create a matrix with 1's on the diagonal and -1's on the sub-diagonal
        # This computes first differences
        D1 = np.zeros((n-2, n))
        for i in range(n-2):
            D1[i, i] = 1
            D1[i, i+1] = -2
            D1[i, i+2] = 1
        return D1
    
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
            
        #apply unimodality constraint if enabled
        if enforce_unimodality:
            if not hasattr(residuals_regularized, 'unimodality_matrix'):
                residuals_regularized.unimodality_matrix = create_unimodality_matrix(n)
            
            #compute second differences and penalize sign changes beyond the first one
            second_diff = residuals_regularized.unimodality_matrix @ f
            #only penalize if there are multiple sign changes
            sign_changes = np.where(np.diff(np.signbit(np.diff(f))))[0]
            if len(sign_changes) > 1:
                unimodality_penalty = 10.0 * second_diff
                all_residuals.append(unimodality_penalty)
        
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

    #calculate normalized amplitudes
    normalized_amplitudes_sum = peak_amplitudes / np.sum(peak_amplitudes) if np.sum(peak_amplitudes) > 0 else np.zeros_like(peak_amplitudes)

    # Check if centroid mode is enabled
    use_centroid = nnls_reg_params.get('use_centroid', False)
    
    #enhanced peak analysis for normalized distributions
    peak_stats = {}
    if normalize and len(peaks) > 0:
        #calculate peak areas, skewness, and other statistics
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
            
            #calculate peak skewness
            # ! at least 3 points to calculate skewness required !
            if len(peak_segment) >= 3:
                # Use moments to calculate skewness
                peak_skewness = stats.skew(peak_segment, bias=False)
            else:
                peak_skewness = 0
                
            #calculate peak kurtosis
            if len(peak_segment) >= 4:
                peak_kurtosis = stats.kurtosis(peak_segment, fisher=True, bias=False)
            else:
                peak_kurtosis = 0
                
            #calculate mean (centroid) and standard deviation of peak
            #weight by y values (intensity)
            weighted_mean = np.sum(peak_x * peak_segment) / np.sum(peak_segment) if np.sum(peak_segment) > 0 else 0
            variance = np.sum(peak_segment * (peak_x - weighted_mean)**2) / np.sum(peak_segment) if np.sum(peak_segment) > 0 else 0
            std_dev = np.sqrt(variance) if variance > 0 else 0
            
            #statistics for this peak
            peak_stats[f'peak_{i+1}'] = {
                'position': decay_times[peak_idx],
                'amplitude': peak_amplitudes[i],
                'normalized_amplitude': normalized_amplitudes_sum[i],
                'area': peak_area,
                'fwhm': fwhm,
                'skewness': peak_skewness,
                'kurtosis': peak_kurtosis,
                'centroid': weighted_mean,
                'std_dev': std_dev,
                'left_idx': left_base_idx,
                'right_idx': right_base_idx
            }
    
    #create plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'[{plot_number}]: Analysis for {name}', fontsize=16)
    
    # First subplot: Data and optimized function
    ax1.semilogx(tau, D, 'ro', label='Data (D)')
    ax1.semilogx(tau, optimized_values, 'g-', label='Optimized Function (f_optimized)')
    ax1.set_xlabel('lag time [s]')
    ax1.set_ylabel('g(2)-1')
    ax1.set_title('Optimization Results')
    ax1.legend()
    ax1.grid(True)
    
    # Second subplot: Tau distribution with peak markers and colored areas
    ax2.semilogx(decay_times, f_optimized, 'bo-', label='Tau Distribution (f_optimized)')
    
    #colors for peak areas
    colors = ['red', 'green', 'orange', 'purple', 'cyan', 'magenta']
    
    if normalize and len(peaks) > 0:
        #colored peak areas
        for i, peak_idx in enumerate(peaks):
            color = colors[i % len(colors)]
            
            #get peak information from peak_stats
            peak_info = peak_stats[f'peak_{i+1}']
            left_idx = peak_info['left_idx']
            right_idx = peak_info['right_idx']
            
            #fill the peak area
            x_fill = decay_times[left_idx:right_idx+1]
            y_fill = f_optimized[left_idx:right_idx+1]
            ax2.fill_between(x_fill, 0, y_fill, alpha=0.3, color=color, label=f'Area {i+1}')
            
            #mark peak position
            ax2.plot(decay_times[peak_idx], f_optimized[peak_idx], 'x', color=color, markersize=10)
            
            #create annotation text with peak info
            annotation_text = (f'τ={decay_times[peak_idx]:.2e}\n'
                              f'I={f_optimized[peak_idx]:.2e}\n'
                              f'Norm={normalized_amplitudes_sum[i]:.2f}\n'
                              f'Area={peak_info["area"]:.2e}\n'
                              f'FWHM={peak_info["fwhm"]:.2e}\n'
                              f'Skew={peak_info["skewness"]:.2f}')
            
            #annotation with peak info
            ax2.annotate(annotation_text,
                        xy=(decay_times[peak_idx], f_optimized[peak_idx]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, color='black', backgroundcolor='white',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    else:
        #original peak marking without coloring areas
        for i, peak_idx in enumerate(peaks):
            color = colors[i % len(colors)]
            #mark peak position
            ax2.plot(decay_times[peak_idx], f_optimized[peak_idx], 'x', color=color, markersize=10)
            
            #text annotation with peak info
            ax2.annotate(f'τ={decay_times[peak_idx]:.2e}\nI={f_optimized[peak_idx]:.2e}\nNorm={normalized_amplitudes_sum[i]:.2f}',
                        xy=(decay_times[peak_idx], f_optimized[peak_idx]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, color='black', backgroundcolor='white',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    ax2.set_xlabel('Tau (Decay Times)')
    ax2.set_ylabel('Intensity (f_optimized)')
    ax2.set_title('Tau Distribution')
    ax2.grid(True, which="both", ls="--")
    
    # Third subplot: Residuals
    ax3.plot(residuals_values)
    ax3.set_xlabel('Sample Index')
    ax3.set_ylabel('Residual Value')
    ax3.set_title('Residuals')
    ax3.axhline(0, color='r', linestyle='--')
    ax3.grid(True)
    
    #legend
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(loc='best', fontsize=8)
    
    plt.tight_layout()
        
    plt.show()
    
    #prepare results for this dataframe
    results = {'filename': name}
    for i, peak_index in enumerate(peaks):
        #basic peak metrics
        percentage = normalized_amplitudes_sum[i] * 100

        # Calculate tau value: use centroid or maximum
        if use_centroid:
            tau_value = calculate_peak_centroid(decay_times, f_optimized, peak_index, peak_properties)
        else:
            tau_value = decay_times[peak_index]

        results[f'tau_{i+1}'] = tau_value
        results[f'intensity_{i+1}'] = f_optimized[peak_index]
        results[f'normalized_sum_percent_{i+1}'] = percentage

        #additional peak metrics if normalize is True
        if normalize and f'peak_{i+1}' in peak_stats:
            peak_info = peak_stats[f'peak_{i+1}']
            results[f'area_{i+1}'] = peak_info['area']
            results[f'fwhm_{i+1}'] = peak_info['fwhm']
            results[f'skewness_{i+1}'] = peak_info['skewness']
            results[f'kurtosis_{i+1}'] = peak_info['kurtosis']
            results[f'centroid_{i+1}'] = peak_info['centroid']
            results[f'std_dev_{i+1}'] = peak_info['std_dev']
    
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
    tau = df['t (s)'].to_numpy()
    D = df['g(2)'].to_numpy()
    
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
    peaks, peak_properties = find_peaks(f_optimized, prominence=prominence, distance=distance, width=0)

    # Check if centroid mode is enabled
    use_centroid = nnls_reg_params.get('use_centroid', False)

    #prepare results for this dataframe
    results = {'filename': name}
    for i, peak_index in enumerate(peaks):
        # Calculate tau value: use centroid or maximum
        if use_centroid:
            tau_value = calculate_peak_centroid(decay_times, f_optimized, peak_index, peak_properties)
        else:
            tau_value = decay_times[peak_index]

        results[f'tau_{i+1}'] = tau_value
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
                      convert_to_radius=True, 
                      figsize=(12, 8), title="Distribution Comparison"):
    decay_times = np.array(nnls_reg_params['decay_times'])
    
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
    ax.set_ylabel('Intensity [a.u.]', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    #plt.yticks([]) 
    plt.tight_layout()
    plt.show()
    

    return fig, ax
