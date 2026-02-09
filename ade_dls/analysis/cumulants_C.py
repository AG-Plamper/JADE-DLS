# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 11:41:12 2025

@author: vinci

cumulant analysis: method C
"""
import numpy as np
from scipy.optimize import curve_fit
import inspect
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

#get the meaningful initial parameters for cumulant fit method C
def get_meaningful_parameters(fit_function, full_parameter_list):
    #extract individual parameters
    a, b, c, d, e, f = full_parameter_list
    
    func_name = fit_function.__name__
    
    if func_name == 'fit_function2':
        return [a, b, c, f]
    elif func_name == 'fit_function3':  
        return [a, b, c, d, f]
    elif func_name == 'fit_function4':
        return [a, b, c, d, e, f]
    else:
        raise ValueError(f"Unknown fit function: {func_name}")

#estimation of inital parameter for cumulant fit method C
def estimate_parameters_from_data(x_data, y_data, base_parameters):
    try:
        y_max = np.max(y_data)
        y_min = np.min(y_data)
        y_range = y_max - y_min
        
        #amplitude estimate (with safety check)
        amplitude_estimate = max(0.1, y_range)
        
        #baseline estimate
        baseline_estimate = y_min
        
        #decay rate estimation
        #1: Find where signal drops to ~37% (1/e)
        threshold = y_min + 0.37 * y_range
        decay_indices = np.where(y_data <= threshold)[0]
        
        if len(decay_indices) > 0 and decay_indices[0] < len(x_data):
            decay_time = x_data[decay_indices[0]]
            if decay_time > 0:
                decay_rate_estimate = 1.0 / decay_time
            else:
                decay_rate_estimate = base_parameters[1]  # Fall back to base
        else:
            #2: Use characteristic time from log-linear region
            try:
                #find a reasonable range for linear fit in log space
                log_y = np.log(np.maximum(y_data - baseline_estimate, 1e-10))
                mid_idx = len(x_data) // 4  #use first quarter of data
                end_idx = len(x_data) // 2
                
                if end_idx > mid_idx + 3:  #ensure enough points
                    x_fit_region = x_data[mid_idx:end_idx]
                    log_y_fit_region = log_y[mid_idx:end_idx]
                    
                    #linear fit in log space
                    coeffs = np.polyfit(x_fit_region, log_y_fit_region, 1)
                    decay_rate_estimate = abs(coeffs[0])  #slope magnitude
                else:
                    decay_rate_estimate = base_parameters[1]
            except:
                decay_rate_estimate = base_parameters[1]
        
        #for monodisperse systems, c ≈ 0, for polydisperse systems, c can be positive (typical range: 0 to 0.5*b²)
        c_estimate = 0.1 * decay_rate_estimate**2  #start with small polydispersity
        
        # d (3rd cumulant) - related to asymmetry of size distribution
        #typical range: -0.1*b³ to +0.1*b³
        d_estimate = 0.01 * decay_rate_estimate**3  #small positive asymmetry
        
        # e (4th cumulant) - related to kurtosis of size distribution
        #for Gaussian distributions: e = 3*c²
        e_estimate = 3 * c_estimate**2
        
        #alternative approach: estimate from curvature in early decay
        try:
            #look at curvature in the first 20% of data for better c estimate
            early_idx = int(0.2 * len(x_data))
            if early_idx > 5:
                x_early = x_data[:early_idx]
                y_early = y_data[:early_idx]
                
                #fit a simple exponential to get deviation
                simple_exp = amplitude_estimate * np.exp(-decay_rate_estimate * x_early) + baseline_estimate
                residual_early = y_early - simple_exp
                
                #if there's systematic curvature, adjust c
                if len(residual_early) > 3:
                    #positive curvature suggests polydispersity
                    mean_residual = np.mean(residual_early[1:-1])  #exclude endpoints
                    if abs(mean_residual) > 0.01 * amplitude_estimate:
                        c_estimate = np.clip(mean_residual / amplitude_estimate * decay_rate_estimate**2, 
                                           0, 0.5 * decay_rate_estimate**2)
                        e_estimate = 3 * c_estimate**2
        except:
            pass  #keep original estimates if curvature analysis fails
        
        #apply reasonable bounds to all estimates
        decay_rate_estimate = np.clip(decay_rate_estimate, 1, 1e6)
        amplitude_estimate = np.clip(amplitude_estimate, 0.01, 10)
        baseline_estimate = np.clip(baseline_estimate, -1, 1)
        
        #bounds for cumulant parameters
        c_estimate = np.clip(c_estimate, 0, decay_rate_estimate**2)  #non-negative, physical limit
        d_estimate = np.clip(d_estimate, -0.1 * decay_rate_estimate**3, 0.1 * decay_rate_estimate**3)
        e_estimate = np.clip(e_estimate, 0, 10 * c_estimate**2)  #non-negative, reasonable upper bound
        
        return {
            'a': amplitude_estimate,
            'b': decay_rate_estimate,
            'c': c_estimate,
            'd': d_estimate,
            'e': e_estimate,
            'f': baseline_estimate
        }
        
    except Exception as e:
        print(f"Warning: Parameter estimation failed ({e}), using base parameters")
        return {
            'a': base_parameters[0] if len(base_parameters) > 0 else 1.0,
            'b': base_parameters[1] if len(base_parameters) > 1 else 1000.0,
            'c': base_parameters[2] if len(base_parameters) > 2 else 100.0,
            'd': base_parameters[3] if len(base_parameters) > 3 else 1.0,
            'e': base_parameters[4] if len(base_parameters) > 4 else 30000.0,
            'f': base_parameters[5] if len(base_parameters) > 5 else 0.0
        }

#different approaches for adaptive fitting
def get_adaptive_parameters_strategy(dataframes_dict, fit_function, base_parameters, 
                                   strategy='individual', x_col='t (s)', y_col='g(2)'):
    
    if strategy == 'individual':
        return _individual_strategy(dataframes_dict, fit_function, base_parameters, x_col, y_col)
    elif strategy == 'global':
        return _global_strategy(dataframes_dict, fit_function, base_parameters, x_col, y_col)
    elif strategy == 'representative':
        return _representative_strategy(dataframes_dict, fit_function, base_parameters, x_col, y_col)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

def _individual_strategy(dataframes_dict, fit_function, base_parameters, x_col, y_col):
    adaptive_params = {}
    
    for name, df in dataframes_dict.items():
        try:
            x_data = df[x_col].values
            y_data = df[y_col].values
            
            #get adaptive estimates
            estimates = estimate_parameters_from_data(x_data, y_data, base_parameters)
            
            #apply estimates to base parameters
            updated_params = base_parameters.copy()
            for param_name, value in estimates.items():
                param_index = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5}[param_name]
                updated_params[param_index] = value
            
            #extract meaningful parameters for the fit function
            meaningful_params = get_meaningful_parameters(fit_function, updated_params)
            adaptive_params[name] = meaningful_params
            
        except Exception as e:
            print(f"Warning: Failed to adapt parameters for {name} ({e}), using base parameters")
            adaptive_params[name] = get_meaningful_parameters(fit_function, base_parameters)
    
    return adaptive_params

def _global_strategy(dataframes_dict, fit_function, base_parameters, x_col, y_col):
    all_amplitudes = []
    all_baselines = []
    all_decay_rates = []
    
    #collect estimates from all dataframes
    for name, df in dataframes_dict.items():
        try:
            x_data = df[x_col].values
            y_data = df[y_col].values
            estimates = estimate_parameters_from_data(x_data, y_data, base_parameters)
            
            if estimates:
                all_amplitudes.append(estimates.get('a', base_parameters[0]))
                all_baselines.append(estimates.get('f', base_parameters[5]))
                all_decay_rates.append(estimates.get('b', base_parameters[1]))
        except:
            continue
    
    #calculate global estimates
    if all_amplitudes and all_baselines and all_decay_rates:
        global_params = base_parameters.copy()
        global_params[0] = np.median(all_amplitudes)  #use median for robustness
        global_params[1] = np.median(all_decay_rates)
        global_params[5] = np.median(all_baselines)
    else:
        global_params = base_parameters.copy()
    
    #apply same global parameters to all dataframes
    meaningful_params = get_meaningful_parameters(fit_function, global_params)
    return {name: meaningful_params for name in dataframes_dict.keys()}

def _representative_strategy(dataframes_dict, fit_function, base_parameters, x_col, y_col):
    #choose the dataset with the best signal-to-noise ratio
    best_snr = -1
    representative_estimates = {}
    
    for name, df in dataframes_dict.items():
        try:
            y_data = df[y_col].values
            y_mean = np.mean(y_data)
            y_std = np.std(y_data)
            snr = y_mean / y_std if y_std > 0 else 0
            
            if snr > best_snr:
                best_snr = snr
                x_data = df[x_col].values
                representative_estimates = estimate_parameters_from_data(x_data, y_data, base_parameters)
        except:
            continue
    
    #apply representative estimates
    if representative_estimates:
        updated_params = base_parameters.copy()
        for param_name, value in representative_estimates.items():
            param_index = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5}[param_name]
            updated_params[param_index] = value
    else:
        updated_params = base_parameters.copy()
    
    meaningful_params = get_meaningful_parameters(fit_function, updated_params)
    return {name: meaningful_params for name in dataframes_dict.keys()}

#convenience wrapper function for easy integration
def get_adaptive_initial_parameters(dataframes_dict, fit_function, base_parameters, 
                                  strategy='individual', x_col='t (s)', y_col='g(2)', 
                                  verbose=True):
    if verbose:
        print(f"Using '{strategy}' strategy for parameter adaptation...")
        print(f"Base parameters: {base_parameters}")
        print(f"Fit function: {fit_function.__name__}")
    
    adaptive_params = get_adaptive_parameters_strategy(
        dataframes_dict, fit_function, base_parameters, strategy, x_col, y_col
    )
    
    if verbose:
        print(f"Generated adaptive parameters for {len(adaptive_params)} datasets")

        #show parameters
        sample_names = list(adaptive_params.keys())
        for name in sample_names:
            print(f"  {name}: {adaptive_params[name]}")
    
    return adaptive_params

#fit and plot for cumulant-method C
def plot_processed_correlations_iterative(dataframes_dict, fit_function2, fit_x_limits, initial_guesses,
                                         max_iterations=5, tolerance=1e-4, maxfev=5000,
                                         method='lm', show_plots=True):
    """
    Optimized iterative fitting with optional plotting

    Performance improvements:
    - Reduced default max_iterations from 10 to 5 (usually converges earlier)
    - Reduced default maxfev from 50000 to 5000 (sufficient for most cases)
    - Added show_plots parameter to skip plotting for speed
    - Optimized redundant calculations
    """
    all_fit_results = []
    plot_number = 1
    total = len(dataframes_dict)

    for idx, (name, df) in enumerate(dataframes_dict.items(), 1):
        print(f"[{idx}/{total}] Fitting {name}...")

        fit_result = {'filename': name}
        try:
            x_data = df['t (s)']
            y_data = df['g(2)']

            x_fit = x_data[(x_data >= fit_x_limits[0]) & (x_data <= fit_x_limits[1])]
            y_fit = y_data[(x_data >= fit_x_limits[0]) & (x_data <= fit_x_limits[1])]

            if len(x_fit) < 2:
                print(f"  Not enough data points in the specified range for fitting {name}. Skipping.")
                all_fit_results.append(fit_result)
                continue

            if isinstance(initial_guesses, dict):
                current_guess = initial_guesses[name].copy()  #get the parameters for this dataset (adaptive mode)
            else:
                current_guess = initial_guesses.copy()  #use the parameters for all datasets

            #store iterations for plotting (only if needed)
            all_y_fits = [] if show_plots else None
            all_r_squared = []
            all_rmse = []
            all_aic = []

            param_names = inspect.getfullargspec(fit_function2).args[1:]  #get parameter names from function

            # Pre-calculate constants for metrics
            y_fit_mean = np.mean(y_fit)
            ss_tot = np.sum((y_fit - y_fit_mean)**2)
            n = len(y_fit)
            k = len(current_guess) + 1  # number of parameters

            for i in range(max_iterations):
                try:
                    #use the specified optimization method
                    if method == 'lm':
                        popt, pcov = curve_fit(fit_function2, x_fit, y_fit, p0=current_guess, maxfev=maxfev)
                    else:
                        #for 'trf' and 'dogbox', bounds should be specified, but just very wide bounds are used
                        bounds = ([-np.inf] * len(current_guess), [np.inf] * len(current_guess))
                        popt, pcov = curve_fit(fit_function2, x_fit, y_fit, p0=current_guess,
                                              method=method, bounds=bounds, maxfev=maxfev)

                    # Calculate fitted values and metrics
                    y_fit_current = fit_function2(x_fit, *popt)

                    if show_plots:
                        y_fit_full = fit_function2(x_data, *popt)
                        all_y_fits.append((popt, y_fit_full))

                    current_guess = popt

                    #calculate metrics efficiently
                    residuals = y_fit - y_fit_current
                    ss_res = np.sum(residuals**2)

                    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                    all_r_squared.append(r_squared)

                    rmse = np.sqrt(ss_res / n)
                    all_rmse.append(rmse)

                    aic = n * np.log(ss_res / n) + 2 * k
                    all_aic.append(aic)

                    #store metrics for this iteration
                    fit_result[f'R-squared_iter{i+1}'] = r_squared
                    fit_result[f'RMSE_iter{i+1}'] = rmse
                    fit_result[f'AIC_iter{i+1}'] = aic

                    #store parameters for this iteration
                    for j, param_name in enumerate(param_names):
                        fit_result[f'{param_name}_iter{i+1}'] = popt[j]

                    #check for convergence
                    if i > 0:
                        delta_r_squared = abs(all_r_squared[-1] - all_r_squared[-2])
                        if delta_r_squared < tolerance:
                            print(f"  Converged after {i+1} iterations.")
                            break

                except RuntimeError as e:
                    print(f"  Fit error in iteration {i+1}: {e}")
                    break

            #find the best iteration (using R-squared)
            best_iteration_index = np.argmax(all_r_squared)
            best_iteration = best_iteration_index + 1

            if show_plots:
                best_popt, best_y_fit = all_y_fits[best_iteration_index]
            else:
                # Recalculate best fit values for final metrics
                best_popt = popt  # Last popt is the best if we didn't store all
                best_y_fit = fit_function2(x_data, *best_popt)

            #store best fit results
            fit_result['best_R-squared'] = all_r_squared[best_iteration_index]
            fit_result['best_RMSE'] = all_rmse[best_iteration_index]
            fit_result['best_AIC'] = all_aic[best_iteration_index]

            for j, param_name in enumerate(param_names):
                param_value = best_popt[j]
                if isinstance(param_value, np.ndarray) and param_value.size == 1:
                    fit_result['best_' + param_name] = param_value.item()
                elif isinstance(param_value, list) and len(param_value) == 1:
                    fit_result['best_' + param_name] = param_value[0]
                else:
                    fit_result['best_' + param_name] = param_value

            #calculate parameter uncertainties from covariance matrix
            if pcov is not None:
                perr = np.sqrt(np.diag(pcov))
                for j, param_name in enumerate(param_names):
                    if j < len(perr):  # Safety check
                        err_value = perr[j]
                        if isinstance(err_value, np.ndarray) and err_value.size == 1:
                            fit_result['err_' + param_name] = err_value.item()
                        else:
                            fit_result['err_' + param_name] = err_value

            all_fit_results.append(fit_result)

            # Only create plots if requested
            if show_plots:
                #calculate residuals for the best fit for Q-Q plot
                best_y_fit_values_in_range = fit_function2(x_fit, *best_popt)
                best_residuals = y_fit - best_y_fit_values_in_range

                #horizontal subplot layout
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

                #left plot: Original data and fit iterations
                ax1.plot(x_data, y_data, marker='.', linestyle='', label='Data')

                #plot all iterations with increasing opacity
                for i, (popt_iter, y_fit_values) in enumerate(all_y_fits):
                    if i == best_iteration_index:
                        #highlight the best fit
                        ax1.plot(x_data, y_fit_values, 'r-', linewidth=2,
                                 label=f'Best Fit (Iter {i+1})')
                    else:
                        #plot other iterations with lower opacity
                        opacity = 0.3 + 0.5 * (i / len(all_y_fits))
                        ax1.plot(x_data, y_fit_values, '-', alpha=opacity,
                                 color='gray', linewidth=1)

                ax1.set_xlabel(r"lag time [s]")
                ax1.set_ylabel('g(2)-1')
                ax1.set_title(f'[{plot_number}]: g(2)-1 vs. lag time for {name}')
                ax1.grid(True)
                ax1.set_xscale('log')
                ax1.set_xlim(0, 10)
                ax1.legend()

                #right plot: Q-Q plot for the best fit
                stats.probplot(best_residuals, dist="norm", plot=ax2)
                ax2.set_title(f'[{plot_number}]: Q-Q Plot of Residuals (Best Fit: Iter {best_iteration})')
                ax2.grid(True)

                #add parameters of best fit as text
                param_text = f"Best Fit (Iteration {best_iteration}):\n"
                param_text += f"R² = {all_r_squared[best_iteration_index]:.4f}\n"
                param_text += f"RMSE = {all_rmse[best_iteration_index]:.4e}\n"
                param_text += f"AIC = {all_aic[best_iteration_index]:.2f}"

                #position the text
                ax1.text(0.95, 0.95, param_text, transform=ax1.transAxes,
                        verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

                plt.tight_layout()
                plt.show()
                plot_number += 1

        except (KeyError, TypeError) as e:
            print(f"  Error processing DataFrame '{name}': {e}")
            fit_result['Error'] = str(e)
            all_fit_results.append(fit_result)

    final_results_df = pd.DataFrame(all_fit_results)
    print(f"\nCompleted fitting {len(all_fit_results)} datasets.")
    return final_results_df


#calculate mean fit metrics to compare different fit methods
def calculate_mean_fit_metrics(results_df):
    #filter out rows with errors
    valid_results = results_df[~results_df['filename'].str.contains('Error', na=False)]
    
    #calculate means of "best fit" metrics
    mean_metrics = {
        'mean_R_squared': valid_results['best_R-squared'].mean(),
        'mean_RMSE': valid_results['best_RMSE'].mean(),
        'mean_AIC': valid_results['best_AIC'].mean(),
        'num_datasets': len(valid_results)
    }
    
    #calculate standard deviations
    mean_metrics['std_R_squared'] = valid_results['best_R-squared'].std()
    mean_metrics['std_RMSE'] = valid_results['best_RMSE'].std()
    mean_metrics['std_AIC'] = valid_results['best_AIC'].std()
    

    return mean_metrics
