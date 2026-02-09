"""
Cumulant Plotting Functions (No Show)
Modified versions of cumulant plotting functions that return figures instead of showing them
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import curve_fit, OptimizeWarning
import inspect
import warnings


def plot_processed_correlations_no_show(dataframes_dict, fit_function, fit_x_limits):
    """
    Plot and fit correlation data without showing plots

    Returns:
        tuple: (results_df, plots_dict)
            - results_df: DataFrame with fit parameters
            - plots_dict: Dictionary {filename: (fig, fit_data)}
    """
    all_fit_results = []
    plots_dict = {}
    plot_number = 1

    for name, df in dataframes_dict.items():
        try:
            fit_result = {'filename': name}

            # Main processing
            x_data = df['t (s)']
            y_data = df['g(2)_mod']
            x_fit = x_data[(x_data >= fit_x_limits[0]) & (x_data <= fit_x_limits[1])]
            y_fit = y_data[(x_data >= fit_x_limits[0]) & (x_data <= fit_x_limits[1])]

            # Perform the fit (optimized: reduced maxfev from 50000 to 5000)
            popt, pcov = curve_fit(fit_function, x_fit, y_fit, method='lm', maxfev=5000)

            # Calculate parameter errors
            perr = np.sqrt(np.diag(pcov))

            # Generate fit curve
            y_fit_values = fit_function(x_data, *popt)

            # Calculate residuals
            residuals = y_fit - fit_function(x_fit, *popt)

            # Extract parameter names and store fit parameters
            param_names = inspect.getfullargspec(fit_function).args[1:]
            for i, param_name in enumerate(param_names):
                fit_result[param_name] = popt[i]
                fit_result[f'{param_name}_error'] = perr[i]
                fit_result[f'{param_name}_relative_error'] = (perr[i] / abs(popt[i])) * 100 if popt[i] != 0 else np.inf

            # Calculate fit quality metrics
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y_fit - np.mean(y_fit))**2)
            r_squared = 1 - (ss_res / ss_tot)
            fit_result['R-squared'] = r_squared

            # Create figure (but don't show it)
            # Use Figure() instead of plt.subplots() to avoid memory leak
            from matplotlib.figure import Figure
            fig = Figure(figsize=(14, 5))
            ax1, ax2 = fig.subplots(1, 2)

            # Left plot: original data and fit
            ax1.plot(x_data, y_data, marker='.', linestyle='', label='Data')
            ax1.plot(x_data, y_fit_values, 'r-', label=f'Fit')
            ax1.set_xlabel('lag time (s)')
            ax1.set_ylabel(r"$\sqrt{g(2)-1}$")
            ax1.set_title(f'[{plot_number}]: g(2)-1 vs. lag time for {name}')
            ax1.grid(True)
            ax1.set_yscale('log')
            ax1.set_xlim(0, 0.002)
            ax1.legend()

            # Right plot: Q-Q plot
            stats.probplot(residuals, dist="norm", plot=ax2)
            ax2.set_title(f'[{plot_number}]: Q-Q Plot of Residuals')
            ax2.grid(True)

            # Add R^2 as text
            param_text = f"R² = {r_squared:.4f}"
            ax1.text(0.95, 0.95, param_text, transform=ax1.transAxes,
                    va='top', ha='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

            fig.tight_layout()

            # Store figure instead of showing it
            plots_dict[name] = (fig, {
                'popt': popt,
                'perr': perr,
                'r_squared': r_squared,
                'residuals': residuals
            })

            plot_number += 1

            # Store all results
            all_fit_results.append(fit_result)

        except (KeyError, TypeError) as e:
            print(f"Error processing DataFrame '{name}': {e}")
            fit_result['Error'] = str(e)
            all_fit_results.append(fit_result)
        except RuntimeError as e:
            print(f"Fit error for DataFrame '{name}': {e}")
            fit_result['Error'] = str(e)
            all_fit_results.append(fit_result)

    final_results_df = pd.DataFrame(all_fit_results)
    return final_results_df, plots_dict


def _fit_single_dataset_method_c(name, df, fit_function, fit_limits, initial_parameters,
                                  method, max_iterations, tolerance, plot_number):
    """
    Worker function to fit a single dataset for Method C (for multiprocessing)

    Args:
        name: Dataset name
        df: DataFrame with 't (s)' and 'g(2)' columns
        fit_function: The fit function to use
        fit_limits: Tuple of (min, max) time limits
        initial_parameters: Initial parameter guesses
        method: Optimization method ('lm', 'trf', 'dogbox')
        max_iterations: Maximum number of iterations
        tolerance: Convergence tolerance for R-squared
        plot_number: Plot number for labeling

    Returns:
        tuple: (fit_result dict, plot_data tuple)
    """
    import numpy as np
    from scipy.optimize import curve_fit
    from scipy import stats
    import inspect
    from matplotlib.figure import Figure

    fit_result = {'filename': name}

    try:
        # Get data
        x_data = df['t (s)'].values
        y_data = df['g(2)'].values

        # Filter data by fit limits
        mask = (x_data >= fit_limits[0]) & (x_data <= fit_limits[1])
        x_fit = x_data[mask]
        y_fit = y_data[mask]

        if len(x_fit) < 2:
            print(f"[Worker] Not enough data points in range for {name}. Skipping.")
            return (fit_result, None)

        # Determine which parameters to use
        param_names = inspect.getfullargspec(fit_function).args[1:]
        num_params = len(param_names)

        # Get initial parameters
        if isinstance(initial_parameters, dict):
            current_guess = initial_parameters.get(name, initial_parameters.get('default', []))[:num_params].copy()
        elif isinstance(initial_parameters, list):
            current_guess = initial_parameters[:num_params].copy()
        else:
            current_guess = initial_parameters

        # Storage for iterations
        all_y_fits = []
        all_r_squared = []
        all_rmse = []
        all_aic = []
        all_popt = []
        all_pcov = []

        # Iterative fitting loop
        for i in range(max_iterations):
            try:
                # Perform fit - suppress OptimizeWarning about covariance
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=OptimizeWarning)
                    if method == 'lm':
                        popt, pcov = curve_fit(fit_function, x_fit, y_fit,
                                              p0=current_guess, maxfev=5000)
                    else:
                        bounds = ([-np.inf] * len(current_guess), [np.inf] * len(current_guess))
                        popt, pcov = curve_fit(fit_function, x_fit, y_fit,
                                              p0=current_guess, method=method,
                                              bounds=bounds, maxfev=5000)

                # Generate fit curve
                y_fit_values = fit_function(x_data, *popt)
                all_y_fits.append(y_fit_values)
                all_popt.append(popt)
                all_pcov.append(pcov)

                # Use fitted parameters as next initial guess
                current_guess = popt.copy()

                # Calculate metrics
                y_fit_current = fit_function(x_fit, *popt)
                residuals = y_fit - y_fit_current

                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((y_fit - np.mean(y_fit))**2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                all_r_squared.append(r_squared)

                n = len(y_fit)
                rmse = np.sqrt(ss_res / n)
                all_rmse.append(rmse)

                k = len(popt) + 1
                aic = n * np.log(ss_res / n) + 2 * k
                all_aic.append(aic)

                # Store iteration metrics
                fit_result[f'R-squared_iter{i+1}'] = r_squared
                fit_result[f'RMSE_iter{i+1}'] = rmse
                fit_result[f'AIC_iter{i+1}'] = aic

                for j, param_name in enumerate(param_names):
                    fit_result[f'{param_name}_iter{i+1}'] = popt[j]

                # Check convergence
                if i > 0:
                    delta_r_squared = abs(all_r_squared[i] - all_r_squared[i-1])
                    if delta_r_squared < tolerance:
                        break

            except RuntimeError as e:
                print(f"[Worker] Fit error in iteration {i+1} for {name}: {e}")
                break

        # Find best iteration
        if len(all_r_squared) == 0:
            print(f"[Worker] No successful fits for {name}")
            return (fit_result, None)

        best_iteration_index = np.argmax(all_r_squared)
        best_y_fit = all_y_fits[best_iteration_index]
        best_popt = all_popt[best_iteration_index]
        best_pcov = all_pcov[best_iteration_index]
        best_iteration = best_iteration_index + 1

        # Calculate errors
        perr = np.sqrt(np.diag(best_pcov))

        # Calculate best residuals
        best_y_fit_in_range = fit_function(x_fit, *best_popt)
        residuals = y_fit - best_y_fit_in_range

        # Store best results
        r_squared = all_r_squared[best_iteration_index]
        rmse = all_rmse[best_iteration_index]
        aic = all_aic[best_iteration_index]

        for i, param_name in enumerate(param_names):
            fit_result[f'best_{param_name}'] = best_popt[i]
            fit_result[f'best_{param_name}_error'] = perr[i]

        fit_result['R_squared'] = r_squared
        fit_result['RMSE'] = rmse
        fit_result['AIC'] = aic

        # Create figure
        fig = Figure(figsize=(14, 5))
        ax1, ax2 = fig.subplots(1, 2)

        # Left plot: Data and fits
        ax1.plot(x_data, y_data, marker='.', linestyle='', label='Data')

        for i, y_fit_iter in enumerate(all_y_fits):
            if i == best_iteration_index:
                ax1.plot(x_data, y_fit_iter, 'r-', linewidth=2,
                        label=f'Best Fit (Iter {i+1})')
            else:
                opacity = 0.3 + 0.5 * (i / len(all_y_fits))
                ax1.plot(x_data, y_fit_iter, '-', alpha=opacity,
                        color='gray', linewidth=1)

        ax1.set_xlabel('lag time [s]')
        ax1.set_ylabel('g(2)-1')
        ax1.set_title(f'[{plot_number}]: g(2)-1 vs. lag time for {name}')
        ax1.grid(True)
        ax1.set_xscale('log')
        ax1.set_xlim(left=x_data.min()*0.9, right=x_data.max()*1.1)
        ax1.legend()

        param_text = f"Best Fit (Iteration {best_iteration}):\n"
        param_text += f"R² = {r_squared:.4f}\n"
        param_text += f"RMSE = {rmse:.4e}\n"
        param_text += f"AIC = {aic:.2f}"
        ax1.text(0.95, 0.95, param_text, transform=ax1.transAxes,
                va='top', ha='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        # Right plot: Q-Q plot
        stats.probplot(residuals, dist="norm", plot=ax2)
        ax2.set_title(f'[{plot_number}]: Q-Q Plot of Residuals (Best Fit: Iter {best_iteration})')
        ax2.grid(True)

        fig.tight_layout()

        # Return figure and metadata
        plot_data = (fig, {
            'popt': best_popt,
            'perr': perr,
            'r_squared': r_squared,
            'rmse': rmse,
            'aic': aic,
            'residuals': residuals,
            'x_data': x_data,
            'y_data': y_data,
            'y_fit': best_y_fit
        })

        return (fit_result, plot_data)

    except Exception as e:
        print(f"[Worker] Error processing {name}: {e}")
        import traceback
        traceback.print_exc()
        return (fit_result, None)


def plot_processed_correlations_iterative_no_show(dataframes_dict, fit_function, fit_limits,
                                                   initial_parameters, method='lm',
                                                   max_iterations=5, tolerance=1e-4,
                                                   use_multiprocessing=False):
    """
    Optimized iterative non-linear fit without showing plots

    Performance improvements:
    - Reduced default max_iterations from 10 to 5 (usually converges earlier)
    - Reduced maxfev from 50000 to 5000 (sufficient for most cases)
    - Added progress logging
    - Optional multiprocessing support for faster processing

    Args:
        dataframes_dict: Dictionary of dataframes with 't (s)' and 'g(2)' columns
        fit_function: The fit function to use
        fit_limits: Tuple of (min, max) time limits
        initial_parameters: Initial parameter guesses (dict or list)
        method: Optimization method ('lm', 'trf', 'dogbox')
        max_iterations: Maximum number of iterations (default: 5, optimized from 10)
        tolerance: Convergence tolerance for R-squared (default: 1e-4)
        use_multiprocessing: Enable parallel processing (default: False)

    Returns:
        tuple: (results_df, plots_dict)
    """
    all_fit_results = []
    plots_dict = {}
    plot_number = 1
    total = len(dataframes_dict)

    # Try multiprocessing if requested and enough datasets
    if use_multiprocessing and total > 3:
        try:
            from joblib import Parallel, delayed
            import multiprocessing as mp

            n_jobs = mp.cpu_count()
            print(f"[Method C] Using parallel processing with {n_jobs} CPU cores")

            # Prepare arguments for workers
            args_list = []
            for idx, (name, df) in enumerate(dataframes_dict.items(), 1):
                args_list.append((name, df, fit_function, fit_limits, initial_parameters,
                                method, max_iterations, tolerance, idx))

            # Process in parallel
            results_list = Parallel(n_jobs=n_jobs, backend='loky', verbose=5)(
                delayed(_fit_single_dataset_method_c)(*args)
                for args in args_list
            )

            # Collect results
            for fit_result, plot_data in results_list:
                all_fit_results.append(fit_result)
                if plot_data is not None:
                    plots_dict[fit_result['filename']] = plot_data

            print(f"[Method C] Parallel processing complete: {len(all_fit_results)} datasets processed")

        except ImportError as e:
            print(f"[Method C] Warning: joblib not available ({e})")
            print("[Method C] Install required packages: pip install joblib")
            print("[Method C] Falling back to sequential processing")
            use_multiprocessing = False
        except Exception as e:
            print(f"[Method C] Warning: Parallel processing failed: {type(e).__name__}: {str(e)}")
            print("[Method C] Falling back to sequential processing")
            use_multiprocessing = False
            import traceback
            traceback.print_exc()

    # Sequential processing (fallback or if multiprocessing disabled)
    if not use_multiprocessing or total <= 3:
        print(f"[Method C] Using sequential processing for {total} datasets")

        for idx, (name, df) in enumerate(dataframes_dict.items(), 1):
            print(f"[Method C: {idx}/{total}] Fitting {name}...")
            try:
                fit_result = {'filename': name}

                # Get data
                x_data = df['t (s)'].values
                y_data = df['g(2)'].values

                # Filter data by fit limits
                mask = (x_data >= fit_limits[0]) & (x_data <= fit_limits[1])
                x_fit = x_data[mask]
                y_fit = y_data[mask]

                if len(x_fit) < 2:
                    print(f"Not enough data points in the specified range for fitting {name}. Skipping Fit.")
                    all_fit_results.append(fit_result)
                    continue

                # Determine which parameters to use based on fit_function
                param_names = inspect.getfullargspec(fit_function).args[1:]
                num_params = len(param_names)

                # Get initial parameters for this function
                if isinstance(initial_parameters, dict):
                    # If it's a dict, use parameters for this specific dataset (adaptive mode)
                    current_guess = initial_parameters.get(name, initial_parameters.get('default', []))[:num_params].copy()
                elif isinstance(initial_parameters, list):
                    current_guess = initial_parameters[:num_params].copy()
                else:
                    current_guess = initial_parameters

                # Store iterations for analysis
                all_y_fits = []
                all_r_squared = []
                all_rmse = []
                all_aic = []
                all_popt = []
                all_pcov = []

                # Iterative fitting loop (key difference from simple fit!)
                for i in range(max_iterations):
                    try:
                        # Perform fit with current guess (optimized: maxfev reduced from 50000 to 5000)
                        if method == 'lm':
                            popt, pcov = curve_fit(fit_function, x_fit, y_fit,
                                                  p0=current_guess, maxfev=5000)
                        else:
                            # For 'trf' and 'dogbox', use wide bounds
                            bounds = ([-np.inf] * len(current_guess), [np.inf] * len(current_guess))
                            popt, pcov = curve_fit(fit_function, x_fit, y_fit,
                                                  p0=current_guess, method=method,
                                                  bounds=bounds, maxfev=5000)

                        # Generate fit curve
                        y_fit_values = fit_function(x_data, *popt)
                        all_y_fits.append(y_fit_values)
                        all_popt.append(popt)
                        all_pcov.append(pcov)

                        # Use fitted parameters as next initial guess
                        current_guess = popt.copy()

                        # Calculate residuals (only in fitting range)
                        y_fit_current = fit_function(x_fit, *popt)
                        residuals = y_fit - y_fit_current

                        # Calculate R-squared
                        ss_res = np.sum(residuals**2)
                        ss_tot = np.sum((y_fit - np.mean(y_fit))**2)
                        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                        all_r_squared.append(r_squared)

                        # Calculate RMSE
                        n = len(y_fit)
                        rmse = np.sqrt(ss_res / n)
                        all_rmse.append(rmse)

                        # Calculate AIC (Akaike Information Criterion)
                        k = len(popt) + 1
                        aic = n * np.log(ss_res / n) + 2 * k
                        all_aic.append(aic)

                        # Store metrics for this iteration
                        fit_result[f'R-squared_iter{i+1}'] = r_squared
                        fit_result[f'RMSE_iter{i+1}'] = rmse
                        fit_result[f'AIC_iter{i+1}'] = aic

                        # Store parameters for this iteration
                        for j, param_name in enumerate(param_names):
                            fit_result[f'{param_name}_iter{i+1}'] = popt[j]

                        # Check for convergence
                        if i > 0:
                            delta_r_squared = abs(all_r_squared[i] - all_r_squared[i-1])
                            if delta_r_squared < tolerance:
                                print(f"  Converged after {i+1} iterations.")
                                break

                    except RuntimeError as e:
                        print(f"  Fit error in iteration {i+1}: {e}")
                        break

                # Find the best iteration (using R-squared)
                if len(all_r_squared) > 0:
                    best_iteration_index = np.argmax(all_r_squared)
                    best_y_fit = all_y_fits[best_iteration_index]
                    best_popt = all_popt[best_iteration_index]
                    best_pcov = all_pcov[best_iteration_index]
                    best_iteration = best_iteration_index + 1

                    # Calculate errors from best fit
                    perr = np.sqrt(np.diag(best_pcov))

                    # Calculate best residuals
                    best_y_fit_in_range = fit_function(x_fit, *best_popt)
                    residuals = y_fit - best_y_fit_in_range

                    # Generate smooth fit curve for plotting
                    x_plot = np.logspace(np.log10(x_data.min()), np.log10(x_data.max()), 1000)
                    y_fit_plot = fit_function(x_plot, *best_popt)

                    # Store best fit results
                    r_squared = all_r_squared[best_iteration_index]
                    rmse = all_rmse[best_iteration_index]
                    aic = all_aic[best_iteration_index]

                    # Store results with 'best_' prefix
                    for i, param_name in enumerate(param_names):
                        fit_result[f'best_{param_name}'] = best_popt[i]
                        fit_result[f'best_{param_name}_error'] = perr[i]

                    fit_result['R_squared'] = r_squared
                    fit_result['RMSE'] = rmse
                    fit_result['AIC'] = aic
                else:
                    # No successful iterations
                    print(f"No successful fits for {name}")
                    all_fit_results.append(fit_result)
                    continue

                # Create figure (2x1 layout like original notebook)
                # Use Figure() instead of plt.subplots() to avoid memory leak
                from matplotlib.figure import Figure
                fig = Figure(figsize=(14, 5))
                ax1, ax2 = fig.subplots(1, 2)

                # Left plot: Data and all fit iterations
                ax1.plot(x_data, y_data, marker='.', linestyle='', label='Data')

                # Plot all iterations with increasing opacity
                for i, y_fit_iter in enumerate(all_y_fits):
                    if i == best_iteration_index:
                        # Highlight the best fit
                        ax1.plot(x_data, y_fit_iter, 'r-', linewidth=2,
                                label=f'Best Fit (Iter {i+1})')
                    else:
                        # Plot other iterations with lower opacity
                        opacity = 0.3 + 0.5 * (i / len(all_y_fits))
                        ax1.plot(x_data, y_fit_iter, '-', alpha=opacity,
                                color='gray', linewidth=1)

                ax1.set_xlabel('lag time [s]')
                ax1.set_ylabel('g(2)-1')
                ax1.set_title(f'[{plot_number}]: g(2)-1 vs. lag time for {name}')
                ax1.grid(True)
                ax1.set_xscale('log')
                ax1.set_xlim(left=x_data.min()*0.9, right=x_data.max()*1.1)
                ax1.legend()

                # Add parameters of best fit as text
                param_text = f"Best Fit (Iteration {best_iteration}):\n"
                param_text += f"R² = {r_squared:.4f}\n"
                param_text += f"RMSE = {rmse:.4e}\n"
                param_text += f"AIC = {aic:.2f}"
                ax1.text(0.95, 0.95, param_text, transform=ax1.transAxes,
                        va='top', ha='right',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

                # Right plot: Q-Q plot for the best fit
                stats.probplot(residuals, dist="norm", plot=ax2)
                ax2.set_title(f'[{plot_number}]: Q-Q Plot of Residuals (Best Fit: Iter {best_iteration})')
                ax2.grid(True)

                fig.tight_layout()

                # Store figure
                plots_dict[name] = (fig, {
                    'popt': best_popt,
                    'perr': perr,
                    'r_squared': r_squared,
                    'rmse': rmse,
                    'aic': aic,
                    'residuals': residuals,
                    'num_iterations': len(all_y_fits),
                    'best_iteration': best_iteration
                })

                plot_number += 1
                all_fit_results.append(fit_result)

            except Exception as e:
                print(f"[CUMULANT METHOD C ERROR] Failed to fit '{name}': {e}")
                import traceback
                traceback.print_exc()
                fit_result['Error'] = str(e)
                all_fit_results.append(fit_result)

    final_results_df = pd.DataFrame(all_fit_results)
    print(f"[CUMULANT METHOD C] Generated {len(plots_dict)} plots from {len(dataframes_dict)} datasets")
    return final_results_df, plots_dict


def create_summary_plot(data_df, q_squared_col, gamma_cols, method_names=None, gamma_unit='1/s'):
    """
    Create summary plot (Γ vs q²) and return the figure

    Returns:
        matplotlib.figure.Figure: The summary plot figure
    """
    import statsmodels.api as sm

    if not isinstance(gamma_cols, list):
        gamma_cols = [gamma_cols]

    if method_names is None:
        method_names = gamma_cols
    elif not isinstance(method_names, list):
        method_names = [method_names]

    # Create figure (use Figure() to avoid memory leak)
    from matplotlib.figure import Figure
    fig = Figure(figsize=(10, 6))
    ax = fig.add_subplot(111)

    X_full = data_df[q_squared_col]

    colors = ['blue', 'green', 'red', 'orange', 'purple']

    for i, gamma_col in enumerate(gamma_cols):
        if gamma_col not in data_df.columns:
            continue

        method_name = method_names[i] if i < len(method_names) else ''
        Y_full = data_df[gamma_col]

        # Scatter plot with larger, more visible markers
        ax.scatter(X_full, Y_full, alpha=0.7, s=20,
                  label=f'{method_name} Data',
                  color=colors[i % len(colors)],
                  marker='o', edgecolors='black', linewidths=0.5)

        # Linear regression
        X_fit_with_constant = sm.add_constant(X_full)
        model = sm.OLS(Y_full, X_fit_with_constant).fit()

        # Plot regression line with thinner line to emphasize data points
        X_line = np.linspace(X_full.min(), X_full.max(), 100)
        X_line_with_constant = sm.add_constant(X_line)
        y_predicted_line = model.predict(X_line_with_constant)
        ax.plot(X_line, y_predicted_line, '-', linewidth=2,
               label=f'{method_name} Fit (R²={model.rsquared:.4f})',
               color=colors[i % len(colors)], alpha=0.8)

    ax.set_xlabel(f'q² [nm⁻²]')
    ax.set_ylabel(f'Γ [{gamma_unit}]')
    ax.set_title('Diffusion Coefficient Analysis')
    ax.legend()
    ax.grid(True)

    fig.tight_layout()

    return fig
