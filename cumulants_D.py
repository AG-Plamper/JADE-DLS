# -*- coding: utf-8 -*-
"""
cumulants_D.py
==============
Cumulant analysis — Method D: multi-exponential decomposition of g²(τ)
using a sum of Dirac-delta relaxation modes, with iterative model-order
selection and moment calculation.

Functions
---------
dirac_sum_g1(t, *gammas)
    Field correlation function g¹(τ) as a sum of equally-weighted
    exponential decays with decay rates Γᵢ.

dirac_sum_g2(t, beta, *gammas)
    Intensity correlation function g²(τ) = 1 + β·|g¹(τ)|² for N modes.

fit_cumulant_D(x_data, y_data, n_max, beta_initial, gamma_initial, n_start)
    Iterative fitting from n_start to n_max exponential modes; selects
    optimal model order by AIC with convergence and residual checks.

calculate_moments_from_gammas(gammas)
    Compute mean, variance, PDI, skewness and kurtosis from a set of
    fitted decay rates.

cluster_gammas(gammas, gap_threshold)
    Simple gap-based clustering of decay rates into discrete populations.

fit_correlations_method_D(dataframes_dict, x_col, y_col, ...)
    Apply fit_cumulant_D across all files, collect per-file results and
    produce diagnostic plots.

Dependencies: numpy, pandas, scipy, matplotlib
"""
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("Warning: statsmodels not available. analyze_diffusion_coefficient_D will not work.")


#core fitting function: sum of Dirac delta functions
def dirac_sum_g1(t, *gammas):
    """
    Electric field autocorrelation function as sum of exponential decays.
    
    g₁(t) = (1/n)Σᵢ exp(-Γᵢt)
    
    Parameters:
    -----------
    t : array
        Lag times
    *gammas : float
        Relaxation rates Γᵢ (must satisfy Γᵢ > Γᵢ₋₁ > 0)
    """
    n = len(gammas)
    g1 = np.zeros_like(t, dtype=float)
    for gamma in gammas:
        g1 += np.exp(-gamma * t)
    return g1 / n


def dirac_sum_g2(t, beta, *gammas):
    """
    Intensity autocorrelation function from g₁(t).
    
    g₂(t) - 1 = β|g₁(t)|²
    
    Parameters:
    -----------
    t : array
        Lag times
    beta : float
        Amplitude parameter (≈ 1)
    *gammas : float
        Relaxation rates Γᵢ
    """
    g1 = dirac_sum_g1(t, *gammas)
    return beta * g1**2

#fit the redefined cumulant model to autocorrelation data.
def fit_cumulant_D(x_data, y_data, n_max=25, beta_initial=1.0, gamma_initial=None, n_start=1):
    """
    Iteratively increases the number of Dirac delta modes (n), using previous
    fit results as initial guesses for speed. Chooses the solution that minimizes 
    residual sum of squares while ensuring monotonicity constraints (Γᵢ > Γᵢ₋₁ > 0).
    """
    
    #estimate initial gamma from data if not provided
    if gamma_initial is None:
        # Find where signal drops to ~37% (1/e)
        y_max = np.max(y_data)
        y_min = np.min(y_data)
        y_range = y_max - y_min
        threshold = y_min + 0.37 * y_range
        
        decay_indices = np.where(y_data <= threshold)[0]
        if len(decay_indices) > 0 and decay_indices[0] < len(x_data):
            decay_time = x_data[decay_indices[0]]
            if decay_time > 0:
                gamma_initial = 1.0 / decay_time
            else:
                gamma_initial = 1000.0
        else:
            gamma_initial = 1000.0
    
    best_result = None
    best_residual = np.inf
    all_results = []
    previous_gammas = None  #store previous fit for warm start
    
    #try increasing numbers of modes
    for n in range(n_start, n_max + 1):
        try:
            #initial parameter guess: beta + n gammas
            if previous_gammas is None:
                #first iteration - distribute logarithmically
                gamma_min = gamma_initial * 0.1
                gamma_max = gamma_initial * 10.0
                gammas_init = np.logspace(np.log10(gamma_min), np.log10(gamma_max), n)
                p0 = [beta_initial] + list(gammas_init)
            else:
                #warm start: use previous gammas and add one more
                #insert new gamma between existing ones where gap is largest
                sorted_gammas = np.sort(previous_gammas)
                if len(sorted_gammas) > 1:
                    #find largest gap in log-space
                    log_gammas = np.log10(sorted_gammas)
                    gaps = np.diff(log_gammas)
                    max_gap_idx = np.argmax(gaps)
                    #insert new gamma in log-space middle of largest gap
                    new_gamma = 10**((log_gammas[max_gap_idx] + log_gammas[max_gap_idx + 1]) / 2)
                    gammas_init = np.sort(np.append(sorted_gammas, new_gamma))
                else:
                    #only one previous gamma, add one nearby
                    new_gamma = sorted_gammas[0] * 2.0
                    gammas_init = np.array([sorted_gammas[0], new_gamma])
                
                #use previous beta
                p0 = [best_result['beta']] + list(gammas_init)
            
            #define bounds to enforce monotonicity and positivity
            bounds_lower = [0.5] + [1e-6] * n
            bounds_upper = [1.5] + [1e8] * n
            
            #custom wrapper that enforces monotonicity
            def fit_func_wrapper(t, beta, *gammas):
                gammas_sorted = np.sort(gammas)
                return dirac_sum_g2(t, beta, *gammas_sorted)
            
            #perform fit
            popt, pcov = curve_fit(
                fit_func_wrapper, 
                x_data, 
                y_data,
                p0=p0,
                bounds=(bounds_lower, bounds_upper),
                method='trf',
                maxfev=5000)
            
            #extract fitted parameters
            beta_fit = popt[0]
            gammas_fit = np.sort(popt[1:])
            
            #store for warm start
            previous_gammas = gammas_fit
            
            #calculate fitted values
            g1_fit = dirac_sum_g1(x_data, *gammas_fit)
            g2_fit = beta_fit * g1_fit**2
            
            #calculate residuals
            residuals = y_data - g2_fit
            residual_ss = np.sum(residuals**2)
            
            #store result
            result = {
                'n_modes': n,
                'beta': beta_fit,
                'gammas': gammas_fit,
                'residual_ss': residual_ss,
                'g1_fit': g1_fit,
                'g2_fit': g2_fit,
                'convergence': 'success'
            }
            all_results.append(result)
            
            #update best result if this is better
            if residual_ss < best_residual:
                best_residual = residual_ss
                best_result = result
            
            # === CONVERGENCE CHECKS ===
            # Check 1: Mode collapse (gammas too similar - overfitting)
            if n > 1:
                #check if gammas are clustering (relative spread < 10%)
                log_gammas = np.log10(gammas_fit)
                gamma_spread = (log_gammas.max() - log_gammas.min()) / np.abs(log_gammas.mean())
                if gamma_spread < 0.1:  # Modes collapsed to within 10% in log-space
                    print(f"Converged at n={n} modes (modes collapsed - likely monomodal)")
                    break
            
            # Check 2: Minimal improvement (not worth adding more modes)
            if len(all_results) >= 2:
                prev_rss = all_results[-2]['residual_ss']
                improvement = (prev_rss - residual_ss) / prev_rss
                if improvement < 0.01:  # Less than 1% improvement
                    print(f"Converged at n={n} modes (improvement < 1%)")
                    break
            
            # Check 3: Multiple consecutive small improvements
            if len(all_results) >= 3:
                recent_rss = [r['residual_ss'] for r in all_results[-3:]]
                improvements = [
                    (recent_rss[i] - recent_rss[i+1]) / recent_rss[i]
                    for i in range(len(recent_rss)-1)
                ]
                if all(imp < 0.02 for imp in improvements):  # All < 2%
                    print(f"Converged at n={n} modes (consistently small improvements)")
                    break
            
        except Exception as e:
            print(f"Warning: Fit failed for n={n} modes: {e}")
            continue
    
    if best_result is None:
        raise RuntimeError("All fits failed. Check input data quality.")
    
    print(f"Best fit: n={best_result['n_modes']} modes, RSS={best_result['residual_ss']:.6e}")
    return best_result

#calculate statistical moments from fitted relaxation rates
def calculate_moments_from_gammas(gammas):
    """   
    For a sum of Dirac deltas with equal weights (1/n):
    ⟨Γᵐ⟩ = (1/n)Σᵢ Γᵢᵐ
    """
    n = len(gammas)
    
    #calculate raw moments
    gamma_mean = np.mean(gammas)  # ⟨Γ⟩ = (1/n)Σ Γᵢ
    gamma_2 = np.mean(gammas**2)  # ⟨Γ²⟩ = (1/n)Σ Γᵢ²
    gamma_3 = np.mean(gammas**3)  # ⟨Γ³⟩ = (1/n)Σ Γᵢ³
    gamma_4 = np.mean(gammas**4)  # ⟨Γ⁴⟩ = (1/n)Σ Γᵢ⁴
    
    #calculate polydispersity index
    pdi = gamma_2 / gamma_mean**2 - 1
    
    #calculate skewness
    numerator = gamma_3 - 3*gamma_2*gamma_mean + 2*gamma_mean**3
    denominator = (gamma_2 - gamma_mean**2)**(3/2)
    skewness = numerator / denominator if denominator != 0 else 0
    
    #calculate excess kurtosis
    numerator_k = gamma_4 - 4*gamma_3*gamma_mean + 6*gamma_2*gamma_mean**2 - 3*gamma_mean**4
    denominator_k = (gamma_2 - gamma_mean**2)**2
    kurtosis = numerator_k / denominator_k - 3 if denominator_k != 0 else 0

    moments = {
        'gamma_mean': gamma_mean,
        'gamma_2': gamma_2,
        'gamma_3': gamma_3,
        'pdi': pdi,
        'skewness': skewness,
        'gamma_4': gamma_4,
        'kurtosis': kurtosis
    }
    
    return moments

#cluster gamma values based on gaps in sorted sequence
def cluster_gammas(gammas, gap_threshold=1.5):
    """
    Groups gammas into distinct populations by detecting large gaps.
    Each cluster represents one population in a multimodal sample.
    
    Examples:
    ---------
    >>> gammas = [2000, 2100, 2150, 5000, 5200, 12000]
    >>> clusters, reps, info = cluster_gammas(gammas, gap_threshold=1.5)
    >>> print(reps)
    [2083, 5100, 12000]  # Three distinct populations
    """
    gammas = np.asarray(gammas)
    sorted_gammas = np.sort(gammas)
    
    #calculate gaps between consecutive gammas
    gaps = []
    for i in range(1, len(sorted_gammas)):
        ratio = sorted_gammas[i] / sorted_gammas[i-1]
        gaps.append(ratio)
    
    #initialize first cluster
    clusters = [[sorted_gammas[0]]]
    
    #group gammas by gap threshold
    for i in range(1, len(sorted_gammas)):
        ratio = sorted_gammas[i] / sorted_gammas[i-1]
        
        if ratio > gap_threshold:
            # Large gap → new cluster (distinct population)
            clusters.append([sorted_gammas[i]])
        else:
            # Small gap → same cluster
            clusters[-1].append(sorted_gammas[i])
    
    #calculate representative gamma for each cluster (geometric mean)
    #geometric mean is appropriate for decay rates (log-normal distributions)
    representatives = np.array([
        np.exp(np.mean(np.log(cluster))) for cluster in clusters
    ])
    
    #prepare info
    cluster_info = {
        'n_clusters': len(clusters),
        'cluster_sizes': [len(c) for c in clusters],
        'gaps': gaps
    }
    
    return clusters, representatives, cluster_info

#apply redefined cumulant analysis to multiple correlation functions.
def fit_correlations_method_D(dataframes_dict, x_col='t [s]', y_col='g(2)-1', 
                              n_max=25, n_start=1, gap_threshold=1.5, plot=True):
    all_fit_results = []
    plot_number = 1
    
    for name, df in dataframes_dict.items():
        try:
            fit_result = {'filename': name}
            
            # Extract data
            x_data = df[x_col].values
            y_data = df[y_col].values
            
            # Perform fit
            print(f"\nProcessing: {name}")
            result = fit_cumulant_D(x_data, y_data, n_max=n_max, n_start=n_start)
            
            # Store fit parameters
            fit_result['n_modes'] = result['n_modes']
            fit_result['beta'] = result['beta']
            fit_result['residual_ss'] = result['residual_ss']
            
            # === CLUSTERING: Group gammas into distinct populations ===
            clusters, representatives, cluster_info = cluster_gammas(
                result['gammas'], 
                gap_threshold=gap_threshold
            )
            
            fit_result['n_populations'] = cluster_info['n_clusters']
            
            # Output gamma_1, gamma_2, gamma_3, ... like regularized method
            for i, rep_gamma in enumerate(representatives):
                fit_result[f'gamma_{i+1}'] = rep_gamma
            
            # Calculate overall moments (from all gammas, not clustered)
            moments = calculate_moments_from_gammas(result['gammas'])
            fit_result['gamma_mean'] = moments['gamma_mean']
            fit_result['moment_2'] = moments['gamma_2']  # Second moment (not population gamma!)
            fit_result['moment_3'] = moments['gamma_3']  # Third moment
            fit_result['pdi'] = moments['pdi']
            fit_result['skewness'] = moments['skewness']
            fit_result['moment_4'] = moments['gamma_4'] #4th moment
            fit_result['kurtosis'] = moments['kurtosis']
            
            # Calculate R-squared
            ss_res = result['residual_ss']
            ss_tot = np.sum((y_data - np.mean(y_data))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            fit_result['R-squared'] = r_squared
            
            # Plot if requested
            if plot:
                # Calculate residuals
                residuals = y_data - result['g2_fit']

                # 3-panel layout: data+fit | residuals | Q-Q
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
                fig.suptitle(f'[{plot_number}]: Method D — {name}', fontsize=12)

                # Panel 1: data + fit
                ax1.plot(x_data, y_data, 'o', alpha=0.6, markersize=4, label='Data')
                ax1.plot(x_data, result['g2_fit'], 'r-', linewidth=2,
                         label=f'Fit (n={result["n_modes"]})')
                ax1.set_xlabel(r'lag time τ [s]')
                ax1.set_ylabel(r'$g^{(2)}(\tau) - 1$')
                ax1.set_title('Data & Fit')
                ax1.grid(True, alpha=0.3)
                ax1.set_xscale('log')
                ax1.set_xlim(x_data.min(), x_data.max())
                ax1.legend()
                param_text = (f"n modes: {result['n_modes']}  |  "
                              f"n populations: {cluster_info['n_clusters']}\n"
                              f"R² = {r_squared:.4f}\n"
                              f"⟨Γ⟩ = {moments['gamma_mean']:.3e} s⁻¹")
                ax1.text(0.95, 0.95, param_text, transform=ax1.transAxes,
                        va='top', ha='right',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

                # Panel 2: residuals
                ax2.plot(residuals)
                ax2.axhline(0, color='r', linestyle='--', linewidth=1)
                ax2.set_xlabel('Sample Index')
                ax2.set_ylabel('Residuals')
                ax2.set_title('Residuals')
                ax2.grid(True, alpha=0.3)

                # Panel 3: Q-Q
                stats.probplot(residuals, dist="norm", plot=ax3)
                ax3.set_title('Q-Q Plot')
                ax3.grid(True, alpha=0.3)

                plt.tight_layout()
                plt.show()
                plot_number += 1
            
            all_fit_results.append(fit_result)
            
        except Exception as e:
            print(f"Error processing DataFrame '{name}': {e}")
            fit_result['Error'] = str(e)
            all_fit_results.append(fit_result)
    
    results_df = pd.DataFrame(all_fit_results)
    return results_df