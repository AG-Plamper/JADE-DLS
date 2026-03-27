# -*- coding: utf-8 -*-
"""
cumulants_C.py
==============
Cumulant analysis — Method C: nonlinear fitting of g²(τ) using
scipy.optimize.curve_fit with two operating modes:

  adaptive mode  (initial_guesses dict)  — hierarchical log-uniform zoom grid
                                           search for the best starting decay
                                           rate, with per-dataset pre-fitting.
  expert   mode  (initial_guesses list)  — single-start iterative refinement
                                           from user-supplied parameters.

Public functions
----------------
get_meaningful_parameters(fit_function, full_parameter_list)
    Return the parameter subset relevant to the given fit function
    (fit_function1–4 use subsets of [a, b, c, d, e, f]).

_simple_exp_prefit(x_data, y_data)
    Fast 3-parameter single-exponential pre-fit; returns (a, b, f).
    Falls back to data-derived heuristics on failure.

estimate_parameters_from_data(x_data, y_data, base_parameters)
    Estimate all six initial parameters from correlation data using
    _simple_exp_prefit for a, b, f and scaled priors for c, d, e.

get_adaptive_initial_parameters(dataframes_dict, fit_function, base_parameters, ...)
    Return a dict of per-dataset initial guesses (one pre-fit per file).

plot_processed_correlations_iterative(dataframes_dict, fit_function2, fit_x_limits,
                                      initial_guesses, ...)
    Fit and plot all datasets. Dispatches to adaptive or expert mode based on
    whether initial_guesses is a dict or list. Returns a DataFrame of results.

Dependencies: numpy, pandas, scipy, matplotlib
"""
import numpy as np
from scipy.optimize import curve_fit
import inspect
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

#physical parameter bounds used when method='trf' or 'dogbox'
#c = μ₂ (variance) and e = μ₄ (4th moment) are mathematically non-negative;
#d = μ₃ (3rd moment) is unconstrained in sign (left/right skew both valid)
_BOUNDS_LOWER = {'a':  0.0, 'b':  0.0, 'c':  0.0, 'd': -np.inf, 'e':  0.0, 'f': -0.1}
_BOUNDS_UPPER = {'a':  1.0, 'b':  1e6, 'c':  np.inf, 'd': np.inf, 'e': np.inf, 'f':  0.1}

#get the meaningful initial parameters for cumulant fit method C
def get_meaningful_parameters(fit_function, full_parameter_list):
    #extract individual parameters
    a, b, c, d, e, f = full_parameter_list
    
    func_name = fit_function.__name__
    
    if func_name == 'fit_function1':
        return [a, b, f]
    elif func_name == 'fit_function2':
        return [a, b, c, f]
    elif func_name == 'fit_function3':  
        return [a, b, c, d, f]
    elif func_name == 'fit_function4':
        return [a, b, c, d, e, f]
    else:
        raise ValueError(f"Unknown fit function: {func_name}")

#module-level helper: single exponential model used by the pre-fit
def _simple_exp(t, a, b, f):
    return a * np.exp(-b * t) + f

#module-level pre-fit: returns (a, b, f) from a fast 3-parameter single-exponential fit.
#used both by estimate_parameters_from_data() and the zoom loop in the main fitting function.
def _simple_exp_prefit(x_data, y_data):
    """Fast 3-parameter single-exponential pre-fit to estimate a, b, f.
    Falls back to safe data-derived heuristics if the fit diverges."""
    try:
        p0 = [float(y_data[0] - y_data[-1]),
              float(1.0 / x_data[len(x_data) // 2]),
              float(y_data[-1])]
        popt, _ = curve_fit(
            _simple_exp, x_data, y_data,
            p0=p0,
            method='trf',
            bounds=([0.0,  0.0, -0.1],
                    [1.0, 1e6,  0.1]),
            maxfev=5000
        )
        return float(popt[0]), float(popt[1]), float(popt[2])
    except Exception:
        a = float(np.clip(y_data[0] - y_data[-1], 0.01, 1.0))
        f = float(np.clip(y_data[-1], -0.1, 0.1))
        b = float(np.clip(1.0 / x_data[len(x_data) // 2], 1.0, 1e6))
        return a, b, f

#estimation of initial parameters for cumulant fit method C.
#a, b, f from _simple_exp_prefit(); c, d, e from physically motivated priors scaled to b.
def estimate_parameters_from_data(x_data, y_data, base_parameters):
    try:
        amplitude_estimate, decay_rate_estimate, baseline_estimate = _simple_exp_prefit(x_data, y_data)

        c_estimate = 0.05 * decay_rate_estimate**2   # PDI ≈ 0.05 starting point
        d_estimate = 0.01 * decay_rate_estimate**3   # small positive asymmetry prior
        e_estimate = 3    * c_estimate**2             # Gaussian-distribution assumption

        #apply physical bounds as a safety net
        decay_rate_estimate = np.clip(decay_rate_estimate, 1,    1e6)
        amplitude_estimate  = np.clip(amplitude_estimate,  0.01, 10)
        baseline_estimate   = np.clip(baseline_estimate,  -1,    1)
        c_estimate = np.clip(c_estimate, 0,                       decay_rate_estimate**2)
        d_estimate = np.clip(d_estimate, -0.1 * decay_rate_estimate**3, 0.1 * decay_rate_estimate**3)
        e_estimate = np.clip(e_estimate, 0,                       10 * c_estimate**2)

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
#estimate per-dataset initial parameters via fast single-exponential pre-fit
def get_adaptive_initial_parameters(dataframes_dict, fit_function, base_parameters,
                                    x_col='t [s]', y_col='g(2)-1', verbose=True):
    if verbose:
        print(f"Estimating adaptive parameters for {fit_function.__name__}...")

    idx_map         = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5}
    adaptive_params = {}
    for name, df in dataframes_dict.items():
        try:
            x_data   = df[x_col].to_numpy()
            y_data   = df[y_col].to_numpy()
            estimates = estimate_parameters_from_data(x_data, y_data, base_parameters)
            updated  = base_parameters.copy()
            for pname, val in estimates.items():
                updated[idx_map[pname]] = val
            adaptive_params[name] = get_meaningful_parameters(fit_function, updated)
        except Exception as e:
            print(f"Warning: Failed to adapt parameters for {name} ({e}), using base parameters")
            adaptive_params[name] = get_meaningful_parameters(fit_function, base_parameters)

    if verbose:
        print(f"Generated adaptive parameters for {len(adaptive_params)} datasets")
    return adaptive_params

#fit and plot for cumulant-method C
#  adaptive mode  (initial_guesses is a dict)  → hierarchical zoom grid search
#  expert   mode  (initial_guesses is a list)  → direct iterative refinement from user parameters
def plot_processed_correlations_iterative(dataframes_dict, fit_function2, fit_x_limits, initial_guesses,
                                         max_iterations=10, tolerance=1e-4, maxfev=50000,
                                         method='lm', plot_number_start=1):

    #zoom rounds: coarse → medium → fine (adaptive mode only)
    _ZOOM_ROUNDS = [
        dict(half_decades=1.5, n_points=5, maxfev=1_000),
        dict(half_decades=0.5, n_points=5, maxfev=5_000),
        dict(half_decades=0.15, n_points=5, maxfev=50_000),
    ]

    all_fit_results = []
    plot_number     = plot_number_start
    param_names     = inspect.getfullargspec(fit_function2).args[1:]
    _is_adaptive    = isinstance(initial_guesses, dict)

    for name, df in dataframes_dict.items():
        fit_result = {'filename': name}
        try:
            x_data = df['t [s]']
            y_data = df['g(2)-1']

            x_fit = x_data[(x_data >= fit_x_limits[0]) & (x_data <= fit_x_limits[1])]
            y_fit = y_data[(x_data >= fit_x_limits[0]) & (x_data <= fit_x_limits[1])]

            if len(x_fit) < 2:
                print(f"Not enough data points in the specified range for fitting {name}. Skipping.")
                all_fit_results.append(fit_result)
                continue

            all_y_fits    = []
            all_r_squared = []

            #helpers scoped to this dataset
            def _metrics(popt):
                yc     = fit_function2(x_fit, *popt)
                res    = y_fit - yc
                ss_res = np.sum(res**2)
                ss_tot = np.sum((y_fit - np.mean(y_fit))**2)
                r2     = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                n      = len(y_fit)
                k      = len(popt) + 1
                return r2, float(np.sqrt(ss_res / n)), float(n * np.log(ss_res / n) + 2 * k)

            def _do_fit(p0, maxfev_override=None):
                mfev = maxfev_override if maxfev_override is not None else maxfev
                if method == 'lm':
                    return curve_fit(fit_function2, x_fit, y_fit, p0=p0, maxfev=mfev)
                bounds = ([_BOUNDS_LOWER[pn] for pn in param_names],
                          [_BOUNDS_UPPER[pn] for pn in param_names])
                return curve_fit(fit_function2, x_fit, y_fit, p0=p0,
                                 method=method, bounds=bounds, maxfev=mfev)

            best_popt = None
            best_pcov = None
            pcov      = None

            # ----------------------------------------------------------------
            # PATH A — Adaptive / Zoom mode
            # ----------------------------------------------------------------
            if _is_adaptive:
                guess    = initial_guesses[name].copy()
                b_center = float(guess[1])
                a_base   = float(guess[0])
                f_base   = float(guess[-1])
                c_base   = float(guess[param_names.index('c')]) if 'c' in param_names else None
                d_base   = float(guess[param_names.index('d')]) if 'd' in param_names else None
                e_base   = float(guess[param_names.index('e')]) if 'e' in param_names else None

                best_r2 = -np.inf
                prev_r2 = -np.inf

                for rnd in _ZOOM_ROUNDS:
                    b_grid        = b_center * np.logspace(-rnd['half_decades'], rnd['half_decades'], rnd['n_points'])
                    rnd_best_popt = None
                    rnd_best_pcov = None
                    rnd_best_r2   = -np.inf

                    for b_i in b_grid:
                        scale = b_i / b_center
                        p0 = []
                        for pn in param_names:
                            if   pn == 'a': p0.append(a_base)
                            elif pn == 'b': p0.append(float(b_i))
                            elif pn == 'c': p0.append(c_base * scale**2)
                            elif pn == 'd': p0.append(d_base * scale**3)
                            elif pn == 'e': p0.append(e_base * scale**4)
                            elif pn == 'f': p0.append(f_base)
                        try:
                            popt_i, pcov_i = _do_fit(p0, rnd['maxfev'])
                            r2_i, _, _     = _metrics(popt_i)
                            all_y_fits.append((popt_i, fit_function2(x_data, *popt_i)))
                            all_r_squared.append(r2_i)
                            if r2_i > rnd_best_r2:
                                rnd_best_r2   = r2_i
                                rnd_best_popt = popt_i
                                rnd_best_pcov = pcov_i
                        except RuntimeError:
                            continue  # skip failed grid point

                    if rnd_best_popt is None:
                        continue  # round failed — try next round with higher maxfev

                    if rnd_best_r2 > best_r2:
                        best_r2   = rnd_best_r2
                        best_popt = rnd_best_popt
                        best_pcov = rnd_best_pcov

                    #update base for next round from best solution found so far
                    b_center = float(best_popt[1])
                    a_base   = float(best_popt[0])
                    f_base   = float(best_popt[-1])
                    if 'c' in param_names: c_base = float(best_popt[param_names.index('c')])
                    if 'd' in param_names: d_base = float(best_popt[param_names.index('d')])
                    if 'e' in param_names: e_base = float(best_popt[param_names.index('e')])

                    if (best_r2 - prev_r2) < tolerance:
                        break  # no room for improvement
                    prev_r2 = best_r2

                if best_popt is None:
                    raise RuntimeError("All zoom grid points failed to converge.")
                pcov = best_pcov

            # ----------------------------------------------------------------
            # PATH B — Expert mode (user-supplied static parameters)
            # ----------------------------------------------------------------
            else:
                current_guess = get_meaningful_parameters(fit_function2, initial_guesses)

                #warn if user's b deviates from data estimate by more than 1 decade
                try:
                    _, b_pre, _ = _simple_exp_prefit(x_fit.to_numpy(), y_fit.to_numpy())
                    b_user = current_guess[1]
                    if b_pre > 0 and b_user > 0 and abs(np.log10(b_user) - np.log10(b_pre)) > 1.0:
                        print(f"[INFO] '{name}': provided b={b_user:.2e} deviates from "
                              f"estimated b={b_pre:.2e} by more than 1 decade.")
                except Exception:
                    pass

                for i in range(max_iterations):
                    try:
                        popt, pcov = _do_fit(current_guess)
                    except RuntimeError as e:
                        print(f"Fit error in iteration {i+1} for '{name}': {e}")
                        break

                    all_y_fits.append((popt, fit_function2(x_data, *popt)))
                    current_guess = popt

                    r2, rmse, aic = _metrics(popt)
                    all_r_squared.append(r2)
                    fit_result[f'R-squared_iter{i+1}'] = r2
                    fit_result[f'RMSE_iter{i+1}']      = rmse
                    fit_result[f'AIC_iter{i+1}']       = aic
                    for j, pn in enumerate(param_names):
                        fit_result[f'{pn}_iter{i+1}'] = popt[j]

                    if i > 0 and abs(all_r_squared[-1] - all_r_squared[-2]) < tolerance:
                        print(f"Converged after {i+1} iterations for {name}.")
                        break

            # ----------------------------------------------------------------
            # Common: select best result, store, plot
            # ----------------------------------------------------------------
            if not all_r_squared:
                raise RuntimeError("No successful fit attempts.")

            best_idx              = int(np.argmax(all_r_squared))
            best_popt, best_y_fit = all_y_fits[best_idx]
            best_r2_val           = all_r_squared[best_idx]

            yc_best        = fit_function2(x_fit, *best_popt)
            best_residuals = y_fit - yc_best
            ss_res_best    = np.sum(best_residuals**2)
            n              = len(y_fit)
            k              = len(best_popt) + 1

            fit_result['best_R-squared'] = best_r2_val
            fit_result['best_RMSE']      = float(np.sqrt(ss_res_best / n))
            fit_result['best_AIC']       = float(n * np.log(ss_res_best / n) + 2 * k)

            for j, pn in enumerate(param_names):
                pv = best_popt[j]
                fit_result['best_' + pn] = float(pv.item() if isinstance(pv, np.ndarray) and pv.size == 1 else pv)

            if pcov is not None:
                perr = np.sqrt(np.diag(pcov))
                for j, pn in enumerate(param_names):
                    if j < len(perr):
                        ev = perr[j]
                        fit_result['err_' + pn] = float(ev.item() if isinstance(ev, np.ndarray) and ev.size == 1 else ev)

            if fit_function2.__name__ == 'fit_function4':
                c_p = best_popt[2]
                e_p = best_popt[4]
                fit_result['kurtosis'] = float(e_p / c_p**2) if c_p != 0 else float('nan')

            all_fit_results.append(fit_result)

            #3-panel diagnostic plot: data+fit | residuals | Q-Q
            mode_str = (f'adaptive ({len(all_r_squared)} attempts)'
                        if _is_adaptive else f'expert (iter {best_idx+1})')
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
            fig.suptitle(f'[{plot_number}]: Method C ({fit_function2.__name__}) — {name}', fontsize=12)

            ax1.plot(x_data, y_data, 'o', alpha=0.6, markersize=4, label='Data')
            for i, (popt_i, y_i) in enumerate(all_y_fits):
                if i == best_idx:
                    ax1.plot(x_data, y_i, 'r-', linewidth=2, label=f'Best fit ({mode_str})')
                else:
                    ax1.plot(x_data, y_i, '-', color='gray', linewidth=1,
                             alpha=max(0.1, 0.3 + 0.5 * (i / len(all_y_fits))))
            ax1.set_xlabel(r'lag time τ [s]')
            ax1.set_ylabel(r'$g^{(2)}(\tau) - 1$')
            ax1.set_title('Data & Fit')
            ax1.grid(True, alpha=0.3)
            ax1.set_xscale('log')
            ax1.set_xlim(1e-6, 10)
            ax1.legend()
            ax1.text(0.95, 0.95,
                     f"R² = {best_r2_val:.4f}\n⟨Γ⟩ = {best_popt[1]:.3e} s⁻¹",
                     transform=ax1.transAxes, va='top', ha='right',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

            ax2.plot(best_residuals)
            ax2.axhline(0, color='r', linestyle='--', linewidth=1)
            ax2.set_xlabel('Sample Index')
            ax2.set_ylabel('Residuals')
            ax2.set_title('Residuals')
            ax2.grid(True, alpha=0.3)

            stats.probplot(best_residuals, dist="norm", plot=ax3)
            ax3.set_title('Q-Q Plot')
            ax3.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()
            plot_number += 1

        except RuntimeError as e:
            print(f"[WARNING] '{name}': fit failed — {e}. Filling with NaN.")
            fit_result['Error'] = str(e)
            for col in ['best_R-squared', 'best_RMSE', 'best_AIC']:
                fit_result.setdefault(col, np.nan)
            for pn in param_names:
                fit_result.setdefault('best_' + pn, np.nan)
                fit_result.setdefault('err_'  + pn, np.nan)
            all_fit_results.append(fit_result)

        except (KeyError, TypeError) as e:
            print(f"Error processing DataFrame '{name}': {e}")
            fit_result['Error'] = str(e)
            all_fit_results.append(fit_result)

    return pd.DataFrame(all_fit_results)