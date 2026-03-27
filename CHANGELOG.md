# CHANGELOG
## v2.2.0
improved cumulant_C fitting process, bug fixes and improved workflow
### preprocessing.py

- Added plot_countrate_fft(): plots the one-sided power spectral density (PSD) of the countrate time series for each file using the FFT. Displayed as a log-log plot (PSD = |FFT(I)|² · dt / N, units: kHz²/Hz). DC component (f = 0) skipped to allow log-scale display. All-zero and all-NaN detector columns silently skipped. Supports multi-column detector slots and the same ncols/show_indices layout as the existing countrate and correlation plot functions. Useful for identifying periodic noise, dust spikes or instrument artefacts in the countrate signal before fitting.

---

### cumulants_C.py

Major rewrite of the fitting engine in plot_processed_correlations_iterative(). The function now supports two distinct operating modes dispatched by the type of initial_guesses:

- **Adaptive mode** (initial_guesses is a dict): replaces the old individual/global/representative strategy system with a hierarchical log-uniform zoom grid search. Three successive rounds (coarse ±1.5 decades, medium ±0.5, fine ±0.15) each sweep b on a log-uniform grid of 5 points, scaling c, d, e to match (c ∝ b², d ∝ b³, e ∝ b⁴). The best result across all grid points propagates as the starting centre for the next round. Convergence exits early if R² improvement between rounds falls below tolerance.

- **Expert mode** (initial_guesses is a list): preserves the previous iterative single-start refinement from user-supplied parameters. Added a diagnostic warning when the user-supplied b deviates from the data estimate by more than 1 decade.

Additional changes to plot_processed_correlations_iterative():
- Physical parameter bounds _BOUNDS_LOWER / _BOUNDS_UPPER added as module-level constants and applied to trf/dogbox fits (c, e ≥ 0; d unconstrained; a ∈ [0, 1]; b ∈ [1, 1e6]; f ∈ [−0.1, 0.1]).
- _metrics() and _do_fit() defined as closures scoped per dataset to reduce code duplication.
- Graceful RuntimeError handling: failed fits now fill results with NaN and append an Error field instead of crashing the loop.
- Plot title now shows mode string (adaptive / expert) and number of fit attempts.

estimate_parameters_from_data() simplified: the long heuristic for a, b, f (1/e threshold, log-linear slope, curvature analysis) replaced by a call to the new _simple_exp_prefit(). Priors for c, d, e unchanged (c = 0.05·b², d = 0.01·b³, e = 3·c²).

New module-level helpers:
- _simple_exp(): 3-parameter single-exponential model used by the pre-fit.
- _simple_exp_prefit(): fast 3-parameter curve_fit to estimate a, b, f; falls back to safe data-derived heuristics on convergence failure.

Removed functions:
- get_adaptive_parameters_strategy() — dispatcher wrapper, superseded by mode dispatch in the main fitting function.
- _individual_strategy(), _global_strategy(), _representative_strategy() — all three strategy implementations removed. The individual strategy logic is now handled directly inside get_adaptive_initial_parameters(). Global and representative strategies are no longer supported.

get_adaptive_initial_parameters() simplified to a single individual pre-fit loop (one _simple_exp_prefit per file), removing the strategy dispatcher call and verbose parameter printing.

---

### clustering.py

- cluster_all_gammas(): monomodal z-score outlier detection is now gated behind if uncertainty_flags:. Previously, the z-score path ran whenever n_reliable_populations == 1 and need_silhouette == True, which could be triggered by clustering_strategy == 'silhouette_refined' alone even without uncertainty_flags=True. This caused unexpected interactive prompts during normal silhouette-refined runs on monomodal samples. The z-score outlier detection now only activates when the user explicitly sets uncertainty_flags=True.

### regularized.py

- nnls_reg(): weighted moments now computed in **log-τ space** instead of linear τ-space. Previously weighted_mean, variance, std_dev, skewness and kurtosis used arithmetic τ values directly (weighted_mean = Σ(w·τ)). Now ln_x = np.log(peak_x) is used throughout: ln_mean = Σ(w·ln_x), weighted_mean = exp(ln_mean) (geometric mean in τ-space), variance = Σ(w·(ln_x − ln_mean)²), with skewness and kurtosis following in log-τ space. This is physically correct since the decay-time grid is log-spaced and area is integrated via d(ln τ); the old arithmetic centroid was biased toward the right tail on a log grid.

- nnls_reg(): centroid marker and annotation in the distribution plot now use f_display = np.interp(tau_display, decay_times, f_optimized) when peak_method == 'centroid', instead of always using if_optimized[peak_idx]. Previously the marker and annotation xy coordinate were always anchored at the amplitude of the peak maximum index regardless of peak_method, causing visual misalignment when the geometric-mean centroid differs from the maximum position.


## v2.1.0
added intensity and guinier analysis for regularized fit to Jupyter Notebook-File

### New Files

- **sls_functions_for_regularized.py** — Static light scattering (SLS) analysis functions for population-resolved intensity analysis from regularized DLS fits. Decomposes total scattering intensity into per-population contributions using area fractions from the regularized fit, applies number-weighting correction to remove Rh-dependent intensity bias, and performs Guinier analysis per population and on the total intensity. Includes: compute_sls_data(), compute_sls_data_number_weighted(), compute_guinier_total(), compute_guinier_extrapolation(), plot_sls_intensity(), plot_guinier(), summarize_sls(), summarize_sls_combined(), and internal helpers _style_ax(), _guinier_fit().

---

### clustering.py

- cluster_all_gammas(): added per-peak column remapping block. After assigning gamma values to population columns, per-peak statistics (tau, intensity, normalized_area_percent, normalized_sum_percent, area, fwhm, centroid, std_dev, skewness, kurtosis) are now remapped from their original peak-index columns (e.g. skewness_3) to population-indexed columns (e.g. skewness_pop2) using the cluster_id_to_pop mapping. Previously these columns were only available in raw peak-index form and were not aligned to population order.

- aggregate_peak_stats(): fixed indentation bug where pop_num, pop_means[pop_num] and pop_stds[pop_num] assignments ran outside the inner for _, row in cluster_rows.iterrows() loop, causing only the last row of each cluster to contribute to the population statistics. De-indented by one level so all rows in a cluster are correctly accumulated before computing the mean and std.

- _show_uncertainty_removal_preview(): fixed KeyError: 'silhouette' crash in the monomodal path. The function unconditionally read gammas_df['silhouette'], but the silhouette column is only added in the multimodal path (n_reliable_populations > 1). The monomodal path uses z-scores instead. Fixed by checking which column is present (silhouette vs z_score) and displaying the appropriate metric. Also guarded the per-population breakdown against the monomodal case where no cluster column exists.

---

### regularized.py

- nnls_reg(): fixed calculation for normalized_area_percent per detected peak (area-fraction relative to total detected peak area, in %). Fixed normalized_sum_percent per detected peak (amplitude-fraction relative to total peak amplitude sum, in %). Both are stored in results and are used by sls_functions_for_regularized.py for intensity decomposition.

  
## v2.0.0
complete Jupyter Notebook-File reworked

### New Files

- **noise.py** - Baseline and noise correction for autocorrelation functions. Provides apply_noise_corrections() (baseline subtraction, intercept flattening) and plot_correction_sample() (two-panel before/after overview coloured by angle).

- **cumulants_D.py** - Method D: multi-exponential Dirac-delta decomposition of g2(t). Iteratively fits increasing numbers of exponential modes, selects the optimal model order by residual improvement, clusters fitted decay rates into physical populations, and computes distribution moments (mean, PDI, skewness, kurtosis). Includes dirac_sum_g1, dirac_sum_g2, fit_cumulant_D, calculate_moments_from_gammas, cluster_gammas, fit_correlations_method_D.

- **clustering.py** - Cross-file population clustering for multi-angle data. Groups decay rates from all files into discrete populations using hierarchical or threshold-based agglomerative clustering, applies abundance filtering, assigns reliability flags, and aggregates per-peak statistics per population. Includes cluster_all_gammas, get_reliable_gamma_cols, aggregate_peak_stats and several internal helpers.

---

### preprocessing.py

- plot_countrates(): axis labels hardcoded to time and countrate units (previously derived from column name; capitalisation fixed).
- plot_correlations(): axis labels hardcoded to time [ms] and g(2)-1 (previously derived from column name).
- process_correlation_data(): output column names changed from t(s) to t[s] and from g(2) to g(2)-1 for consistency with all downstream modules.

---

### cumulants.py

- calculate_g2_B(): input column changed from g(2) to g(2)-1; transformation changed from sqrt(g2) to 0.5*ln(g2-1), matching the linear form required by Method B.
- plot_processed_correlations() (Method B): completely rewritten. Replaced nonlinear curve_fit with scipy.stats.linregress on ln(sqrt(g2-1)) vs tau. Column references updated. Added xlim/ylim parameters. Expanded to a 3-panel layout (data+fit | residuals | Q-Q). Added a None-guard to skip missing DataFrames gracefully.
- analyze_diffusion_coefficient(): added experiment_name parameter appended to plot title. Added NaN-filtering for multimodal data. Fixed matplotlib string interpolation issue in plot title.
- Removed calculate_cumulant_results_A() and create_zero_cumulant_results_A(). Method A result assembly is now handled directly in the notebook.
- Added module-level docstring.

---

### cumulants_C.py

- get_meaningful_parameters(): added case for fit_function1 (1st-order fit, returns [a, b, f]).
- plot_processed_correlations_iterative(): added plot_number_start parameter for continuous plot numbering. Column references updated to t[s] and g(2)-1. Expanded to a 3-panel layout (data+fit iterations | residuals | Q-Q). Added kurtosis output for fit_function4 (e/c^2, stored as fit_result[kurtosis]). Fixed module docstring: best-iteration selection is by R2, not RMSE.
- get_adaptive_parameters_strategy() and all strategy helpers: default column names updated.
- Added module-level docstring.

---

### intensity.py

- No functional changes. Carried over unchanged from v1.

---

### regularized.py

- Added module-level docstring.
- nnls(): column references updated to t[s] and g(2)-1. Expanded to a 4-panel layout (data+fit | distribution | residuals | Q-Q).
- nnls_reg(): expanded to a 4-panel layout. Added computation of area fractions, centroids, FWHM, skewness and excess kurtosis per detected peak.
- nnls_reg_all(), nnls_reg_simple(), analyze_random_datasets_grid(), tau_to_hydrodynamic_radius(), find_dataset_key(), plot_distributions(): carried over from v1 with minor updates for consistency with new column naming.
