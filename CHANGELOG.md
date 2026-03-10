# CHANGELOG

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
