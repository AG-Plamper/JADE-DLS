import marimo

__generated_with = "0.23.10"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # JADE-DLS: Jupyter-based Angular Dependent Evaluator for Dynamic Light Scattering
    ### [vers. 2.3 — marimo port]

    Ported from `JADE-DLS_workingvariant.ipynb`.

    Changes made to satisfy marimo's one-definition-per-global rule:
    - Repeated per-method config variable names (e.g. `remove_bad_fits`, `fname`,
      `fit_x_range`, clustering thresholds) are now suffixed per method (`_A`, `_B`, `_nnls`, ...).
    - Repeated per-method loop/plot logic (D/Rh computation, file extraction) was
      moved into small shared helper functions so their internals don't leak as globals.
    - The 5 blocking `input("Enter row indices to remove...")` prompts became
      reactive `mo.ui.text` boxes — type indices, downstream cells recompute automatically.
    - `IPython.display.display(df)` calls became plain last-line expressions
      (marimo auto-renders the last expression of a cell, same as Jupyter).
    """)
    return


@app.cell
def _():
    import glob
    import os
    import sys
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    if sys.version_info < (3, 8):
        raise RuntimeError("Python 3.8 or higher required!")

    #force UTF-8 stdout/stderr: `marimo run` on Windows can inherit a cp1252
    #console codec, and this notebook prints Greek letters/superscripts (Γ, ⁻, °, ±, √)
    #everywhere. An encoding error in one print() cascades and blocks every
    #downstream cell — this is why results could disappear outside `marimo edit`.
    for _stream in (sys.stdout, sys.stderr):
        if hasattr(_stream, 'reconfigure'):
            _stream.reconfigure(encoding='utf-8', errors='replace')

    print("Ready to analyze!")
    return glob, np, os, pd, plt


@app.cell
def _():
    from scipy.constants import k as k_B
    from preprocessing import (get_folder_name, extract_data, extract_countrate, extract_correlation,
                                plot_countrates, plot_countrate_fft, plot_correlations,
                                remove_from_data, remove_dataframes, process_correlation_data)
    from intensity import extract_intensity, plot_meancr
    from noise import apply_noise_corrections, plot_correction_sample
    from cumulants import (extract_cumulants, analyze_diffusion_coefficient,
                            calculate_g2_B, plot_processed_correlations)
    from cumulants_C import (get_adaptive_initial_parameters, get_meaningful_parameters,
                              plot_processed_correlations_iterative)
    from cumulants_D import fit_correlations_method_D
    from clustering import cluster_all_gammas, get_reliable_gamma_cols, aggregate_peak_stats
    from regularized import (nnls_all, nnls_reg_all, calculate_decay_rates,
                              nnls_reg_simple, analyze_random_datasets_grid, plot_distributions)
    from sls_functions_for_regularized import (compute_sls_data, compute_guinier_total, compute_guinier_extrapolation,
                                                plot_sls_intensity, plot_guinier, summarize_sls_combined)

    return (
        aggregate_peak_stats,
        analyze_diffusion_coefficient,
        analyze_random_datasets_grid,
        apply_noise_corrections,
        calculate_decay_rates,
        calculate_g2_B,
        cluster_all_gammas,
        compute_guinier_extrapolation,
        compute_guinier_total,
        compute_sls_data,
        extract_correlation,
        extract_countrate,
        extract_cumulants,
        extract_data,
        extract_intensity,
        fit_correlations_method_D,
        get_adaptive_initial_parameters,
        get_folder_name,
        get_meaningful_parameters,
        get_reliable_gamma_cols,
        k_B,
        nnls_all,
        nnls_reg_all,
        nnls_reg_simple,
        plot_correction_sample,
        plot_correlations,
        plot_countrate_fft,
        plot_countrates,
        plot_distributions,
        plot_guinier,
        plot_meancr,
        plot_processed_correlations,
        plot_processed_correlations_iterative,
        plot_sls_intensity,
        process_correlation_data,
        remove_from_data,
        summarize_sls_combined,
    )


@app.cell
def _(cluster_all_gammas, mo, np, os, pd, plt):
    def extract_dataframe(files, extractor_fn):
        """Run extractor_fn(file) over files, concat non-None results into one DataFrame."""
        rows = []
        n_failed = 0
        for f in files:
            extracted = extractor_fn(f)
            if extracted is not None:
                extracted = extracted.copy()
                extracted['filename'] = os.path.basename(f)
                rows.append(extracted)
            else:
                n_failed += 1
        if not rows:
            raise ValueError("NO data extracted!")
        df = pd.concat(rows, ignore_index=True)
        df.index = df.index + 1
        return df, n_failed


    def extract_dict(files, extractor_fn, column_names=None):
        """Run extractor_fn(file) over files, return {basename: (renamed) df}."""
        result = {}
        n_failed = 0
        for f in files:
            extracted = extractor_fn(f)
            if extracted is not None:
                name = os.path.basename(f)
                if column_names:
                    extracted = extracted.rename(columns=column_names)
                result[name] = extracted
            else:
                n_failed += 1
        return result, n_failed


    def rh_from_slope(q2_coef, q2_se, c, delta_c, unit_factor=1e-18):
        """D [m^2/s] and Rh [nm] (Stokes-Einstein) from a Gamma-vs-q^2 regression slope."""
        D = q2_coef * unit_factor
        D_err = q2_se * unit_factor
        Rh = c / D * 1e9
        Rh_err = np.sqrt((delta_c / c) ** 2 + (D_err / D) ** 2) * Rh
        return D, D_err, Rh, Rh_err


    def exclude_by_index(data_dict, indices_str):
        """Drop entries from a {name: df} dict by position, given a comma-separated
        index string (indices match the [N] shown in plot_countrates/plot_correlations
        titles with show_indices=True). Replaces the original's input()-based
        cli_countrate_exclusion/cli_correlation_exclusion, which can't run in marimo."""
        if not indices_str.strip():
            return data_dict
        names = list(data_dict.keys())
        try:
            excluded_idx = [int(i.strip()) for i in indices_str.split(',')]
        except ValueError as e:
            print(f"Invalid input ({e}); using all datasets.")
            return data_dict
        excluded_names = [names[i] for i in excluded_idx if 0 <= i < len(names)]
        if not excluded_names:
            print("No valid exclusions. Using all datasets.")
            return data_dict
        print(f"Excluded {len(excluded_names)} dataset(s): {', '.join(excluded_names)}")
        return {k: v for k, v in data_dict.items() if k not in excluded_names}


    def make_fit_range_ui(label, default_range, restrict_default, slider_bounds=(0, 0.002), step=0.00005):
        """A checkbox + range_slider pair for a q^2 fit-range toggle, used identically
        across all six regression cells (A/B/C/D/NNLS/Reg)."""
        lo, hi = slider_bounds
        ui_restrict = mo.ui.checkbox(value=restrict_default, label=f"Restrict {label} fit range")
        ui_range = mo.ui.range_slider(lo, hi, step=step, value=default_range or (lo, hi),
                                       label=f"{label}: q² fit range [nm⁻²]", show_value=True)
        return ui_restrict, ui_range


    def make_clustering_ui(label, distance_threshold_default, min_abundance_default, clustering_strategy_default):
        """Checkbox/slider/dropdown set for one method's clustering settings, used
        identically across the three multi-mode methods (D/NNLS/Reg)."""
        ui_enable = mo.ui.checkbox(value=True, label="Enable clustering")
        ui_normalize = mo.ui.checkbox(value=True, label="Normalize by q² (cluster on D)")
        ui_uncertainty = mo.ui.checkbox(value=False, label="Flag uncertain assignments")
        ui_distance = mo.ui.slider(0.3, 3.0, step=0.1, value=distance_threshold_default,
                                    label=f"{label}: distance threshold", show_value=True)
        ui_abundance = mo.ui.slider(0.1, 0.9, step=0.05, value=min_abundance_default,
                                     label=f"{label}: min abundance", show_value=True)
        ui_strategy = mo.ui.dropdown(options=['simple', 'silhouette_refined'],
                                      value=clustering_strategy_default, label=f"{label}: strategy")
        layout = mo.vstack([
            mo.hstack([ui_enable, ui_normalize, ui_uncertainty]),
            mo.hstack([ui_distance, ui_abundance, ui_strategy]),
        ])
        return layout, ui_enable, ui_normalize, ui_uncertainty, ui_distance, ui_abundance, ui_strategy


    #grid used by each method's own clustering-sensitivity sweep, below in its section
    distance_thresholds = [0.3, 0.7, 1.0, 1.5, 2.0]
    min_abundances       = [0.1, 0.2, 0.3, 0.4, 0.5]


    def clustering_sensitivity_sweep(df, gamma_cols, q_squared_col, clustering_strategy,
                                      distance_thresholds, min_abundances):
        """Run cluster_all_gammas over a (distance_threshold, min_abundance) grid,
        recording how many populations are found and how separated they are."""
        rows = []
        for dt in distance_thresholds:
            for ma in min_abundances:
                plt.ioff()
                _, info = cluster_all_gammas(
                    df,
                    gamma_cols            = gamma_cols,
                    q_squared_col         = q_squared_col,
                    enable_clustering     = True,
                    normalize_by_q2       = True,
                    n_clusters            = 'auto',
                    distance_threshold    = dt,
                    min_abundance         = ma,
                    clustering_strategy   = clustering_strategy,
                    uncertainty_flags     = False,
                    plot                  = False
                )
                plt.close('all')
                plt.ion()
                rows.append({
                    'distance_threshold': dt,
                    'min_abundance'     : ma,
                    'n_populations'     : info['n_populations'],
                    'silhouette'        : info.get('silhouette_score', np.nan),
                })
        return pd.DataFrame(rows)


    def plot_clustering_heatmaps(df_summary, title):
        """Two heatmaps side by side: number of populations found, and
        silhouette score, over the (distance_threshold, min_abundance) grid."""
        import matplotlib.gridspec as gridspec

        pivot_n = df_summary.pivot(index='distance_threshold', columns='min_abundance', values='n_populations')
        pivot_s = df_summary.pivot(index='distance_threshold', columns='min_abundance', values='silhouette')

        fig = plt.figure(figsize=(11, 4.5))
        gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.45)
        ax_heat = fig.add_subplot(gs[0])
        ax_sil = fig.add_subplot(gs[1])

        im1 = ax_heat.imshow(pivot_n.values, aspect='auto', cmap='YlOrRd_r', origin='lower')
        ax_heat.set_xticks(range(len(pivot_n.columns)))
        ax_heat.set_xticklabels([f'{v:.1f}' for v in pivot_n.columns], fontsize=9)
        ax_heat.set_yticks(range(len(pivot_n.index)))
        ax_heat.set_yticklabels([f'{v:.1f}' for v in pivot_n.index], fontsize=9)
        ax_heat.set_xlabel(r'min abundance ($f_{min}$)', fontsize=11)
        ax_heat.set_ylabel(r'distance threshold ($d_{thr}$)', fontsize=11)
        ax_heat.set_title(f'{title} — No. of populations', fontsize=11, fontweight='bold', loc='left')
        for i in range(len(pivot_n.index)):
            for j in range(len(pivot_n.columns)):
                val = pivot_n.values[i, j]
                if not np.isnan(val):
                    ax_heat.text(j, i, f'{int(val)}', ha='center', va='center', fontsize=11, color='black')
        plt.colorbar(im1, ax=ax_heat, label='n populations')

        im2 = ax_sil.imshow(pivot_s.values, aspect='auto', cmap='YlGn', origin='lower', vmin=0.5, vmax=1.0)
        ax_sil.set_xticks(range(len(pivot_s.columns)))
        ax_sil.set_xticklabels([f'{v:.1f}' for v in pivot_s.columns], fontsize=9)
        ax_sil.set_yticks(range(len(pivot_s.index)))
        ax_sil.set_yticklabels([f'{v:.1f}' for v in pivot_s.index], fontsize=9)
        ax_sil.set_xlabel(r'min abundance $f_{min}$', fontsize=11)
        ax_sil.set_ylabel(r'distance threshold $d_{thr}$', fontsize=11)
        ax_sil.set_title(f'{title} — Silhouette score', fontsize=11, fontweight='bold', loc='left')
        for i in range(len(pivot_s.index)):
            for j in range(len(pivot_s.columns)):
                val = pivot_s.values[i, j]
                if not np.isnan(val):
                    ax_sil.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=11, color='black')
        plt.colorbar(im2, ax=ax_sil, label='Silhouette score')

        plt.tight_layout()
        return fig


    def select_fits_table(df, show_cols, label):
        """Interactive row-selection table: every fit runs unconditionally, and
        this shows show_cols for review while keeping every column of df
        reachable through .value for whichever rows stay selected. All rows
        start selected (kept) — deselect the bad ones instead of typing indices."""
        hidden = [c for c in df.columns if c not in show_cols]
        return mo.ui.table(df, selection='multi', initial_selection=list(range(len(df))),
                            hidden_columns=hidden, label=label, page_size=10)


    def resolve_selection(table, full_df, label):
        """The selected subset of a select_fits_table, or the full df (with a
        warning) if the user deselected everything."""
        selected = table.value
        if len(selected) == 0:
            print(f"[!] No rows selected for {label} — keeping all rows to avoid an empty dataset.")
            return full_df
        return selected


    def maybe_export(df, do_export, fname):
        if do_export:
            df.to_csv(fname, sep='\t', index=False)
            print(f"Data exported to {fname}")

    return (
        clustering_sensitivity_sweep,
        distance_thresholds,
        exclude_by_index,
        extract_dataframe,
        extract_dict,
        make_clustering_ui,
        make_fit_range_ui,
        maybe_export,
        min_abundances,
        plot_clustering_heatmaps,
        resolve_selection,
        rh_from_slope,
        select_fits_table,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### I. PREPROCESSING
    """)
    return


@app.cell
def _(mo):
    ui_folder_browser = mo.ui.file_browser(
        initial_path=r"C:/Users/vince/Documents/DLS",
        selection_mode="directory",
        multiple=False,
        label="Browse for a data folder (overrides the pasted path below if you pick one)",
    )
    ui_folder_browser
    return (ui_folder_browser,)


@app.cell
def _(mo):
    ui_datafolder = mo.ui.text(
        value=r"C:/Users/vince/Documents/DLS/goldparticles/2_80-30nm+15-200nm/*.asc",
        label="...or paste a path here (with or without \\*.asc, backslash or forward slash, quotes are fine)",
        full_width=True,
    )
    ui_experiment_name = mo.ui.text(
        value="", label="Experiment name (blank = auto from folder name)"
    )
    mo.vstack([ui_datafolder, ui_experiment_name])
    return ui_datafolder, ui_experiment_name


@app.cell
def _(
    get_folder_name,
    glob,
    os,
    ui_datafolder,
    ui_experiment_name,
    ui_folder_browser,
):
    #pull the folder + name from the widgets above, then validate.
    #prefer a folder picked via the browser; otherwise fall back to the pasted text.
    #normalize the pasted path: Windows "Copy as path" adds quotes and backslashes,
    #and pasting a bare folder (no \*.asc) is a common paste - handle all three.
    if ui_folder_browser.value:
        datafolder = str(ui_folder_browser.value[0].path).replace('\\', '/').rstrip('/') + '/*.asc'
    else:
        datafolder = ui_datafolder.value.strip().strip('"').strip("'").replace('\\', '/')
        if '*' not in datafolder:
            datafolder = datafolder.rstrip('/') + '/*.asc'
    experiment_name = ui_experiment_name.value

    base_path = datafolder.rsplit('/', 1)[0]
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"directory not found: {base_path}")

    #preview
    preview_files = glob.glob(datafolder)
    n_files = len(preview_files)

    if n_files == 0:
        raise FileNotFoundError(f"no .asc files found in: {base_path}")

    # Set experiment name
    experiment_name = experiment_name or get_folder_name(datafolder)

    print(f"Experiment: {experiment_name}")
    print(f"Location: {base_path}")
    print(f"Files found: {n_files} .asc files")
    return datafolder, experiment_name, n_files


@app.cell
def _(datafolder, glob, os):
    #choose between method of catching datafiles
    #datafiles = glob.glob(os.path.join(datafolder, '*'))
    datafiles = glob.glob(datafolder)

    #excluding the *_average.asc-files [for ALV-Input-data]
    filtered_files = [f for f in datafiles
                      if "averaged" not in os.path.basename(f).lower()]

    n_excluded = len(datafiles) - len(filtered_files)
    print(f"collected {len(filtered_files)} files" +
          (f" (excluded {n_excluded} averaged)" if n_excluded > 0 else ""))

    #basename -> full path, needed by several extraction steps below
    file_to_path = {os.path.basename(f): f for f in filtered_files}
    return file_to_path, filtered_files


@app.cell
def _(extract_data, extract_dataframe, filtered_files, np):
    #base-data extraction
    #extract metadata: angle, temperature, wavelength, refractive index, viscosity
    #calculate q and q^2

    print("Extracting metadata from files...")

    df_basedata, n_failed_meta = extract_dataframe(filtered_files, extract_data)

    # Calculate scattering vector q and q^2
    # q = (4*pi*n/lambda) * sin(theta/2), where n=refractive index, lambda=wavelength, theta=angle
    df_basedata['q'] = abs(
        (4 * np.pi * df_basedata['refractive_index']) /
        df_basedata['wavelength [nm]'] *
        np.sin(np.radians(df_basedata['angle [°]']) / 2)
    )
    df_basedata['q^2'] = df_basedata['q'] ** 2

    print(f"successfully extracted: {len(df_basedata)} files")
    if n_failed_meta > 0:
        print(f"failed to extract: {n_failed_meta} files")
    return (df_basedata,)


@app.cell
def _(df_basedata, k_B, np, pd):
    #consistency-check: checks extracted metadata for physical plausibility and internal consistency across all measurement files.
    #precomputes c = kT/(6*pi*eta),

    df_check = df_basedata.copy()

    warnings_issued = 0

    #instrument parameters: wavelength and refractive index these should be identical across all files
    unique_wavelengths = df_check['wavelength [nm]'].unique()
    unique_ri = df_check['refractive_index'].unique()

    if len(unique_wavelengths) > 1:
        print(f"[!] multiple wavelengths detected: {unique_wavelengths} nm")
        warnings_issued += 1
    else:
        print(f"[ok] wavelength:        {unique_wavelengths[0]:.1f} nm (uniform)")

    if len(unique_ri) > 1:
        print(f"[!] multiple refractive indices detected: {unique_ri}")
        warnings_issued += 1
    else:
        print(f"[ok] refractive index:  {unique_ri[0]:.4f} (uniform)")

    #temperature: small spread expected for a single experiment
    mean_T   = df_check['temperature [K]'].mean()
    std_T    = df_check['temperature [K]'].std()
    sem_T    = df_check['temperature [K]'].sem()
    range_T  = df_check['temperature [K]'].max() - df_check['temperature [K]'].min()

    print(f"\n  temperature:        {mean_T:.2f} ± {std_T:.2f} K  "
          f"(range: {range_T:.2f} K, n={len(df_check)})")
    if range_T > 1.0:
        print(f"[!] temperature range > 1 K.")
        warnings_issued += 1
    else:
        print(f"[ok] temperature spread < 1 K.")

    #viscosity: should be consistent if T is consistent
    mean_eta  = df_check['viscosity [cp]'].mean()
    std_eta   = df_check['viscosity [cp]'].std()
    sem_eta   = df_check['viscosity [cp]'].sem()
    range_eta = df_check['viscosity [cp]'].max() - df_check['viscosity [cp]'].min()

    print(f"\n  viscosity:          {mean_eta:.4f} ± {std_eta:.4f} cP  "
          f"(range: {range_eta:.4f} cP)")
    if std_eta / mean_eta > 0.01:  # >1% relative spread
        print(f"[!] viscosity spread > 1% relative.")
        warnings_issued += 1
    else:
        print(f"[ok] viscosity spread < 1% relative.")

    #angle coverage
    n_angles = df_check['angle [°]'].nunique()
    angle_list = sorted(df_check['angle [°]'].unique())
    print(f"\n  angles measured:    {angle_list}")
    if n_angles < 5:
        print(f"[!] only {n_angles} angles available.")
        warnings_issued += 1
    else:
        print(f"[ok] {n_angles} angles — sufficient angles available.")

    #q^2 range
    q2_min = df_check['q^2'].min()
    q2_max = df_check['q^2'].max()
    print(f"\n  q² range:           {q2_min:.4e} – {q2_max:.4e} nm⁻²")

    #summary
    print("\n" + "=" * 55)
    if warnings_issued == 0:
        print("data look consistent.")
    else:
        print(f"  {warnings_issued} warning(s) issued — review before proceeding.")

    #summary dataframe
    df_basedata_stats = pd.DataFrame({
        'mean temperature [K]':  [mean_T],
        'std temperature [K]':   [std_T],
        'sem temperature [K]':   [sem_T],
        'mean viscosity [cp]':   [mean_eta],
        'std viscosity [cp]':    [std_eta],
        'sem viscosity [cp]':    [sem_eta]
    })

    #precompute c = kT / (6*pi*eta) and propagate uncertainty
    #uncertainty propagated in quadrature from T and eta (std used as measure of experimental spread; if all files are from one sample, std reflects real instrument uncertainty rather than sample variability)

    c = (k_B * mean_T) / (6 * np.pi * mean_eta * 1e-3)  #[m^2/s * Pa*s = m^3 -> m^2/s, converted to SI]

    fractional_error_c = np.sqrt(
        (std_T   / mean_T)**2 +
        (std_eta / mean_eta)**2)
    delta_c = fractional_error_c * c
    relative_error_c = delta_c / c

    print(f"\n  c  =  {c:.4e}  ±  {delta_c:.4e}  m³/s")
    print(f"  relative uncertainty in c:  {relative_error_c:.3%}")
    if relative_error_c > 0.02:
        print(f"  [!] relative uncertainty in c > 2%.")
    return c, delta_c, df_check


@app.cell
def _(mo):
    #extract scattering intensity (not required for DLS)
    ui_perform_intensity = mo.ui.checkbox(value=True, label="Extract scattering intensity (not required for DLS)")
    ui_perform_intensity
    return (ui_perform_intensity,)


@app.cell
def _(
    extract_dataframe,
    extract_intensity,
    filtered_files,
    ui_perform_intensity,
):
    perform_intensity_processing = ui_perform_intensity.value

    if perform_intensity_processing:

        print("Extracting scattering intensity data...")
        df_intensity, n_failed_intensity = extract_dataframe(filtered_files, extract_intensity)

        print(f"successfully extracted: {len(df_intensity)} files")
        if n_failed_intensity > 0:
            print(f"failed to extract: {n_failed_intensity} files")
    else:
        print("intensity processing disabled")
        df_intensity = None
    return df_intensity, perform_intensity_processing


@app.cell
def _(df_intensity, np, perform_intensity_processing, plot_meancr):
    if perform_intensity_processing and df_intensity is not None:

        #calculate angle-corrected intensity
        #correction factor: sin(theta) accounts for scattering geometry
        df_intensity['MeanCR_corr [kHz]'] = (
            (df_intensity['meancr0 [kHz]'] + df_intensity['meancr1 [kHz]']) / 2 /
            (df_intensity['monitordiode [cps]'] * 10**(-3)) *
            np.sin(np.radians(df_intensity['angle [°]'])))

        #plot intensity vs angle
        plot_meancr(df_intensity, 'angle [°]', 'MeanCR_corr [kHz]')

        #intensity at 90° (reference angle)
        intensity_90 = df_intensity[df_intensity['angle [°]'] == 90]['MeanCR_corr [kHz]']
        if not intensity_90.empty:
            print(f"intensity at 90°: {intensity_90.mean():.3f} ± {intensity_90.std():.3f} kHz")

        #angular range
        print(f"angle range: {df_intensity['angle [°]'].min():.0f}° - {df_intensity['angle [°]'].max():.0f}°")
        print(f"intensity range: {df_intensity['MeanCR_corr [kHz]'].min():.2f} - {df_intensity['MeanCR_corr [kHz]'].max():.2f} kHz")
    return


@app.cell
def _(extract_countrate, extract_dict, filtered_files):
    #extracts the time-resolved countrate traces from each file.
    #in this setup: 4 detector slots, but slots 1&3 and 2&4 are physically the same detector (cross-correlation mode).

    countrate_column_names = {0: 'time [s]',
                               1: 'detectorslot 1',
                               2: 'detectorslot 2',
                               3: 'detectorslot 3',
                               4: 'detectorslot 4'}

    all_countrates, n_failed_cr = extract_dict(filtered_files, extract_countrate, countrate_column_names)

    print(f"Countrate extraction complete: {len(all_countrates)} successful"
          + (f", {n_failed_cr} failed" if n_failed_cr > 0 else ""))
    return (all_countrates,)


@app.cell
def _(mo):
    #optionally plots the full traces for visual quality inspection.
    ui_plot_countrate_graphs = mo.ui.checkbox(value=False, label="Inspect full countrate traces")
    ui_countrate_fft = mo.ui.checkbox(value=False, label="Also show FFT view")
    mo.hstack([ui_plot_countrate_graphs, ui_countrate_fft])
    return ui_countrate_fft, ui_plot_countrate_graphs


@app.cell
def _(
    all_countrates,
    plot_countrate_fft,
    plot_countrates,
    ui_countrate_fft,
    ui_plot_countrate_graphs,
):
    plot_countrate_graphs = ui_plot_countrate_graphs.value
    countrate_fft = ui_countrate_fft.value

    if plot_countrate_graphs:
        if countrate_fft:
            plot_countrate_fft(all_countrates)
        plot_countrates(all_countrates, show_indices=True)
    else:
        print("countrate plotting skipped.")
    return


@app.cell
def _(mo):
    #exclude countrate datasets by the index shown in the plot titles above
    ui_exclude_countrate = mo.ui.text(
        value="", label="Countrate dataset indices to EXCLUDE (comma-separated, blank = keep all)")
    ui_exclude_countrate
    return (ui_exclude_countrate,)


@app.cell
def _(all_countrates, exclude_by_index, ui_exclude_countrate):
    all_countrates_final = exclude_by_index(all_countrates, ui_exclude_countrate.value)
    print(f"{len(all_countrates_final)} of {len(all_countrates)} countrate dataset(s) in use.")
    return (all_countrates_final,)


@app.cell
def _(
    all_countrates_final,
    df_check,
    df_intensity,
    experiment_name,
    np,
    pd,
    perform_intensity_processing,
    plt,
):
    #calculates mean CR per file from the raw countrate traces, applies the sin(theta) geometry correction (but no monitordiode!), and plots the result vs angle.
    #if perform_intensity_processing = True, overlays the software-extracted intensity for direct comparison.
    cr_rows = []
    for _filename, df in all_countrates_final.items():
        mean_cr = (df['detectorslot 1'].mean() + df['detectorslot 2'].mean()) / 2
        cr_rows.append({'filename': _filename, 'MeanCR_raw [kHz]': mean_cr})
    df_cr = pd.DataFrame(cr_rows)
    df_cr = pd.merge(df_cr, df_check[['filename', 'angle [°]']], on='filename', how='left')
    df_cr = df_cr.sort_values('angle [°]').reset_index(drop=True)
    #merge with basedata to get angles
    df_cr['MeanCR_corr [kHz]'] = df_cr['MeanCR_raw [kHz]'] * np.sin(np.radians(df_cr['angle [°]']))
    if perform_intensity_processing and df_intensity is not None:
        df_intensity_plot = df_intensity[df_intensity['filename'].isin(all_countrates_final.keys())].copy()
        df_cr_merged = pd.merge(df_cr, df_intensity_plot[['filename', 'monitordiode [cps]']], on='filename', how='left')
        df_cr['MeanCR_norm [kHz]'] = df_cr_merged['MeanCR_raw [kHz]'] / (df_cr_merged['monitordiode [cps]'] * 0.001) * np.sin(np.radians(df_cr['angle [°]']))
    #sin(theta) geometry correction
    fig_cr, ax_cr = plt.subplots(figsize=(8, 4))
    if perform_intensity_processing and df_intensity is not None:
    #if intensity data available: filter to surviving files, then apply monitordiode normalisation to make both curves directly comparable on the same scale
        ax_cr.scatter(df_cr['angle [°]'], df_cr['MeanCR_norm [kHz]'], alpha=0.7, s=50, label='from countrate traces', color='steelblue')
        ax_cr.scatter(df_intensity_plot.sort_values('angle [°]')['angle [°]'], df_intensity_plot.sort_values('angle [°]')['MeanCR_corr [kHz]'], alpha=0.7, s=50, label='from intensity extraction', color='tomato')
        ax_cr.legend()
        print('both intensity sources plotted.')
    else:
        print('only calculated intensity from extracted countrate plotted.')
        ax_cr.scatter(df_cr['angle [°]'], df_cr['MeanCR_corr [kHz]'], alpha=0.7, s=50, color='steelblue')
    ax_cr.set_xlabel('angle [°]')
    ax_cr.set_ylabel('corrected mean intensity [kHz]')
    #plot
    ax_cr.set_title(experiment_name)
    ax_cr.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_cr
    return


@app.cell
def _(all_countrates_final, extract_correlation, extract_dict, file_to_path):
    #extracts g(2)-1 correlation functions for all files that survived countrate filtering.
    #4 correlation channels are extracted. In cross-correlation mode, channels 1&2 are the relevant pair.
    #-> channels 3&4 may be zero depending on measurement settings.
    correlation_column_names = {0: 'time [ms]', 1: 'correlation 1', 2: 'correlation 2', 3: 'correlation 3', 4: 'correlation 4'}
    correlation_paths = []
    for _filename in all_countrates_final.keys():
        if _filename in file_to_path:
            correlation_paths.append(file_to_path[_filename])
        else:
            print(f'[!]NO matching file found for: {_filename}')
    all_correlations, n_failed_corr = extract_dict(correlation_paths, extract_correlation, correlation_column_names)
    print(f'Correlation extraction complete: {len(all_correlations)} successful' + (f', {n_failed_corr} failed' if n_failed_corr > 0 else ''))
    return (all_correlations,)


@app.cell
def _(mo):
    #optionally plots the full correlations for visual quality inspection.
    ui_plot_correlation_graphs = mo.ui.checkbox(value=False, label="Inspect full correlation curves")
    ui_plot_correlation_graphs
    return (ui_plot_correlation_graphs,)


@app.cell
def _(all_correlations, plot_correlations, ui_plot_correlation_graphs):
    plot_correlation_graphs = ui_plot_correlation_graphs.value

    if plot_correlation_graphs:
        plot_correlations(all_correlations, show_indices=True)
    else:
        print("correlation plotting skipped.")
    return


@app.cell
def _(mo):
    #exclude correlation datasets by the index shown in the plot titles above
    ui_exclude_correlation = mo.ui.text(
        value="", label="Correlation dataset indices to EXCLUDE (comma-separated, blank = keep all)")
    ui_exclude_correlation
    return (ui_exclude_correlation,)


@app.cell
def _(all_correlations, exclude_by_index, ui_exclude_correlation):
    all_correlations_final = exclude_by_index(all_correlations, ui_exclude_correlation.value)
    print(f"{len(all_correlations_final)} of {len(all_correlations)} correlation dataset(s) in use.")
    return (all_correlations_final,)


@app.cell
def _(
    all_correlations_final,
    df_basedata,
    process_correlation_data,
    remove_from_data,
):
    #locks in the final working datasets after all filtering steps.
    #df_basedata_mod is re-synced here to exactly match the files that survived both countrate and correlation filtering
    #processed_correlations_1 contains the cross-correlation mean [time in seconds, mean of channels 1 and 2] ready for fitting.


    #lock in final correlation dataset
    all_correlations_mod = all_correlations_final.copy()

    #re-sync df_basedata to surviving files
    files_to_exclude = [f for f in df_basedata['filename']
                        if f not in all_correlations_mod.keys()]

    if files_to_exclude:
        df_basedata_mod = remove_from_data(df_basedata, files_to_exclude)
        df_basedata_mod = df_basedata_mod.reset_index(drop=True)
        df_basedata_mod.index = df_basedata_mod.index + 1
        print(f"df_basedata_mod updated: {len(files_to_exclude)} file(s) removed "
              f"to match filtered correlations.")
    else:
        df_basedata_mod = df_basedata.copy()

    print(f"final dataset: {len(all_correlations_mod)} correlation(s), "
          f"{len(df_basedata_mod)} basedata entries.")

    #process into cross-correlation mean for fitting
    #drops raw channel columns and converts time to seconds
    columns_to_drop = ['time [ms]', 'correlation 1', 'correlation 2', 'correlation 3', 'correlation 4']
    processed_correlations_1 = process_correlation_data(all_correlations_mod, columns_to_drop)
    return (
        all_correlations_mod,
        df_basedata_mod,
        files_to_exclude,
        processed_correlations_1,
    )


@app.cell
def _(mo):
    #OPTIONAL NOISE CORRECTION ON CORRELATION DATA
    ui_perform_noise_correction = mo.ui.checkbox(value=False, label="Apply noise correction")
    #baseline_correction: subtracts the mean of the last X% of points from the entire curve (removes residual DC offset)
    ui_baseline_correction = mo.ui.checkbox(value=True, label="Baseline correction")
    ui_baseline_pct = mo.ui.slider(1, 30, value=10, label="Baseline: tail % used for estimate")
    #intercept_correction: replaces the first X% of points with their flat mean
    #(stabilises the plateau/beta used as the fit amplitude in all downstream methods)
    ui_intercept_correction = mo.ui.checkbox(value=True, label="Intercept correction")
    ui_intercept_pct = mo.ui.slider(1, 30, value=5, label="Intercept: head % used for estimate")
    ui_plot_correction = mo.ui.checkbox(value=True, label="Show before/after plot")

    mo.accordion({"Noise correction settings": mo.vstack([
        ui_perform_noise_correction,
        mo.hstack([ui_baseline_correction, ui_baseline_pct]),
        mo.hstack([ui_intercept_correction, ui_intercept_pct]),
        ui_plot_correction,
    ])})
    return (
        ui_baseline_correction,
        ui_baseline_pct,
        ui_intercept_correction,
        ui_intercept_pct,
        ui_perform_noise_correction,
        ui_plot_correction,
    )


@app.cell
def _(
    apply_noise_corrections,
    df_basedata_mod,
    experiment_name,
    plot_correction_sample,
    processed_correlations_1,
    ui_baseline_correction,
    ui_baseline_pct,
    ui_intercept_correction,
    ui_intercept_pct,
    ui_perform_noise_correction,
    ui_plot_correction,
):
    perform_noise_correction = ui_perform_noise_correction.value
    baseline_correction = ui_baseline_correction.value
    baseline_pct = ui_baseline_pct.value
    intercept_correction = ui_intercept_correction.value
    intercept_pct = ui_intercept_pct.value
    plot_correction = ui_plot_correction.value

    #column names in processed_correlations_1
    noise_col      = 'g(2)-1'
    noise_time_col = 't [s]'

    if perform_noise_correction:
        processed_correlations_2 = apply_noise_corrections(
            processed_correlations_1, noise_col,
            baseline_correction, baseline_pct,
            intercept_correction, intercept_pct
        )
        print(f"Noise correction applied to {len(processed_correlations_2)} files.")
        print(f"  Baseline:  mean of last {baseline_pct}% subtracted"   if baseline_correction  else "  Baseline:  skipped")
        print(f"  Intercept: first {intercept_pct}% flattened to mean"  if intercept_correction else "  Intercept: skipped")

        if plot_correction:
            plot_correction_sample(
                processed_correlations_1, processed_correlations_2,
                df_basedata=df_basedata_mod,
                col=noise_col, time_col=noise_time_col,
                title=experiment_name
            )
    else:
        processed_correlations_2 = processed_correlations_1
        print("Noise correction skipped. processed_correlations_2 = processed_correlations_1.")
    return (processed_correlations_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### II. CUMULANT METHODS
    """)
    return


@app.cell
def _(mo):
    #toggle which methods to run
    # ============================================================
    ui_perform_A = mo.ui.checkbox(value=True, label="A — ALV cumulant fit")
    ui_perform_B = mo.ui.checkbox(value=False, label="B — linear ln√(g2-1) fit")
    ui_perform_C = mo.ui.checkbox(value=False, label="C — iterative nonlinear fit")
    ui_perform_D = mo.ui.checkbox(value=True, label="D — multimodal (dirac-delta) fit")
    mo.vstack([mo.md("**Which cumulant methods to run:**"),
               mo.hstack([ui_perform_A, ui_perform_B, ui_perform_C, ui_perform_D])])
    return ui_perform_A, ui_perform_B, ui_perform_C, ui_perform_D


@app.cell
def _(ui_perform_A, ui_perform_B, ui_perform_C, ui_perform_D):
    perform_cumulant_A = ui_perform_A.value
    perform_cumulant_B = ui_perform_B.value
    perform_cumulant_C = ui_perform_C.value
    perform_cumulant_D = ui_perform_D.value
    return (
        perform_cumulant_A,
        perform_cumulant_B,
        perform_cumulant_C,
        perform_cumulant_D,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##### CUMULANT-METHOD A
    {using cumulant-fit data from ALV-Software}
    """)
    return


@app.cell
def _(mo):
    #extract cumulant data only for files in the filtered correlations
    ui_show_raw_data_A = mo.ui.checkbox(value=False, label="Print extracted cumulant table")
    ui_export_data_A = mo.ui.checkbox(value=True, label="Export to .txt")
    mo.hstack([ui_show_raw_data_A, ui_export_data_A])
    return ui_export_data_A, ui_show_raw_data_A


@app.cell
def _(
    all_correlations_mod,
    df_basedata_mod,
    experiment_name,
    extract_cumulants,
    extract_dataframe,
    file_to_path,
    files_to_exclude,
    pd,
    perform_cumulant_A,
    remove_from_data,
    ui_export_data_A,
    ui_show_raw_data_A,
):
    show_raw_data_A = ui_show_raw_data_A.value
    export_data_A = ui_export_data_A.value

    if perform_cumulant_A:

        print("Extracting data from files...")

        #extract cumulants for all files present in the filtered correlations
        cumA_paths = [file_to_path[fn] for fn in all_correlations_mod.keys() if fn in file_to_path]
        df_extracted_cumulants, _n_failed_cumA = extract_dataframe(cumA_paths, extract_cumulants)

        #apply exclusions if already defined upstream
        df_extracted_cumulants_mod = remove_from_data(df_extracted_cumulants, files_to_exclude) if files_to_exclude else df_extracted_cumulants
        df_extracted_cumulants_mod = df_extracted_cumulants_mod.reset_index(drop=True)
        df_extracted_cumulants_mod.index = df_extracted_cumulants_mod.index + 1

        #merge with basedata
        cumulant_method_A_data = pd.merge(df_basedata_mod, df_extracted_cumulants_mod, on='filename', how='outer')
        cumulant_method_A_data = cumulant_method_A_data.reset_index(drop=True)
        cumulant_method_A_data.index = cumulant_method_A_data.index + 1

        print(f"Cumulant Method A: extracted data for {len(df_extracted_cumulants_mod)} files.")

        if show_raw_data_A:
            print("\nExtracted cumulant data:")
            print(cumulant_method_A_data.to_string())

        if export_data_A:
            fname_A = f'Cumulant_Method_A_data_{experiment_name}.txt'
            cumulant_method_A_data.to_csv(fname_A, sep='\t', index=False)
            print(f"Data exported to {fname_A}")
    else:
        print("Cumulant Method A skipped.")
    return (cumulant_method_A_data,)


@app.cell
def _(make_fit_range_ui, mo):
    #linear regression to determine D
    ui_restrict_A, ui_range_A = make_fit_range_ui("Method A", (0, 0.001), restrict_default=True)
    ui_through_origin_A = mo.ui.checkbox(value=False, label="Also fit forced through origin")
    mo.accordion({"Method A: fit-range settings": mo.vstack([mo.hstack([ui_restrict_A, ui_through_origin_A]), ui_range_A])})
    return ui_range_A, ui_restrict_A, ui_through_origin_A


@app.cell
def _(
    analyze_diffusion_coefficient,
    cumulant_method_A_data,
    perform_cumulant_A,
    ui_range_A,
    ui_restrict_A,
    ui_through_origin_A,
):
    fit_x_range_A = tuple(ui_range_A.value) if ui_restrict_A.value else None
    fit_through_origin_A = ui_through_origin_A.value

    if perform_cumulant_A:

        cumulant_method_A_diff = analyze_diffusion_coefficient(
            data_df      = cumulant_method_A_data,
            q_squared_col= 'q^2',
            gamma_cols   = [
                '1st order frequency [1/ms]',
                '2nd order frequency [1/ms]',
                '3rd order frequency [1/ms]'
            ],
            gamma_unit   = '1/ms',   # ALV data: 1/ms
            x_range            = fit_x_range_A,
            fit_through_origin = fit_through_origin_A
        )
    else:
        print("Cumulant Method A skipped — regression not executed.")
    return (cumulant_method_A_diff,)


@app.cell
def _(
    c,
    cumulant_method_A_data,
    cumulant_method_A_diff,
    delta_c,
    np,
    pd,
    perform_cumulant_A,
    rh_from_slope,
):
    #calculate results (D, PDI, Skewness and Rh)

    def _results_A():
        cumulant_method_A_data['PDI_2nd'] = (
            cumulant_method_A_data['2nd order frequency exp param [ms^2]'] /
            cumulant_method_A_data['2nd order frequency [1/ms]']**2)
        cumulant_method_A_data['PDI_3rd'] = (
            cumulant_method_A_data['3rd order frequency exp param [ms^2]'] /
            cumulant_method_A_data['3rd order frequency [1/ms]']**2)
        polydispersity_A_2 = cumulant_method_A_data['PDI_2nd'].mean()
        polydispersity_A_3 = cumulant_method_A_data['PDI_3rd'].mean()

        #skewness from 3rd order cumulants
        cumulant_method_A_data['Skewness_3rd'] = (
            cumulant_method_A_data['3rd order frequency exp param [ms^2]'] /
            cumulant_method_A_data['2nd order frequency exp param [ms^2]']**(3/2))
        skewness_A_3 = cumulant_method_A_data['Skewness_3rd'].mean()

        fit_labels = [
            'Rh from 1st order cumulant fit',
            'Rh from 2nd order cumulant fit',
            'Rh from 3rd order cumulant fit']
        rows = []
        for i in range(3):
            D, D_err, Rh, Rh_err = rh_from_slope(
                cumulant_method_A_diff['q^2_coef'].iloc[i],
                cumulant_method_A_diff['q^2_se'].iloc[i],
                c, delta_c, unit_factor=1e-15)
            rows.append({
                'Fit'           : fit_labels[i],
                'D [m²/s]'      : D,
                'D error [m²/s]': D_err,
                'Rh [nm]'       : Rh,
                'Rh error [nm]' : Rh_err,
                'R_squared'     : cumulant_method_A_diff['R_squared'].iloc[i],
                'intercept'     : cumulant_method_A_diff['intercept'].iloc[i],
                'Residuals'     : cumulant_method_A_diff['Normality'].iloc[i],
                'PDI'           : polydispersity_A_2 if i == 1 else polydispersity_A_3 if i == 2 else np.nan,
                'Skewness'      : skewness_A_3 if i == 2 else np.nan,
                'Kurtosis'      : np.nan,
            })
        return pd.DataFrame(rows)

    if perform_cumulant_A:
        method_A_cumulant_result = _results_A()
    else:
        method_A_cumulant_result = pd.DataFrame({
            'Fit'           : ['Rh from 1st order cumulant fit',
                               'Rh from 2nd order cumulant fit',
                               'Rh from 3rd order cumulant fit'],
            'D [m²/s]'      : [0, 0, 0],
            'D error [m²/s]': [0, 0, 0],
            'Rh [nm]'       : [0, 0, 0],
            'Rh error [nm]' : [0, 0, 0],
            'R_squared'     : [0, 0, 0],
            'intercept'     : [np.nan, np.nan, np.nan],
            'Residuals'     : [0, 0, 0],
            'PDI'           : [np.nan, 0, 0],
            'Skewness'      : [np.nan, np.nan, 0],
            'Kurtosis'      : [np.nan, np.nan, np.nan],
        })
        print("Cumulant Method A skipped — zero result placeholder created.")

    method_A_cumulant_result
    return (method_A_cumulant_result,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##### CUMULANT-METHOD B
    {linear regression of ln sqrt(g(2)-1)}
    """)
    return


@app.cell
def _(
    calculate_g2_B,
    df_basedata_mod,
    pd,
    perform_cumulant_B,
    plot_processed_correlations,
    processed_correlations_2,
):
    #fitting
    fit_limits_B = (0, 0.0002)  #time window [s] for individual curve fitting — keep narrow (only valid for small lag times)
    xlim_B = (0, 0.0002) #plotting limits x-axis or None for auto
    ylim_B = (-0.15, -0.025) #plotting limits y-axis or None for auto
    #fit function: ln(sqrt(g(2)-1)) = 0.5*ln(a) - b*t, wheras b = decay rate Gamma [1/s]

    if perform_cumulant_B:

        #compute sqrt(g(2)-1), drop non-positive values
        processed_correlations_B = calculate_g2_B(processed_correlations_2)

        #fit each correlation individually and plot
        cumulant_method_B_fit = plot_processed_correlations(
            processed_correlations_B, fit_limits_B, xlim_B, ylim_B)

        #merge fit results with basedata
        cumulant_method_B_data = pd.merge(
            df_basedata_mod, cumulant_method_B_fit, on='filename', how='outer')
        cumulant_method_B_data = cumulant_method_B_data.reset_index(drop=True)
        cumulant_method_B_data.index = cumulant_method_B_data.index + 1

        print(f"Cumulant Method B: fitted {len(cumulant_method_B_data)} files.")
    else:
        print("Cumulant Method B skipped.")
    return (cumulant_method_B_data,)


@app.cell
def _(mo):
    ui_export_data_B = mo.ui.checkbox(value=True, label="Export to .txt")
    ui_export_data_B
    return (ui_export_data_B,)


@app.cell
def _(
    cumulant_method_B_data,
    perform_cumulant_B,
    select_fits_table,
    ui_export_data_B,
):
    export_data_B = ui_export_data_B.value

    if perform_cumulant_B:
        ui_table_B = select_fits_table(cumulant_method_B_data, ['filename', 'Gamma', 'R_squared'],
                                        label="Method B — every file fit; deselect any bad ones")
    else:
        ui_table_B = None
    ui_table_B
    return export_data_B, ui_table_B


@app.cell
def _(
    cumulant_method_B_data,
    experiment_name,
    export_data_B,
    maybe_export,
    perform_cumulant_B,
    resolve_selection,
    ui_table_B,
):
    if perform_cumulant_B:
        cumulant_method_B_data_mod = resolve_selection(ui_table_B, cumulant_method_B_data, "Method B")
        fname_B = f'Cumulant_Method_B_data_{experiment_name}.txt'
        maybe_export(cumulant_method_B_data_mod, export_data_B, fname_B)
    else:
        print("Cumulant Method B skipped.")
    return (cumulant_method_B_data_mod,)


@app.cell
def _(make_fit_range_ui, mo):
    #linear regression to determine D
    ui_restrict_B, ui_range_B = make_fit_range_ui("Method B", (0, 0.001), restrict_default=True)
    ui_through_origin_B = mo.ui.checkbox(value=True, label="Also fit forced through origin")
    mo.accordion({"Method B: fit-range settings": mo.vstack([mo.hstack([ui_restrict_B, ui_through_origin_B]), ui_range_B])})
    return ui_range_B, ui_restrict_B, ui_through_origin_B


@app.cell
def _(
    analyze_diffusion_coefficient,
    cumulant_method_B_data_mod,
    perform_cumulant_B,
    ui_range_B,
    ui_restrict_B,
    ui_through_origin_B,
):
    fit_x_range_B = tuple(ui_range_B.value) if ui_restrict_B.value else None
    fit_through_origin_B = ui_through_origin_B.value

    if perform_cumulant_B:
        cumulant_method_B_diff = analyze_diffusion_coefficient(
            data_df       = cumulant_method_B_data_mod,
            q_squared_col = 'q^2',
            gamma_cols    = ['Gamma'],
            method_names  = ['Method B'],
            x_range            = fit_x_range_B,
            fit_through_origin = fit_through_origin_B
        )
    else:
        print("Cumulant Method B skipped — regression not executed.")
    return (cumulant_method_B_diff,)


@app.cell
def _(
    c,
    cumulant_method_B_diff,
    delta_c,
    np,
    pd,
    perform_cumulant_B,
    rh_from_slope,
):
    #calculate results (D, PDI and Rh)

    if perform_cumulant_B:
        D, D_err, Rh, Rh_err = rh_from_slope(
            cumulant_method_B_diff['q^2_coef'].iloc[0],
            cumulant_method_B_diff['q^2_se'].iloc[0],
            c, delta_c, unit_factor=1e-18)

        method_B_cumulant_result = pd.DataFrame([{
            'Fit'           : 'Rh from linear cumulant fit',
            'D [m²/s]'      : D,
            'D error [m²/s]': D_err,
            'Rh [nm]'       : Rh,
            'Rh error [nm]' : Rh_err,
            'R_squared'     : cumulant_method_B_diff['R_squared'].iloc[0],
            'intercept'     : cumulant_method_B_diff['intercept'].iloc[0],
            'Residuals'     : cumulant_method_B_diff['Normality'].iloc[0],
            'PDI'           : np.nan,
            'Skewness'      : np.nan,
            'Kurtosis'      : np.nan,
        }])
    else:
        method_B_cumulant_result = pd.DataFrame([{
            'Fit'           : 'Rh from linear cumulant fit',
            'D [m²/s]'      : 0, 'D error [m²/s]': 0,
            'Rh [nm]'       : 0, 'Rh error [nm]' : 0,
            'R_squared'     : 0, 'intercept'      : np.nan, 'Residuals'      : 0,
            'PDI'           : np.nan, 'Skewness'  : 0,
            'Kurtosis'      : np.nan,
        }])
        print("Cumulant Method B skipped — zero result placeholder created.")

    method_B_cumulant_result
    return (method_B_cumulant_result,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##### CUMULANT-METHOD C
    {iterative nonlinear fit}
    """)
    return


@app.cell
def _(mo, np):
    #fit functions
    def fit_function1(x, a, b, f): #1st order cumulant
        return f + a * np.exp(-2 * b * x)

    def fit_function2(x, a, b, c, f): #2nd order cumulant
        inner_term = 1 + 0.5 * c * x**2
        return f + a * (np.exp(-b * x) * inner_term)**2

    def fit_function3(x, a, b, c, d, f): #3rd order cumulant
        inner_term = 1 + 0.5 * c * x**2 - (d * x**3) / 6
        return f + a * (np.exp(-b * x) * inner_term)**2

    def fit_function4(x, a, b, c, d, e, f): #4th order cumulant
        inner_term = 1 + 0.5 * c * x**2 - (d * x**3) / 6 + (e * x**4) / 24
        return f + a * (np.exp(-b * x) * inner_term)**2

    #initial parameters [a, b, c, d, e, f]
    # a: amplitude ≈ beta (typically 0.8-0.9)
    # b: mean decay rate Gamma [1/s] — adjust to your expected size range
    # c: 2nd cumulant (≈ 0 monodisperse, > 0 polydisperse)
    # d: 3rd cumulant — asymmetry (start at 0)
    # e: 4th cumulant (start at 0)
    # f: baseline offset (≈ 0 after noise correction)

    fit_function_options = {'1st order': fit_function1, '2nd order': fit_function2,
                             '3rd order': fit_function3, '4th order': fit_function4}
    ui_chosen_orders = mo.ui.multiselect(options=list(fit_function_options.keys()), value=['4th order'],
                                          label="Cumulant order(s) to fit (select several to compare)")
    ui_adaptive_guesses = mo.ui.checkbox(value=True, label="Adaptive initial-parameter guesses")
    ui_fit_method = mo.ui.dropdown(options=['lm', 'trf'], value='lm', label="Fit method")
    mo.accordion({"Method C: fit-function settings": mo.vstack([ui_chosen_orders, mo.hstack([ui_adaptive_guesses, ui_fit_method])])})
    return (
        fit_function4,
        fit_function_options,
        ui_adaptive_guesses,
        ui_chosen_orders,
        ui_fit_method,
    )


@app.cell
def _(
    df_basedata_mod,
    fit_function4,
    fit_function_options,
    get_adaptive_initial_parameters,
    get_meaningful_parameters,
    pd,
    perform_cumulant_C,
    plot_processed_correlations_iterative,
    processed_correlations_2,
    ui_adaptive_guesses,
    ui_chosen_orders,
    ui_fit_method,
):
    chosen_fit_functions = [fit_function_options[label] for label in ui_chosen_orders.value] or [fit_function4]
    base_initial_parameters = [0.8, 10000, 0, 0, 0, 0]
    adaptive_initial_guesses = ui_adaptive_guesses.value
    fit_method = ui_fit_method.value
    if perform_cumulant_C:
        all_fits = []
        plot_offset = 1
        for _fit_func in chosen_fit_functions:
            print(f'Fitting with {_fit_func.__name__}...')
            if adaptive_initial_guesses:
                initial_parameters = get_adaptive_initial_parameters(processed_correlations_2, _fit_func, base_initial_parameters, verbose=False)
            else:
                initial_parameters = get_meaningful_parameters(_fit_func, base_initial_parameters)
            fit_result = plot_processed_correlations_iterative(processed_correlations_2, _fit_func, (1e-09, 10), initial_parameters, method=fit_method, plot_number_start=plot_offset)
            fit_result['fit_function'] = _fit_func.__name__
            plot_offset += len(fit_result)
            all_fits.append(fit_result)
        cumulant_method_C_fit = pd.concat(all_fits, ignore_index=True)
        cumulant_method_C_data = pd.merge(df_basedata_mod, cumulant_method_C_fit, on='filename', how='outer')
        cumulant_method_C_data = cumulant_method_C_data.reset_index(drop=True)
        cumulant_method_C_data.index = cumulant_method_C_data.index + 1
        print(f'\nCumulant Method C: {len(chosen_fit_functions)} function(s), {len(cumulant_method_C_data)} total rows.')
    else:
        print('Cumulant Method C skipped.')  #concatenate results from all fit functions — continuous index across all  #merge with basedata
    return chosen_fit_functions, cumulant_method_C_data


@app.cell
def _(mo):
    ui_export_data_C = mo.ui.checkbox(value=True, label="Export to .txt")
    ui_export_data_C
    return (ui_export_data_C,)


@app.cell
def _(
    cumulant_method_C_data,
    perform_cumulant_C,
    select_fits_table,
    ui_export_data_C,
):
    export_data_C = ui_export_data_C.value

    if perform_cumulant_C:
        ui_table_C = select_fits_table(cumulant_method_C_data, ['filename', 'fit_function', 'best_R-squared'],
                                        label="Method C — every file fit; deselect any bad ones")
    else:
        ui_table_C = None
    ui_table_C
    return export_data_C, ui_table_C


@app.cell
def _(
    cumulant_method_C_data,
    experiment_name,
    export_data_C,
    maybe_export,
    perform_cumulant_C,
    resolve_selection,
    ui_table_C,
):
    if perform_cumulant_C:
        cumulant_method_C_data_mod = resolve_selection(ui_table_C, cumulant_method_C_data, "Method C")
        fname_C = f'Cumulant_Method_C_data_{experiment_name}.txt'
        maybe_export(cumulant_method_C_data_mod, export_data_C, fname_C)
    else:
        print("Cumulant Method C skipped.")
    return (cumulant_method_C_data_mod,)


@app.cell
def _(make_fit_range_ui, mo):
    #linear regression to determine D
    ui_restrict_C, ui_range_C = make_fit_range_ui("Method C", (0, 0.001), restrict_default=False)
    ui_through_origin_C = mo.ui.checkbox(value=False, label="Also fit forced through origin")
    mo.accordion({"Method C: fit-range settings": mo.vstack([mo.hstack([ui_restrict_C, ui_through_origin_C]), ui_range_C])})
    return ui_range_C, ui_restrict_C, ui_through_origin_C


@app.cell
def _(
    analyze_diffusion_coefficient,
    chosen_fit_functions,
    cumulant_method_C_data_mod,
    pd,
    perform_cumulant_C,
    ui_range_C,
    ui_restrict_C,
    ui_through_origin_C,
):
    fit_x_range_C = tuple(ui_range_C.value) if ui_restrict_C.value else None
    fit_through_origin_C = ui_through_origin_C.value
    if perform_cumulant_C:
        diff_results_C = []
        for _fit_func in chosen_fit_functions:
            subset_C = cumulant_method_C_data_mod[cumulant_method_C_data_mod['fit_function'] == _fit_func.__name__].copy()
            result_C = analyze_diffusion_coefficient(data_df=subset_C, q_squared_col='q^2', gamma_cols=['best_b'], method_names=[_fit_func.__name__], x_range=fit_x_range_C, fit_through_origin=fit_through_origin_C)
            diff_results_C.append(result_C)
        cumulant_method_C_diff = pd.concat(diff_results_C, ignore_index=True)
    else:
        print('Cumulant Method C skipped — regression not executed.')
    return (cumulant_method_C_diff,)


@app.cell
def _(
    c,
    chosen_fit_functions,
    cumulant_method_C_data_mod,
    cumulant_method_C_diff,
    delta_c,
    np,
    pd,
    perform_cumulant_C,
    rh_from_slope,
):
    #calculate results (D, PDI and Rh)
    def _results_C():
        fit_labels_C = {'fit_function1': '1st order cumulant fit', 'fit_function2': '2nd order cumulant fit', 'fit_function3': '3rd order cumulant fit', 'fit_function4': '4th order cumulant fit'}
        rows = []
        for i, _fit_func in enumerate(chosen_fit_functions):
            subset = cumulant_method_C_data_mod[cumulant_method_C_data_mod['fit_function'] == _fit_func.__name__].copy()
            if _fit_func.__name__ == 'fit_function1':
                pdi = np.nan
            else:
                pdi = (subset['best_c'] / subset['best_b'] ** 2).mean()
            if _fit_func.__name__ in ('fit_function3', 'fit_function4'):
                skewness = (subset['best_d'] / subset['best_c'] ** (3 / 2)).mean()
            else:
                skewness = np.nan
            if _fit_func.__name__ == 'fit_function4':  # PDI (not defined for 1st order)
                kurtosis = (subset['best_e'] / subset['best_c'] ** 2).mean()
            else:
                kurtosis = np.nan
            D, D_err, Rh, Rh_err = rh_from_slope(cumulant_method_C_diff['q^2_coef'].iloc[i], cumulant_method_C_diff['q^2_se'].iloc[i], c, delta_c, unit_factor=1e-18)
            rows.append({'Fit': f'Rh from {fit_labels_C.get(_fit_func.__name__, _fit_func.__name__)}', 'D [m²/s]': D, 'D error [m²/s]': D_err, 'Rh [nm]': Rh, 'Rh error [nm]': Rh_err, 'R_squared': cumulant_method_C_diff['R_squared'].iloc[i], 'intercept': cumulant_method_C_diff['intercept'].iloc[i], 'Residuals': cumulant_method_C_diff['Normality'].iloc[i], 'PDI': pdi, 'Skewness': skewness, 'Kurtosis': kurtosis})  # Skewness (only for fit_function3 and fit_function4)
        return pd.DataFrame(rows)
    if perform_cumulant_C:
        method_C_cumulant_result = _results_C()
    else:
        all_fit_labels_C = {'fit_function1': 'Rh from 1st order cumulant fit', 'fit_function2': 'Rh from 2nd order cumulant fit', 'fit_function3': 'Rh from 3rd order cumulant fit', 'fit_function4': 'Rh from 4th order cumulant fit'}  # Kurtosis (only for fit_function4)
        method_C_cumulant_result = pd.DataFrame([{'Fit': label, 'D [m²/s]': 0, 'D error [m²/s]': 0, 'Rh [nm]': 0, 'Rh error [nm]': 0, 'R_squared': 0, 'intercept': np.nan, 'Residuals': 0, 'PDI': np.nan if name == 'fit_function1' else 0, 'Skewness': np.nan if name in ('fit_function1', 'fit_function2') else 0, 'Kurtosis': np.nan if name in ('fit_function1', 'fit_function2', 'fit_function3') else 0} for name, label in all_fit_labels_C.items()])
        print('Cumulant Method C skipped — zero result placeholder created.')
    method_C_cumulant_result
    return (method_C_cumulant_result,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##### CUMULANT-METHOD D
    {multimodal cumulant analysis via dirac delta modes}
    """)
    return


@app.cell
def _(mo):
    #fitting
    ui_n_max = mo.ui.number(2, 50, value=25, label="Max number of Dirac-delta modes to try")
    ui_n_start = mo.ui.number(1, 10, value=1, label="Starting number of modes (1=monomodal, 3-5=known polydisperse)")
    ui_gap_threshold = mo.ui.slider(0.5, 5.0, step=0.1, value=2.0,
                                     label="Min log-ratio between adjacent gammas to count as separate population",
                                     show_value=True)
    mo.accordion({"Method D: fitting settings": mo.vstack([mo.hstack([ui_n_max, ui_n_start]), ui_gap_threshold])})
    return ui_gap_threshold, ui_n_max, ui_n_start


@app.cell
def _(
    df_basedata_mod,
    fit_correlations_method_D,
    pd,
    perform_cumulant_D,
    processed_correlations_2,
    ui_gap_threshold,
    ui_n_max,
    ui_n_start,
):
    n_max = ui_n_max.value
    n_start = ui_n_start.value
    gap_threshold = ui_gap_threshold.value

    if perform_cumulant_D:

        cumulant_method_D_fit = fit_correlations_method_D(
            processed_correlations_2,
            n_max=n_max, n_start=n_start,
            gap_threshold=gap_threshold,
            plot=True
        )

        #merge with basedata
        cumulant_method_D_data = pd.merge(
            df_basedata_mod, cumulant_method_D_fit, on='filename', how='outer'
        )
        cumulant_method_D_data = cumulant_method_D_data.reset_index(drop=True)
        cumulant_method_D_data.index = cumulant_method_D_data.index + 1

        print(f"\nCumulant Method D: fitted {len(cumulant_method_D_data)} files.")
        print(f"Populations found per file: {cumulant_method_D_data['n_populations'].value_counts().sort_index().to_dict()}")
    else:
        print("Cumulant Method D skipped.")
    return (cumulant_method_D_data,)


@app.cell
def _(mo):
    ui_export_data_D = mo.ui.checkbox(value=False, label="Export to .txt")
    ui_export_data_D
    return (ui_export_data_D,)


@app.cell
def _(
    cumulant_method_D_data,
    perform_cumulant_D,
    select_fits_table,
    ui_export_data_D,
):
    export_data_D = ui_export_data_D.value

    if perform_cumulant_D:
        ui_table_D = select_fits_table(cumulant_method_D_data, ['filename', 'n_populations', 'pdi', 'R-squared'],
                                        label="Method D — every file fit; deselect any bad ones")
    else:
        ui_table_D = None
    ui_table_D
    return export_data_D, ui_table_D


@app.cell
def _(
    cumulant_method_D_data,
    experiment_name,
    export_data_D,
    maybe_export,
    perform_cumulant_D,
    resolve_selection,
    ui_table_D,
):
    if perform_cumulant_D:
        cumulant_method_D_data_mod = resolve_selection(ui_table_D, cumulant_method_D_data, "Method D")
        fname_D = f'Cumulant_Method_D_data_{experiment_name}.txt'
        maybe_export(cumulant_method_D_data_mod, export_data_D, fname_D)
    else:
        print("Cumulant Method D skipped.")
    return (cumulant_method_D_data_mod,)


@app.cell
def _(make_clustering_ui, mo):
    #clustering
    layout_clust_D, ui_enable_clust_D, ui_normalize_clust_D, ui_uncertainty_clust_D, \
        ui_distance_clust_D, ui_abundance_clust_D, ui_strategy_clust_D = \
        make_clustering_ui("Method D", distance_threshold_default=3.0, min_abundance_default=0.3,
                            clustering_strategy_default='silhouette_refined')
    mo.accordion({"Method D: clustering settings": layout_clust_D})
    return (
        ui_abundance_clust_D,
        ui_distance_clust_D,
        ui_enable_clust_D,
        ui_normalize_clust_D,
        ui_strategy_clust_D,
        ui_uncertainty_clust_D,
    )


@app.cell
def _(
    cluster_all_gammas,
    cumulant_method_D_data_mod,
    get_reliable_gamma_cols,
    perform_cumulant_D,
    ui_abundance_clust_D,
    ui_distance_clust_D,
    ui_enable_clust_D,
    ui_normalize_clust_D,
    ui_strategy_clust_D,
    ui_uncertainty_clust_D,
):
    enable_clustering_D = ui_enable_clust_D.value
    normalize_by_q2_D = ui_normalize_clust_D.value
    n_clusters_D = 'auto'
    distance_threshold_D = ui_distance_clust_D.value
    min_abundance_D = ui_abundance_clust_D.value
    clustering_strategy_D = ui_strategy_clust_D.value
    uncertainty_flags_D = ui_uncertainty_clust_D.value
    uncertainty_threshold_D = 0.5

    if perform_cumulant_D:

        gamma_cols_D = [col for col in cumulant_method_D_data_mod.columns
                      if col.startswith('gamma_') and col != 'gamma_mean']

        cumulant_method_D_clustered, cluster_info_D = cluster_all_gammas(
            cumulant_method_D_data_mod,
            gamma_cols            = gamma_cols_D,
            q_squared_col         = 'q^2',
            enable_clustering     = enable_clustering_D,
            normalize_by_q2       = normalize_by_q2_D,
            n_clusters            = n_clusters_D,
            distance_threshold    = distance_threshold_D,
            min_abundance         = min_abundance_D,
            clustering_strategy   = clustering_strategy_D,
            uncertainty_flags     = uncertainty_flags_D,
            uncertainty_threshold = uncertainty_threshold_D,
            plot=True
        )

        reliable_gamma_cols_D = get_reliable_gamma_cols(cluster_info_D)
        print(f"\n{cluster_info_D['n_populations']} reliable population(s) found.")
    else:
        print("Cumulant Method D skipped — clustering not executed.")
    return (
        cluster_info_D,
        clustering_strategy_D,
        cumulant_method_D_clustered,
        gamma_cols_D,
        reliable_gamma_cols_D,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Clustering sensitivity sweep for Method D — sweeps `distance_threshold` x
    `min_abundance` and reports populations found + silhouette score at each
    setting. Off by default (25 clustering runs); tick to run one.
    """)
    return


@app.cell
def _(mo):
    ui_run_sweep_D = mo.ui.checkbox(label="Run clustering sensitivity sweep for Method D")
    ui_run_sweep_D
    return (ui_run_sweep_D,)


@app.cell
def _(
    clustering_sensitivity_sweep,
    clustering_strategy_D,
    cumulant_method_D_data_mod,
    distance_thresholds,
    gamma_cols_D,
    min_abundances,
    perform_cumulant_D,
    plot_clustering_heatmaps,
    ui_run_sweep_D,
):
    if perform_cumulant_D and ui_run_sweep_D.value:
        df_sweep_D = clustering_sensitivity_sweep(
            cumulant_method_D_data_mod, gamma_cols_D, 'q^2', clustering_strategy_D,
            distance_thresholds, min_abundances)
        fig_sweep_D = plot_clustering_heatmaps(df_sweep_D, 'Method D')
        fig_sweep_D
    else:
        print("Method D clustering sweep skipped.")
    return


@app.cell
def _(make_fit_range_ui, mo):
    #linear regression
    ui_restrict_D, ui_range_D = make_fit_range_ui("Method D", (0, 0.001), restrict_default=False)
    ui_through_origin_D = mo.ui.checkbox(value=False, label="Also fit forced through origin")
    mo.accordion({"Method D: fit-range settings": mo.vstack([mo.hstack([ui_restrict_D, ui_through_origin_D]), ui_range_D])})
    return ui_range_D, ui_restrict_D, ui_through_origin_D


@app.cell
def _(
    analyze_diffusion_coefficient,
    cumulant_method_D_clustered,
    perform_cumulant_D,
    reliable_gamma_cols_D,
    ui_range_D,
    ui_restrict_D,
    ui_through_origin_D,
):
    fit_x_range_D = tuple(ui_range_D.value) if ui_restrict_D.value else None
    fit_through_origin_D = ui_through_origin_D.value

    if perform_cumulant_D:

        cumulant_method_D_diff = analyze_diffusion_coefficient(
            data_df       = cumulant_method_D_clustered,
            q_squared_col = 'q^2',
            gamma_cols    = reliable_gamma_cols_D,
            x_range            = fit_x_range_D,
            fit_through_origin = fit_through_origin_D
        )
    else:
        print("Cumulant Method D skipped — regression not executed.")
    return (cumulant_method_D_diff,)


@app.cell
def _(
    c,
    cluster_info_D,
    cumulant_method_D_data_mod,
    cumulant_method_D_diff,
    delta_c,
    np,
    pd,
    perform_cumulant_D,
    rh_from_slope,
):
    #calculate results (D, PDI and Rh)
    def _results_D():
        rows = []
        for idx, _row in cumulant_method_D_diff.iterrows():
            pop_num = idx + 1
            D, D_err, Rh, Rh_err = rh_from_slope(_row['q^2_coef'], _row['q^2_se'], c, delta_c, unit_factor=1e-18)
            result = {'Fit': f'Population {pop_num}', 'D [m²/s]': D, 'D error [m²/s]': D_err, 'Rh [nm]': Rh, 'Rh error [nm]': Rh_err, 'R_squared': _row['R_squared'], 'intercept': _row['intercept'], 'Residuals': _row['Normality'], 'PDI': cumulant_method_D_data_mod['pdi'].mean(), 'Skewness': cumulant_method_D_data_mod['skewness'].mean(), 'Kurtosis': cumulant_method_D_data_mod['kurtosis'].mean()}
            if idx < len(cluster_info_D['population_abundances']):
                result['Abundance'] = f'{cluster_info_D['population_abundances'][idx] * 100:.1f}%'
            if cluster_info_D['silhouette_score'] is not None:
                result['Silhouette'] = cluster_info_D['silhouette_score']
            rows.append(result)
        return pd.DataFrame(rows)
    if perform_cumulant_D:
        method_D_cumulant_result = _results_D()
    else:
        method_D_cumulant_result = pd.DataFrame([{'Fit': 'Population 1', 'D [m²/s]': 0, 'D error [m²/s]': 0, 'Rh [nm]': 0, 'Rh error [nm]': 0, 'R_squared': 0, 'intercept': np.nan, 'Residuals': 0, 'PDI': 0, 'Skewness': 0, 'Kurtosis': 0}])
        print('Cumulant Method D skipped — zero result placeholder created.')
    method_D_cumulant_result
    return (method_D_cumulant_result,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### III. INVERSE-LAPLACIAN METHODS
    """)
    return


@app.cell
def _(mo):
    #toggle which methods to run
    # ============================================================
    ui_perform_nnls = mo.ui.checkbox(value=False, label="NNLS")
    ui_analyze_alpha = mo.ui.checkbox(value=False, label="Alpha analysis (regularization-parameter scan)")
    ui_regularized_fit = mo.ui.checkbox(value=True, label="Regularized fit")
    mo.vstack([mo.md("**Which inverse-Laplacian methods to run:**"),
               mo.hstack([ui_perform_nnls, ui_analyze_alpha, ui_regularized_fit])])
    return ui_analyze_alpha, ui_perform_nnls, ui_regularized_fit


@app.cell
def _(ui_analyze_alpha, ui_perform_nnls, ui_regularized_fit):
    perform_nnls = ui_perform_nnls.value
    analyze_alpha = ui_analyze_alpha.value
    regularized_fit = ui_regularized_fit.value
    return analyze_alpha, perform_nnls, regularized_fit


@app.cell
def _(
    analyze_alpha,
    experiment_name,
    mo,
    n_files,
    perform_cumulant_A,
    perform_cumulant_B,
    perform_cumulant_C,
    perform_cumulant_D,
    perform_nnls,
    regularized_fit,
):
    #at-a-glance summary of what's about to run, before the heavy computation cells below
    _cumulant_on = [name for name, on in
                    [('A', perform_cumulant_A), ('B', perform_cumulant_B),
                     ('C', perform_cumulant_C), ('D', perform_cumulant_D)] if on]
    _multimode_on = [name for name, on in
                     [('NNLS', perform_nnls), ('Regularized fit', regularized_fit)] if on]
    mo.md(f"""
    **Current run:** {n_files} files &nbsp;|&nbsp; experiment `{experiment_name}`
    &nbsp;|&nbsp; cumulant methods: {', '.join(_cumulant_on) or 'none'}
    &nbsp;|&nbsp; multi-mode methods: {', '.join(_multimode_on) or 'none'}
    {'&nbsp;|&nbsp; alpha analysis on' if analyze_alpha else ''}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##### NNLS
    {non-negative least squares fit}
    """)
    return


@app.cell
def _(mo):
    #simple NNLS-Fit without additional constraints
    ui_n_points_nnls = mo.ui.slider(50, 400, step=10, value=100, label="Decay-time grid points", show_value=True)
    ui_prominence_nnls = mo.ui.slider(0.001, 0.1, step=0.001, value=0.01,
                                       label="Peak prominence (lower = more sensitive)", show_value=True)
    ui_distance_nnls = mo.ui.number(1, 10, value=1, label="Min distance between peaks")
    mo.accordion({"NNLS: fitting settings": mo.hstack([ui_n_points_nnls, ui_prominence_nnls, ui_distance_nnls])})
    return ui_distance_nnls, ui_n_points_nnls, ui_prominence_nnls


@app.cell
def _(
    df_basedata_mod,
    nnls_all,
    np,
    pd,
    perform_nnls,
    processed_correlations_2,
    ui_distance_nnls,
    ui_n_points_nnls,
    ui_prominence_nnls,
):
    decay_times_nnls = np.logspace(-8, 1, ui_n_points_nnls.value)  # lag time grid for inverse Laplace
    prominence_nnls = ui_prominence_nnls.value
    distance_nnls = ui_distance_nnls.value

    nnls_params = {
        'decay_times': decay_times_nnls,
        'prominence' : prominence_nnls,
        'distance'   : distance_nnls,}

    if perform_nnls:

        nnls_fit = nnls_all(processed_correlations_2, nnls_params)

        #merge with basedata
        nnls_data = pd.merge(df_basedata_mod, nnls_fit, on='filename', how='outer')
        nnls_data = nnls_data.reset_index(drop=True)
        nnls_data.index = nnls_data.index + 1

        print(f"\nNNLS: fitted {len(nnls_data)} files.")
    else:
        print("NNLS skipped.")
    return (nnls_data,)


@app.cell
def _(mo):
    ui_export_data_nnls = mo.ui.checkbox(value=False, label="Export to .txt")
    ui_export_data_nnls
    return (ui_export_data_nnls,)


@app.cell
def _(nnls_data, perform_nnls, select_fits_table, ui_export_data_nnls):
    export_data_nnls = ui_export_data_nnls.value

    if perform_nnls:
        ui_table_nnls = select_fits_table(nnls_data, ['filename', 'R_squared'],
                                           label="NNLS — every file fit; deselect any bad ones")
    else:
        ui_table_nnls = None
    ui_table_nnls
    return export_data_nnls, ui_table_nnls


@app.cell
def _(
    experiment_name,
    export_data_nnls,
    maybe_export,
    nnls_data,
    perform_nnls,
    resolve_selection,
    ui_table_nnls,
):
    if perform_nnls:
        nnls_data_mod = resolve_selection(ui_table_nnls, nnls_data, "NNLS")
        fname_nnls = f'NNLS_data_{experiment_name}.txt'
        maybe_export(nnls_data_mod, export_data_nnls, fname_nnls)
    else:
        print("NNLS skipped.")
    return (nnls_data_mod,)


@app.cell
def _(calculate_decay_rates, nnls_data_mod, perform_nnls):
    #calculating decay rates from decay times
    if perform_nnls:

        # Auto-detect tau columns
        tau_cols_nnls = sorted([col for col in nnls_data_mod.columns if col.startswith('tau_')])
        print(f"Detected tau columns: {tau_cols_nnls}")

        nnls_data_rates = calculate_decay_rates(nnls_data_mod, tau_cols_nnls)
        print(f"Computed gamma columns: {[col.replace('tau', 'gamma') for col in tau_cols_nnls]}")
    else:
        print("NNLS skipped.")
    return (nnls_data_rates,)


@app.cell
def _(make_clustering_ui, mo):
    #clustering
    layout_clust_nnls, ui_enable_clust_nnls, ui_normalize_clust_nnls, ui_uncertainty_clust_nnls, \
        ui_distance_clust_nnls, ui_abundance_clust_nnls, ui_strategy_clust_nnls = \
        make_clustering_ui("NNLS", distance_threshold_default=2.0, min_abundance_default=0.3,
                            clustering_strategy_default='silhouette_refined')
    mo.accordion({"NNLS: clustering settings": layout_clust_nnls})
    return (
        ui_abundance_clust_nnls,
        ui_distance_clust_nnls,
        ui_enable_clust_nnls,
        ui_normalize_clust_nnls,
        ui_strategy_clust_nnls,
        ui_uncertainty_clust_nnls,
    )


@app.cell
def _(
    cluster_all_gammas,
    get_reliable_gamma_cols,
    nnls_data_rates,
    perform_nnls,
    ui_abundance_clust_nnls,
    ui_distance_clust_nnls,
    ui_enable_clust_nnls,
    ui_normalize_clust_nnls,
    ui_strategy_clust_nnls,
    ui_uncertainty_clust_nnls,
):
    enable_clustering_nnls = ui_enable_clust_nnls.value
    normalize_by_q2_nnls = ui_normalize_clust_nnls.value
    n_clusters_nnls = 'auto'
    distance_threshold_nnls = ui_distance_clust_nnls.value
    min_abundance_nnls = ui_abundance_clust_nnls.value
    clustering_strategy_nnls = ui_strategy_clust_nnls.value
    uncertainty_flags_nnls = ui_uncertainty_clust_nnls.value
    uncertainty_threshold_nnls = 0.5

    if perform_nnls:

        gamma_cols_nnls = sorted([col for col in nnls_data_rates.columns if col.startswith('gamma_')])

        nnls_data_clustered, cluster_info_nnls = cluster_all_gammas(
            nnls_data_rates,
            gamma_cols            = gamma_cols_nnls,
            q_squared_col         = 'q^2',
            enable_clustering     = enable_clustering_nnls,
            normalize_by_q2       = normalize_by_q2_nnls,
            n_clusters            = n_clusters_nnls,
            distance_threshold    = distance_threshold_nnls,
            min_abundance         = min_abundance_nnls,
            clustering_strategy   = clustering_strategy_nnls,
            uncertainty_flags     = uncertainty_flags_nnls,
            uncertainty_threshold = uncertainty_threshold_nnls,
            plot=True
        )

        reliable_gamma_cols_nnls = get_reliable_gamma_cols(cluster_info_nnls)
        print(f"\n{cluster_info_nnls['n_populations']} reliable population(s) found.")
    else:
        print("NNLS skipped — clustering not executed.")
    return (
        cluster_info_nnls,
        clustering_strategy_nnls,
        gamma_cols_nnls,
        nnls_data_clustered,
        reliable_gamma_cols_nnls,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Clustering sensitivity sweep for NNLS — sweeps `distance_threshold` x
    `min_abundance` and reports populations found + silhouette score at each
    setting. Off by default (25 clustering runs); tick to run one.
    """)
    return


@app.cell
def _(mo):
    ui_run_sweep_nnls = mo.ui.checkbox(label="Run clustering sensitivity sweep for NNLS")
    ui_run_sweep_nnls
    return (ui_run_sweep_nnls,)


@app.cell
def _(
    clustering_sensitivity_sweep,
    clustering_strategy_nnls,
    distance_thresholds,
    gamma_cols_nnls,
    min_abundances,
    nnls_data_rates,
    perform_nnls,
    plot_clustering_heatmaps,
    ui_run_sweep_nnls,
):
    if perform_nnls and ui_run_sweep_nnls.value:
        df_sweep_nnls = clustering_sensitivity_sweep(
            nnls_data_rates, gamma_cols_nnls, 'q^2', clustering_strategy_nnls,
            distance_thresholds, min_abundances)
        fig_sweep_nnls = plot_clustering_heatmaps(df_sweep_nnls, 'NNLS')
        fig_sweep_nnls
    else:
        print("NNLS clustering sweep skipped.")
    return


@app.cell
def _(make_fit_range_ui, mo):
    #linear regression
    ui_restrict_nnls, ui_range_nnls = make_fit_range_ui("NNLS", (0, 0.001), restrict_default=False)
    ui_through_origin_nnls = mo.ui.checkbox(value=False, label="Also fit forced through origin")
    mo.accordion({"NNLS: fit-range settings": mo.vstack([mo.hstack([ui_restrict_nnls, ui_through_origin_nnls]), ui_range_nnls])})
    return ui_range_nnls, ui_restrict_nnls, ui_through_origin_nnls


@app.cell
def _(
    analyze_diffusion_coefficient,
    nnls_data_clustered,
    perform_nnls,
    reliable_gamma_cols_nnls,
    ui_range_nnls,
    ui_restrict_nnls,
    ui_through_origin_nnls,
):
    fit_x_range_nnls = tuple(ui_range_nnls.value) if ui_restrict_nnls.value else None
    fit_through_origin_nnls = ui_through_origin_nnls.value

    if perform_nnls:

        nnls_diff = analyze_diffusion_coefficient(
            data_df       = nnls_data_clustered,
            q_squared_col = 'q^2',
            gamma_cols    = reliable_gamma_cols_nnls,
            x_range            = fit_x_range_nnls,
            fit_through_origin = fit_through_origin_nnls
        )
    else:
        print("NNLS skipped — regression not executed.")
    return (nnls_diff,)


@app.cell
def _(
    c,
    cluster_info_nnls,
    delta_c,
    nnls_diff,
    np,
    pd,
    perform_nnls,
    rh_from_slope,
):
    #calculate results (D, Rh)
    def _results_nnls():
        rows = []
        for idx, _row in nnls_diff.iterrows():
            pop_num = idx + 1
            D, D_err, Rh, Rh_err = rh_from_slope(_row['q^2_coef'], _row['q^2_se'], c, delta_c, unit_factor=1e-18)
            result = {'Fit': f'Population {pop_num}', 'D [m²/s]': D, 'D error [m²/s]': D_err, 'Rh [nm]': Rh, 'Rh error [nm]': Rh_err, 'R_squared': _row['R_squared'], 'intercept': _row['intercept'], 'Residuals': _row['Normality'], 'PDI': np.nan, 'Skewness': np.nan, 'Kurtosis': np.nan, 'Silhouette': cluster_info_nnls['silhouette_score'] if cluster_info_nnls['silhouette_score'] is not None else np.nan}
            if idx < len(cluster_info_nnls['population_abundances']):
                result['Abundance'] = f'{cluster_info_nnls['population_abundances'][idx] * 100:.1f}%'
            rows.append(result)
        return pd.DataFrame(rows)
    if perform_nnls:
        nnls_result = _results_nnls()
    else:
        nnls_result = pd.DataFrame([{'Fit': 'Population 1', 'D [m²/s]': 0, 'D error [m²/s]': 0, 'Rh [nm]': 0, 'Rh error [nm]': 0, 'R_squared': 0, 'intercept': np.nan, 'Residuals': 0, 'PDI': np.nan, 'Skewness': np.nan, 'Kurtosis': np.nan}])
        print('NNLS skipped — zero result placeholder created.')
    nnls_result
    return (nnls_result,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##### ALPHA ANALYSIS
    {finding suitable and physical meaningful regularization parameter for Regularized Fit}
    """)
    return


@app.cell
def _(mo):
    #optional, run before regularized fit
    #plots randomly selected datasets across a range of alpha values to help determine a suitable smoothing parameter for reg. fit
    ui_num_datasets = mo.ui.number(1, 10, value=3, label="Number of random datasets to plot")
    ui_alpha_range = mo.ui.range_slider(0.01, 1.0, step=0.01, value=(0.01, 1.0),
                                         label="Alpha range to test", show_value=True)
    ui_num_alphas = mo.ui.number(2, 10, value=5, label="Number of alpha values across that range")
    mo.accordion({"Alpha analysis: settings": mo.hstack([ui_num_datasets, ui_alpha_range, ui_num_alphas])})
    return ui_alpha_range, ui_num_alphas, ui_num_datasets


@app.cell
def _(
    analyze_alpha,
    analyze_random_datasets_grid,
    nnls_reg_simple,
    np,
    processed_correlations_2,
    ui_alpha_range,
    ui_num_alphas,
    ui_num_datasets,
):
    num_datasets = ui_num_datasets.value
    alpha_range = tuple(ui_alpha_range.value)
    num_alphas = ui_num_alphas.value

    alpha_params = {
        'decay_times': np.logspace(-8, 1, 200),
        'prominence' : 0.005,
        'distance'   : 1,
    }

    if analyze_alpha:
        fig_alpha, selected_datasets = analyze_random_datasets_grid(
            processed_correlations_2,
            num_datasets             = num_datasets,
            base_nnls_params         = alpha_params,
            nnls_reg_simple_function = nnls_reg_simple,
            alpha_range              = alpha_range,
            num_alphas               = num_alphas
        )
    else:
        print("Alpha analysis skipped.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##### REGULARIZED FIT
    {non-negative least squares fit with Tikhonov Regularization}
    """)
    return


@app.cell
def _(mo):
    #regularized fitting
    ui_n_points_reg = mo.ui.slider(50, 400, step=10, value=200, label="Decay-time grid points", show_value=True)
    ui_prominence_reg = mo.ui.slider(0.001, 0.1, step=0.001, value=0.01,
                                      label="Peak prominence (lower = more sensitive)", show_value=True)
    ui_distance_reg = mo.ui.number(1, 10, value=1, label="Min distance between peaks")
    ui_alpha = mo.ui.slider(0.01, 1.0, step=0.01, value=0.2,
                             label="Alpha — smoothing parameter (higher = smoother)", show_value=True)
    ui_fit_beta = mo.ui.checkbox(value=True, label="Fit beta")
    ui_normalize = mo.ui.checkbox(value=True, label="Normalize distribution")
    ui_peak_method = mo.ui.dropdown(options=['maximum', 'centroid'], value='centroid', label="Peak method")
    ui_sparsity_penalty = mo.ui.number(0, 1, step=0.05, value=0, label="Sparsity penalty (0 = disabled)")
    mo.accordion({"Regularized fit: fitting settings": mo.vstack([
        mo.hstack([ui_n_points_reg, ui_prominence_reg, ui_distance_reg]),
        mo.hstack([ui_alpha, ui_fit_beta, ui_normalize]),
        mo.hstack([ui_peak_method, ui_sparsity_penalty]),
    ])})
    return (
        ui_alpha,
        ui_distance_reg,
        ui_fit_beta,
        ui_n_points_reg,
        ui_normalize,
        ui_peak_method,
        ui_prominence_reg,
        ui_sparsity_penalty,
    )


@app.cell
def _(
    df_basedata_mod,
    nnls_reg_all,
    np,
    pd,
    processed_correlations_2,
    regularized_fit,
    ui_alpha,
    ui_distance_reg,
    ui_fit_beta,
    ui_n_points_reg,
    ui_normalize,
    ui_peak_method,
    ui_prominence_reg,
    ui_sparsity_penalty,
):
    decay_times_reg = np.logspace(-8, 1, ui_n_points_reg.value)
    prominence_reg = ui_prominence_reg.value
    distance_reg = ui_distance_reg.value
    alpha = ui_alpha.value
    fit_beta = ui_fit_beta.value
    normalize = ui_normalize.value
    peak_method = ui_peak_method.value
    sparsity_penalty = ui_sparsity_penalty.value

    nnls_reg_params = {
        'decay_times'        : decay_times_reg,
        'prominence'         : prominence_reg,
        'distance'           : distance_reg,
        'alpha'              : alpha,
        'fit_beta'           : fit_beta,
        'normalize'          : normalize,
        'peak_method'        : peak_method,
        'sparsity_penalty'   : sparsity_penalty,
    }

    if regularized_fit:
        nnls_reg_df, full_results = nnls_reg_all(processed_correlations_2, nnls_reg_params)

        #merge with basedata
        nnls_reg_data = pd.merge(df_basedata_mod, nnls_reg_df, on='filename', how='outer')
        nnls_reg_data = nnls_reg_data.reset_index(drop=True)
        nnls_reg_data.index = nnls_reg_data.index + 1

        print(f"\nRegularized fit: fitted {len(nnls_reg_data)} files.")
    else:
        print("Regularized fit skipped.")
    return full_results, nnls_reg_data, nnls_reg_params


@app.cell
def _(mo):
    ui_export_data_reg = mo.ui.checkbox(value=False, label="Export to .txt")
    ui_export_data_reg
    return (ui_export_data_reg,)


@app.cell
def _(nnls_reg_data, regularized_fit, select_fits_table, ui_export_data_reg):
    export_data_reg = ui_export_data_reg.value

    if regularized_fit:
        ui_table_reg = select_fits_table(nnls_reg_data, ['filename', 'R_squared'],
                                          label="Regularized fit — every file fit; deselect any bad ones")
    else:
        ui_table_reg = None
    ui_table_reg
    return export_data_reg, ui_table_reg


@app.cell
def _(
    experiment_name,
    export_data_reg,
    maybe_export,
    nnls_reg_data,
    regularized_fit,
    resolve_selection,
    ui_table_reg,
):
    if regularized_fit:
        nnls_reg_data_mod = resolve_selection(ui_table_reg, nnls_reg_data, "Regularized fit")
        fname_reg = f'Regularized_data_{experiment_name}.txt'
        maybe_export(nnls_reg_data_mod, export_data_reg, fname_reg)
    else:
        print("Regularized fit skipped.")
    return (nnls_reg_data_mod,)


@app.cell
def _(calculate_decay_rates, nnls_reg_data_mod, regularized_fit):
    #calculating decay rates from decay times
    if regularized_fit:

        #auto-detect tau columns
        tau_cols_reg = sorted([col for col in nnls_reg_data_mod.columns if col.startswith('tau_')])
        print(f"Detected tau columns: {tau_cols_reg}")

        nnls_reg_data_rates = calculate_decay_rates(nnls_reg_data_mod, tau_cols_reg)
        print(f"Computed gamma columns: {[col.replace('tau', 'gamma') for col in tau_cols_reg]}")
    else:
        print("Regularized fit skipped.")
    return (nnls_reg_data_rates,)


@app.cell
def _(make_clustering_ui, mo):
    #clustering
    layout_clust_reg, ui_enable_clust_reg, ui_normalize_clust_reg, ui_uncertainty_clust_reg, \
        ui_distance_clust_reg, ui_abundance_clust_reg, ui_strategy_clust_reg = \
        make_clustering_ui("Regularized fit", distance_threshold_default=1.0, min_abundance_default=0.4,
                            clustering_strategy_default='silhouette_refined')
    mo.accordion({"Regularized fit: clustering settings": layout_clust_reg})
    return (
        ui_abundance_clust_reg,
        ui_distance_clust_reg,
        ui_enable_clust_reg,
        ui_normalize_clust_reg,
        ui_strategy_clust_reg,
        ui_uncertainty_clust_reg,
    )


@app.cell
def _(
    aggregate_peak_stats,
    cluster_all_gammas,
    get_reliable_gamma_cols,
    nnls_reg_data_rates,
    regularized_fit,
    ui_abundance_clust_reg,
    ui_distance_clust_reg,
    ui_enable_clust_reg,
    ui_normalize_clust_reg,
    ui_strategy_clust_reg,
    ui_uncertainty_clust_reg,
):
    enable_clustering_reg = ui_enable_clust_reg.value
    normalize_by_q2_reg = ui_normalize_clust_reg.value
    n_clusters_reg = 'auto'
    distance_threshold_reg = ui_distance_clust_reg.value
    min_abundance_reg = ui_abundance_clust_reg.value
    clustering_strategy_reg = ui_strategy_clust_reg.value
    uncertainty_flags_reg = ui_uncertainty_clust_reg.value
    uncertainty_threshold_reg = 0.5

    if regularized_fit:

        gamma_cols_reg = sorted([col for col in nnls_reg_data_rates.columns
                             if col.startswith('gamma_')])

        nnls_reg_data_clustered, cluster_info_reg = cluster_all_gammas(
            nnls_reg_data_rates,
            gamma_cols            = gamma_cols_reg,
            q_squared_col         = 'q^2',
            enable_clustering     = enable_clustering_reg,
            normalize_by_q2       = normalize_by_q2_reg,
            n_clusters            = n_clusters_reg,
            distance_threshold    = distance_threshold_reg,
            min_abundance         = min_abundance_reg,
            clustering_strategy   = clustering_strategy_reg,
            uncertainty_flags     = uncertainty_flags_reg,
            uncertainty_threshold = uncertainty_threshold_reg,
            plot=True)

        #per-population skewness/kurtosis averaged across files — tau-space, flip sign to compare with cumulants
        cluster_info_reg = aggregate_peak_stats(cluster_info_reg, nnls_reg_data_rates)

        reliable_gamma_cols_reg = get_reliable_gamma_cols(cluster_info_reg)
        print(f"\n{cluster_info_reg['n_populations']} reliable population(s) found.")
    else:
        print("Regularized fit skipped — clustering not executed.")
    return (
        cluster_info_reg,
        clustering_strategy_reg,
        gamma_cols_reg,
        nnls_reg_data_clustered,
        reliable_gamma_cols_reg,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Clustering sensitivity sweep for the regularized fit — sweeps
    `distance_threshold` x `min_abundance` and reports populations found +
    silhouette score at each setting. Off by default (25 clustering runs);
    tick to run one.
    """)
    return


@app.cell
def _(mo):
    ui_run_sweep_reg = mo.ui.checkbox(label="Run clustering sensitivity sweep for Regularized fit")
    ui_run_sweep_reg
    return (ui_run_sweep_reg,)


@app.cell
def _(
    clustering_sensitivity_sweep,
    clustering_strategy_reg,
    distance_thresholds,
    gamma_cols_reg,
    min_abundances,
    nnls_reg_data_rates,
    plot_clustering_heatmaps,
    regularized_fit,
    ui_run_sweep_reg,
):
    if regularized_fit and ui_run_sweep_reg.value:
        df_sweep_reg = clustering_sensitivity_sweep(
            nnls_reg_data_rates, gamma_cols_reg, 'q^2', clustering_strategy_reg,
            distance_thresholds, min_abundances)
        fig_sweep_reg = plot_clustering_heatmaps(df_sweep_reg, 'Regularized fit')
        fig_sweep_reg
    else:
        print("Regularized fit clustering sweep skipped.")
    return


@app.cell
def _(make_fit_range_ui, mo):
    #linear regression
    ui_restrict_reg, ui_range_reg = make_fit_range_ui("Regularized fit", (0, 0.001), restrict_default=False)
    ui_through_origin_reg = mo.ui.checkbox(value=False, label="Also fit forced through origin")
    mo.accordion({"Regularized fit: fit-range settings": mo.vstack([mo.hstack([ui_restrict_reg, ui_through_origin_reg]), ui_range_reg])})
    return ui_range_reg, ui_restrict_reg, ui_through_origin_reg


@app.cell
def _(
    analyze_diffusion_coefficient,
    nnls_reg_data_clustered,
    regularized_fit,
    reliable_gamma_cols_reg,
    ui_range_reg,
    ui_restrict_reg,
    ui_through_origin_reg,
):
    fit_x_range_reg = tuple(ui_range_reg.value) if ui_restrict_reg.value else None
    fit_through_origin_reg = ui_through_origin_reg.value

    if regularized_fit:

        nnls_reg_diff = analyze_diffusion_coefficient(
            data_df       = nnls_reg_data_clustered,
            q_squared_col = 'q^2',
            gamma_cols    = reliable_gamma_cols_reg,
            x_range            = fit_x_range_reg,
            fit_through_origin = fit_through_origin_reg
        )
    else:
        print("Regularized fit skipped — regression not executed.")
    return (nnls_reg_diff,)


@app.cell
def _(
    c,
    cluster_info_reg,
    delta_c,
    nnls_reg_diff,
    np,
    pd,
    regularized_fit,
    rh_from_slope,
):
    #calculate results (D, Rh)
    def _results_reg():
        rows = []
        for idx, _row in nnls_reg_diff.iterrows():
            pop_num = idx + 1
            D, D_err, Rh, Rh_err = rh_from_slope(_row['q^2_coef'], _row['q^2_se'], c, delta_c, unit_factor=1e-18)
            result = {'Fit': f'Population {pop_num}', 'D [m²/s]': D, 'D error [m²/s]': D_err, 'Rh [nm]': Rh, 'Rh error [nm]': Rh_err, 'R_squared': _row['R_squared'], 'intercept': _row['intercept'], 'Residuals': _row['Normality'], 'PDI': np.nan, 'Skewness': cluster_info_reg.get('population_skewness_mean', {}).get(pop_num, np.nan), 'Kurtosis': cluster_info_reg.get('population_kurtosis_mean', {}).get(pop_num, np.nan), 'Silhouette': cluster_info_reg['silhouette_score'] if cluster_info_reg['silhouette_score'] is not None else np.nan}
            if idx < len(cluster_info_reg['population_abundances']):
                result['Abundance'] = f'{cluster_info_reg['population_abundances'][idx] * 100:.1f}%'
            rows.append(result)
        return pd.DataFrame(rows)
    if regularized_fit:
        nnls_reg_result = _results_reg()
    else:
        nnls_reg_result = pd.DataFrame([{'Fit': 'Population 1', 'D [m²/s]': 0, 'D error [m²/s]': 0, 'Rh [nm]': 0, 'Rh error [nm]': 0, 'R_squared': 0, 'intercept': np.nan, 'Residuals': 0, 'PDI': np.nan, 'Skewness': 0, 'Kurtosis': 0}])
        print('Regularized fit skipped — zero result placeholder created.')
    nnls_reg_result  # Note: Skewness and Kurtosis are computed in tau-space (weighted moments of the decay time distribution).  # Skewness > 0 -> tail toward larger tau -> larger particles.  # To compare with cumulant skewness (Gamma-space): flip the sign.
    return (nnls_reg_result,)


@app.cell
def _(mo):
    #possible distribution plots for regularized fit
    ui_plot_reg_fit_distr = mo.ui.checkbox(value=False, label="Plot regularized-fit distributions")
    ui_plot_reg_fit_distr
    return (ui_plot_reg_fit_distr,)


@app.cell
def _(
    full_results,
    nnls_reg_data,
    nnls_reg_params,
    plot_distributions,
    regularized_fit,
    ui_plot_reg_fit_distr,
):
    plot_reg_fit_distr = ui_plot_reg_fit_distr.value

    angles_dist          = [90]      # angles to plot [°]
    measurement_mode_dist= 'all'       # 'first', 'average', or 'all'
    convert_to_radius = True          # plot Rh [nm] instead of decay time tau
    figsize_dist      = (8, 6)
    xlim_regplot      = (10, 1000)      # adjust range of x-axis !pay attention if you plot tau or Rh!
    title_dist        = ''

    # set filenames to override angle/measurement_mode selection:
    #filenames = None
    filenames_dist = ['ca1-5mgmL_ca0-1mMKCl_pH4-77_messung2.ASC',
                  'ca1-5mgmL_ohneSalz_ohnePuffer.ASC',
                 'ca1-5mgmL_ohneSalz_pH4-77.ASC',
                ]

    if plot_reg_fit_distr:
        if regularized_fit:
            plot_distributions(
                full_results, nnls_reg_params, nnls_reg_data,
                angles            = angles_dist,
                measurement_mode  = measurement_mode_dist,
                convert_to_radius = convert_to_radius,
                figsize           = figsize_dist,
                xlim              = xlim_regplot,
                title             = title_dist,
                filenames         = filenames_dist)
        else:
            print("Regularized fit skipped — distribution plot not available.")
    else:
        print("Plotting of distribution functions disabled.")
    return


@app.cell
def _(mo):
    #intensity & guinier analysis (perform_intensity_processing and regularized_fit have to be set True)
    ui_perform_sls = mo.ui.checkbox(value=True, label="I_pop_i = I_total x (normalized area % pop_i / 100), plot vs angle")
    ui_log_y = mo.ui.checkbox(value=True, label="Log-scale y-axis")
    ui_perform_guinier = mo.ui.checkbox(value=True, label="Perform Guinier fit")
    ui_guinier_q2_range_total = mo.ui.range_slider(0, 0.001, step=0.00002, value=(0, 0.0002),
                                                    label="Guinier fit range for total intensity [q², nm⁻²]",
                                                    show_value=True)
    ui_exponent = mo.ui.number(1, 10, value=6, label="Exponent (6=compact spheres, 5=Daoud-Cotton star polymers)")
    mo.accordion({"SLS / Guinier settings": mo.vstack([
        mo.hstack([ui_perform_sls, ui_log_y, ui_perform_guinier]),
        ui_guinier_q2_range_total,
        ui_exponent,
    ])})
    return (
        ui_exponent,
        ui_guinier_q2_range_total,
        ui_log_y,
        ui_perform_guinier,
        ui_perform_sls,
    )


@app.cell
def _(
    cluster_info_reg,
    compute_guinier_extrapolation,
    compute_guinier_total,
    compute_sls_data,
    df_intensity,
    experiment_name,
    nnls_reg_data_clustered,
    nnls_reg_result,
    perform_intensity_processing,
    plot_guinier,
    plot_sls_intensity,
    regularized_fit,
    summarize_sls_combined,
    ui_exponent,
    ui_guinier_q2_range_total,
    ui_log_y,
    ui_perform_guinier,
    ui_perform_sls,
):
    perform_sls = ui_perform_sls.value
    log_y = ui_log_y.value
    guinier_q2_range = None  # None \ (q2_min, q2_max) \ {1:(a,b), 2:(c,d)} — advanced/per-population override, edit here directly
    guinier_q2_range_total = tuple(ui_guinier_q2_range_total.value)
    perform_guinier = ui_perform_guinier.value
    exponent = ui_exponent.value
    #e.g. with Rh_1 = 10 nm, Rh_2 = 83 nm, exponent = 6: the factor is (83/10)^6 ~ 350,000
    if perform_sls and regularized_fit and perform_intensity_processing:
        n_populations_sls = cluster_info_reg['n_populations']
        sls_data = compute_sls_data(nnls_reg_data_clustered, df_intensity, n_populations_sls)
        rh_values = {}
        for idx, _row in nnls_reg_result.iterrows():
            rh_values[idx + 1] = _row['Rh [nm]']
        plot_sls_intensity(sls_data, n_populations_sls, experiment_name, log_y=log_y)
        total_result = compute_guinier_total(sls_data, q2_range=guinier_q2_range_total)  #extract Rh from nnls_reg_result
        if perform_guinier:
            guinier_results = compute_guinier_extrapolation(sls_data, n_populations_sls, q2_range=guinier_q2_range)
            plot_guinier(guinier_results, experiment_name, total_result=total_result)
            sls_summary = summarize_sls_combined(sls_data, guinier_results, total_result, n_populations_sls, rh_values, exponent=exponent)
            sls_summary
    else:
        print('SLS analysis skipped.')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### IV. RESULTS OVERVIEW
    """)
    return


@app.cell
def _():
    #COLUMN DESCRIPTIONS
      #Method        Analysis method used (A, B, C = single-mode; D, NNLS, Regularized = multi-mode)
      #Fit           Specific fit variant (e.g. cumulant order, population number)

      #Rh [nm]       Hydrodynamic radius — from Stokes-Einstein: Rh = kT / (6*pi*eta*D)
      #Rh error [nm] Propagated uncertainty from D and kT/6*pi*eta (T+eta)

      #D [m²/s]      Translational diffusion coefficient — slope of Gamma vs q² regression
      #D error [m²/s]Standard error of the slope from OLS regression

      #R_squared     Goodness of fit of the Gamma vs q² linear regression
      #Residuals     Normality assessment of regression residuals (based on Jarque-Bera + Omnibus tests)

      #PDI           Polydispersity index = mu2/Gamma² — measure of size distribution width
      #                < 0.05: highly monodisperse
      #               0.05-0.20: moderately monodisperse
      #                > 0.30: broad — most likely multimodal

      #Skewness      skewness = mu3/mu2^(3/2)
      #              For Cumulant Methods: Asymmetry of total gamma distribution; NOT per population!; Reg. Fit opposite sign!
      #              For Reg. Fit: Asymmetry of the decay-time distribution (tau-space) [~ 0: symmetric,  > 0: tail toward larger particles]

      #Kurtosis      kurtosis = mu4/mu2^2
      #              For Cumulant Methods: Kurtosis of total gamma distribution; NOT per population!; Reg. Fit opposite sign!
      #              For Reg. Fit: Kurtosis of the decay-time distribution peak shape [> 0: sharp/narrow peak,  < 0: broad/flat peak]
        # NOTE for cumulant C: skewness = d/c^(3/2) and kurtosis = e/c^2 can become very large or NaN when c (2nd cumulant) is small or negative
        # this occurs for near-monodisperse samples where higher cumulants fit noise rather than real distribution width.
        # Values should be interpreted cautiously when PDI is low.

      #Abundance     Fraction of files in which this population was detected [0-1]
      #                e.g. 0.85 = present in 85% of measurements across all angles

      #Silhouette    Clustering quality score [-1 to 1]
      #                > 0.7: well-separated populations
      #                0.5-0.7: reasonable separation
      #                < 0.5: overlapping populations — interpret with care
    return


@app.cell
def _(
    method_A_cumulant_result,
    method_B_cumulant_result,
    method_C_cumulant_result,
    pd,
    perform_cumulant_A,
    perform_cumulant_B,
    perform_cumulant_C,
):
    #SINGLE-MODE COMPARISON TABLE

    def tag_method(df, method):
        out = df.copy()
        out.insert(0, 'Method', method)
        return out

    single_mode_frames = []
    if perform_cumulant_A:
        single_mode_frames.append(tag_method(method_A_cumulant_result, 'A'))
    if perform_cumulant_B:
        single_mode_frames.append(tag_method(method_B_cumulant_result, 'B'))
    if perform_cumulant_C:
        single_mode_frames.append(tag_method(method_C_cumulant_result, 'C'))

    if single_mode_frames:
        df_single_mode_results = pd.concat(single_mode_frames, ignore_index=True)
    else:
        df_single_mode_results = pd.DataFrame()
        print("No single-mode results available.")

    df_single_mode_results
    return df_single_mode_results, tag_method


@app.cell
def _(
    method_D_cumulant_result,
    nnls_reg_result,
    nnls_result,
    pd,
    perform_cumulant_D,
    perform_nnls,
    regularized_fit,
    tag_method,
):
    #MULTI-MODE COMPARISON TABLE

    multi_mode_frames = []
    if perform_cumulant_D:
        multi_mode_frames.append(tag_method(method_D_cumulant_result, 'D'))
    if perform_nnls:
        multi_mode_frames.append(tag_method(nnls_result, 'NNLS'))
    if regularized_fit:
        multi_mode_frames.append(tag_method(nnls_reg_result, 'Regularized'))

    if multi_mode_frames:
        df_multi_mode_results = pd.concat(multi_mode_frames, ignore_index=True)
    else:
        df_multi_mode_results = pd.DataFrame()
        print("No multi-mode results available.")

    df_multi_mode_results
    return (df_multi_mode_results,)


@app.cell
def _(df_multi_mode_results, df_single_mode_results, experiment_name, np, plt):
    #Rh COMPARISON PLOT
    Rh_wmean = np.nan
    #external benchmark values (e.g. from a manufacturer spec sheet) — not computed in this pipeline
    Rh_wstd = np.nan
    fig_rh, axes_rh = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes_rh[0]
    if not df_single_mode_results.empty:
        sm = df_single_mode_results[df_single_mode_results['Rh [nm]'] > 0].copy().reset_index(drop=True)
    # --- Single-mode ---
        sm_colors = {'A': '#2196F3', 'B': '#4CAF50', 'C': '#FF9800'}
        sm_markers = {'A': 'o', 'B': 's', 'C': '^'}
        seen_methods = set()
        for i, _row in sm.iterrows():
            method = _row['Method']
            ax.errorbar(i, _row['Rh [nm]'], yerr=_row['Rh error [nm]'], fmt=sm_markers.get(method, 'o'), color=sm_colors.get(method, 'gray'), capsize=4, markersize=7, label=method if method not in seen_methods else '')
            seen_methods.add(method)
        ax.set_xticks(range(len(sm)))
        ax.set_xticklabels(sm['Fit'], rotation=30, ha='right', fontsize=8)
        if not np.isnan(Rh_wmean):
            ax.axhline(Rh_wmean, color='black', linestyle='--', linewidth=1, label=f'Weighted mean ({Rh_wmean:.1f} nm)')
            ax.fill_between(range(len(sm)), Rh_wmean - Rh_wstd, Rh_wmean + Rh_wstd, alpha=0.1, color='black')
    ax.set_ylabel('Rh [nm]')
    ax.set_title('Single-mode methods (A, B, C)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax = axes_rh[1]
    if not df_multi_mode_results.empty:
        mm = df_multi_mode_results[df_multi_mode_results['Rh [nm]'] > 0].copy().reset_index(drop=True)
        mm_colors = {'D': '#9C27B0', 'NNLS': '#F44336', 'Regularized': '#009688'}
        mm_markers = {'D': 'D', 'NNLS': 'v', 'Regularized': 'P'}
        seen_methods = set()
        for i, _row in mm.iterrows():
            method = _row['Method']
            ax.errorbar(i, _row['Rh [nm]'], yerr=_row['Rh error [nm]'], fmt=mm_markers.get(method, 'o'), color=mm_colors.get(method, 'gray'), capsize=4, markersize=7, label=method if method not in seen_methods else '')
            seen_methods.add(method)
        ax.set_xticks(range(len(mm)))
        ax.set_xticklabels(mm['Fit'], rotation=30, ha='right', fontsize=8)
    # --- Multi-mode ---
    ax.set_ylabel('Rh [nm]')
    ax.set_title('Multi-mode methods (D, NNLS, Regularized)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.suptitle(f'{experiment_name} — Rh comparison', fontsize=12)
    plt.tight_layout()
    fig_rh
    return


if __name__ == "__main__":
    app.run()
