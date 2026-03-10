# -*- coding: utf-8 -*-
"""
sls_functions_for_regularized.py
=================================
Static light scattering (SLS) analysis functions for population-resolved
intensity analysis from regularized DLS fits.

Functions
---------
compute_sls_data(nnls_reg_data_mod, df_intensity, n_populations)
    area-fraction decomposition:
    I_pop = I_total * normalized_area_percent_N / 100

compute_sls_data_number_weighted(sls_data, n_populations, rh_values, exponent)
    number-fraction correction:
    c_pop = I_pop / Rh^exponent → normalised → I_pop_nw = I_total * c_frac
    Removes the Rh^exponent intensity bias to give number-weighted fractions.

compute_guinier_total(sls_data, q2_range)
    Guinier fit on the raw total intensity I_total vs q².
    Reference fit independent of any population decomposition.

compute_guinier_extrapolation(sls_data, n_populations, q2_range)
    Guinier analysis per population: ln(I_pop) vs q² linear fit.
    Yields I₀ and Rg per population.

plot_sls_intensity(sls_data, n_populations, experiment_name, colors, log_y)
    I_pop vs angle per population with mean ± std error bars.

plot_guinier(guinier_results, experiment_name, colors, total_result)
    Guinier plot: ln(I_pop) vs q² with optional total intensity reference.

summarize_sls(sls_data, guinier_results, n_populations)
    Summary DataFrame with I₀, Rg, qRg_max and R² per population.

Dependencies: numpy, pandas, scipy, matplotlib
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats


#shared palette & style helpers

_POP_COLORS  = ['#2C7BB6', '#D7191C', '#1A9641', '#F98400']
_POP_MARKERS = ['o', 's', '^', 'D']
_TOTAL_COLOR  = '#555555'

def _style_ax(ax):
    ax.grid(True, alpha=0.3, linewidth=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for spine in ('left', 'bottom'):
        ax.spines[spine].set_linewidth(0.8)
    ax.tick_params(labelsize=9, direction='in', length=4)


def _guinier_fit(grouped, grouped_fit, label):
    """
    Shared Guinier fit logic. Returns result dict or None on failure.
    """
    if len(grouped_fit) < 3:
        print(f"{label}: fewer than 3 valid angle points in fit range — skipping.")
        return None

    ln_I = np.log(grouped_fit['I_mean'])
    slope, intercept, r, _, _ = stats.linregress(grouped_fit['q2'], ln_I)

    I0  = np.exp(intercept)
    rg2 = -3.0 * slope
    Rg  = np.sqrt(rg2) if rg2 > 0 else np.nan

    q_max   = np.sqrt(grouped_fit['q2'].max())
    qRg_max = q_max * Rg if not np.isnan(Rg) else np.nan

    if not np.isnan(qRg_max) and qRg_max > 1.3:
        print(f"{label}: qRg_max = {qRg_max:.2f} > 1.3 — "
              f"Guinier approximation may not be valid at high angles.")

    return {
        'slope'       : slope,
        'intercept'   : intercept,
        'R2'          : r**2,
        'I0 [kHz]'    : I0,
        'Rg [nm]'     : Rg,
        'qRg_max'     : qRg_max,
        'grouped'     : grouped,
        'grouped_fit' : grouped_fit,
        'ln_I'        : ln_I,
    }


#decomposition functions

def compute_sls_data(nnls_reg_data_mod, df_intensity, n_populations):
    """
    Method A — area-fraction decomposition.

    I_pop = I_total * normalized_area_percent_N / 100

    Uses the peak areas from the regularized tau-distribution directly.
    Simple but sensitive to fit-to-fit variation in how peaks are split.
    """
    intensity_cols = ['filename', 'MeanCR_corr [kHz]']
    if 'angle [°]' not in nnls_reg_data_mod.columns and 'angle [°]' in df_intensity.columns:
        intensity_cols.append('angle [°]')

    sls_data = nnls_reg_data_mod.merge(
        df_intensity[intensity_cols], on='filename', how='left')

    for i in range(1, n_populations + 1):
        area_col = f'normalized_area_percent_{i}'
        if area_col in sls_data.columns:
            sls_data[f'I_pop{i} [kHz]'] = (
                sls_data['MeanCR_corr [kHz]'] * sls_data[area_col] / 100
            )
        else:
            print(f"Warning: '{area_col}' not found — population {i} skipped.")

    return sls_data


def compute_sls_data_number_weighted(sls_data, n_populations, rh_values,
                                      exponent=6):
    """
    number/concentration-fraction correction.

    Converts the intensity-weighted I_popN (from area fractions)
    to number-weighted intensities using the DLS-derived Rh per population:

        c_pop_i  = I_pop_i  /  Rh_i ^ exponent        (per file)
        c_frac_i = c_pop_i  /  sum_j(c_pop_j)          (normalised, per file)
        I_pop_i_nw = I_total * c_frac_i                 (number-weighted intensity)

    This follows: c ~ I / R^exponent
        exponent = 5  Daoud-Cotton scaling for star polymers (M ~ R^5)
        exponent = 6  Rayleigh scattering for compact spheres (I ~ R^6)

    The correction is applied per file, so file-to-file variation in the
    area fractions is preserved but the Rh^exponent bias is removed.
    The small population is no longer suppressed by the size ratio.
    """
    #validate that Method A columns exist
    for i in range(1, n_populations + 1):
        if f'I_pop{i} [kHz]' not in sls_data.columns:
            print(f"Error: 'I_pop{i} [kHz]' not found — run compute_sls_data first.")
            return sls_data

    #per-file number-fraction correction
    c_cols = []
    for i in range(1, n_populations + 1):
        rh = rh_values.get(i, np.nan)
        if np.isnan(rh) or rh <= 0:
            print(f"Warning: invalid Rh for population {i} — skipped.")
            sls_data[f'c_pop{i}_rel'] = np.nan
        else:
            sls_data[f'c_pop{i}_rel'] = sls_data[f'I_pop{i} [kHz]'] / (rh ** exponent)
        c_cols.append(f'c_pop{i}_rel')

    #normalise to get per-file concentration fractions
    c_total = sls_data[c_cols].sum(axis=1).replace(0, np.nan)
    for i in range(1, n_populations + 1):
        if f'c_pop{i}_rel' not in sls_data.columns:
            continue
        c_frac = sls_data[f'c_pop{i}_rel'] / c_total
        sls_data[f'I_pop{i}_nw [kHz]'] = sls_data['MeanCR_corr [kHz]'] * c_frac

    #report mean fractions for reference
    print(f"\nnumber-fraction correction (exponent={exponent})")
    print(f"{'Population':<12} {'Rh [nm]':<12} {'Mean I_frac (A)':<18} {'Mean c_frac (B)':<18}")
    for i in range(1, n_populations + 1):
        rh    = rh_values.get(i, np.nan)
        i_col = f'I_pop{i} [kHz]'
        nw_col= f'I_pop{i}_nw [kHz]'
        i_frac = (sls_data[i_col] / sls_data['MeanCR_corr [kHz]']).mean() * 100 \
                 if i_col in sls_data.columns else np.nan
        c_frac = (sls_data[nw_col] / sls_data['MeanCR_corr [kHz]']).mean() * 100 \
                 if nw_col in sls_data.columns else np.nan
        print(f"  Pop {i:<8} {rh:<12.1f} {i_frac:<18.1f} {c_frac:<18.1f}")

    return sls_data


# ── Guinier functions ─────────────────────────────────────────────────────────

def compute_guinier_total(sls_data, q2_range=None):
    """
    Guinier fit on the raw total intensity — reference fit independent
    of any population decomposition.

        ln I_total(q) = ln I₀  −  (Rg²/3) · q²

    Parameters
    ----------
    sls_data : pd.DataFrame
        Must contain MeanCR_corr [kHz], q^2, angle [°]
    q2_range : tuple (q2_min, q2_max) or None

    Returns
    -------
    dict  same structure as individual population Guinier results
    """
    grouped = sls_data.groupby('angle [°]').agg(
        q2     = ('q^2', 'mean'),
        I_mean = ('MeanCR_corr [kHz]', 'mean'),
        I_std  = ('MeanCR_corr [kHz]', 'std'),
    ).dropna(subset=['q2', 'I_mean'])
    grouped = grouped[grouped['I_mean'] > 0]

    grouped_fit = grouped.copy()
    if q2_range is not None:
        q2_min, q2_max = q2_range
        grouped_fit = grouped_fit[
            (grouped_fit['q2'] >= q2_min) & (grouped_fit['q2'] <= q2_max)
        ]
        print(f"Total intensity Guinier restricted to "
              f"q² = [{q2_min:.2e}, {q2_max:.2e}] nm⁻² "
              f"({len(grouped_fit)}/{len(grouped)} angles used)")

    return _guinier_fit(grouped, grouped_fit, "Total intensity")


def compute_guinier_extrapolation(sls_data, n_populations, q2_range=None,
                                   use_nw_columns=False):
    """
    Guinier analysis per population: ln(I_pop) vs q² linear fit.

        ln I(q) = ln I₀  −  (Rg²/3) · q²

    Parameters
    ----------
    sls_data : pd.DataFrame
    n_populations : int
    q2_range : tuple, dict, or None
        - None             : use all angles for all populations
        - (q2_min, q2_max) : same range applied to all populations
        - {1: (a, b), 2: (c, d), ...} : per-population ranges
    use_nw_columns : bool
        If True, use I_popN_nw [kHz] columns (Method B, number-weighted)
        instead of I_popN [kHz] (Method A, intensity-weighted).

    Returns
    -------
    dict  guinier_results[i] = {slope, intercept, R2, I0, Rg, qRg_max,
                                 grouped, grouped_fit, ln_I}
    """
    guinier_results = {}
    col_suffix = '_nw [kHz]' if use_nw_columns else ' [kHz]'

    for i in range(1, n_populations + 1):
        i_col = f'I_pop{i}{col_suffix}'
        if i_col not in sls_data.columns:
            print(f"Population {i}: column '{i_col}' not found — skipping.")
            continue

        grouped = sls_data.groupby('angle [°]').agg(
            q2     = ('q^2', 'mean'),
            I_mean = (i_col, 'mean'),
            I_std  = (i_col, 'std'),
        ).dropna(subset=['q2', 'I_mean'])
        grouped = grouped[grouped['I_mean'] > 0]

        # resolve q² range for this population
        if isinstance(q2_range, dict):
            pop_range = q2_range.get(i, None)
        else:
            pop_range = q2_range

        grouped_fit = grouped.copy()
        if pop_range is not None:
            q2_min, q2_max = pop_range
            grouped_fit = grouped_fit[
                (grouped_fit['q2'] >= q2_min) & (grouped_fit['q2'] <= q2_max)
            ]
            print(f"Population {i}: Guinier fit restricted to "
                  f"q² = [{q2_min:.2e}, {q2_max:.2e}] nm⁻² "
                  f"({len(grouped_fit)}/{len(grouped)} angles used)")

        result = _guinier_fit(grouped, grouped_fit, f"Population {i}")
        if result is not None:
            guinier_results[i] = result

    return guinier_results


#plotting

def plot_sls_intensity(sls_data, n_populations, experiment_name='',
                       colors=None, log_y=False):
    """
    Single-panel: I_pop vs angle per population, mean ± std error bars.
    """
    if colors is None:
        colors = _POP_COLORS

    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.suptitle(f'{experiment_name} — Population-resolved scattering intensity',
                 fontsize=11, fontweight='semibold', y=1.01)

    for i in range(1, n_populations + 1):
        i_col = f'I_pop{i} [kHz]'
        if i_col not in sls_data.columns:
            continue
        g = sls_data.groupby('angle [°]')[i_col].agg(['mean', 'std'])
        ax.errorbar(
            g.index, g['mean'], yerr=g['std'],
            fmt=_POP_MARKERS[i - 1] + '-',
            color=colors[i - 1],
            capsize=3, capthick=0.8,
            linewidth=1.4, markersize=5,
            label=f'Population {i}'
        )

    if log_y:
        ax.set_yscale('log')
    ax.set_xlabel('Angle [°]', fontsize=10)
    ax.set_ylabel('$I_{pop}$ [kHz]', fontsize=10)
    ax.legend(fontsize=9, framealpha=0.85, edgecolor='#cccccc')
    _style_ax(ax)
    plt.tight_layout()
    plt.show()


def _plot_guinier_population(ax, idx, pop, res, colors):
    c           = colors[idx]
    grouped     = res['grouped']
    grouped_fit = res['grouped_fit']

    q2_fit   = np.linspace(0, grouped_fit['q2'].max() * 1.08, 200)
    ln_I_fit = res['slope'] * q2_fit + res['intercept']

    ln_I_all = np.log(grouped['I_mean'])
    ln_I_err = (grouped['I_std'] / grouped['I_mean']).fillna(0)
    in_fit   = grouped.index.isin(grouped_fit.index)

    # faded excluded points
    if (~in_fit).any():
        ax.errorbar(
            grouped['q2'][~in_fit], ln_I_all[~in_fit],
            yerr=ln_I_err[~in_fit],
            fmt=_POP_MARKERS[idx % len(_POP_MARKERS)], color=c,
            capsize=3, capthick=0.6, markersize=5, linewidth=0,
            alpha=0.25, zorder=2
        )

    ax.errorbar(
        grouped['q2'][in_fit], ln_I_all[in_fit],
        yerr=ln_I_err[in_fit],
        fmt=_POP_MARKERS[idx % len(_POP_MARKERS)], color=c,
        capsize=3, capthick=0.8, markersize=5, linewidth=0, zorder=3
    )
    ax.plot(q2_fit, ln_I_fit, '-', color=c, linewidth=1.5, alpha=0.85)
    ax.scatter([0], [res['intercept']], color=c,
               marker='*', s=120, zorder=5, edgecolors='white', linewidth=0.5)

    Rg_str = f'{res["Rg [nm]"]:.1f} nm' if not np.isnan(res['Rg [nm]']) else 'N/A'
    I0_str = f'{res["I0 [kHz]"]:.4f} kHz'
    label  = (f'Pop {pop}   $R_g$ = {Rg_str}   '
              f'$I_0$ = {I0_str}   $R^2$ = {res["R2"]:.4f}')
    ax.plot([], [], color=c, linewidth=2, label=label)


def plot_guinier(guinier_results, experiment_name='', colors=None,
                 total_result=None):
    """
    Guinier plot: ln(I_pop) vs q² with linear fits, extrapolation to q=0,
    and optional total intensity reference.
    """
    if colors is None:
        colors = _POP_COLORS

    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.suptitle(f'{experiment_name} Guinier analysis',
                 fontsize=11, fontweight='semibold', y=1.01)

    # total intensity reference (drawn first, behind population curves)
    if total_result is not None:
        tr      = total_result
        grouped = tr['grouped']
        gfit    = tr['grouped_fit']
        ln_I_all = np.log(grouped['I_mean'])
        ln_I_err = (grouped['I_std'] / grouped['I_mean']).fillna(0)
        in_fit   = grouped.index.isin(gfit.index)

        q2_fit   = np.linspace(0, gfit['q2'].max() * 1.08, 200)
        ln_I_fit = tr['slope'] * q2_fit + tr['intercept']

        ax.errorbar(
            grouped['q2'][in_fit], ln_I_all[in_fit],
            yerr=ln_I_err[in_fit],
            fmt='o', color=_TOTAL_COLOR,
            capsize=3, capthick=0.6, markersize=4, linewidth=0,
            alpha=0.5, zorder=2
        )
        ax.plot(q2_fit, ln_I_fit, '--', color=_TOTAL_COLOR,
                linewidth=1.2, alpha=0.7, zorder=2)
        ax.scatter([0], [tr['intercept']], color=_TOTAL_COLOR,
                   marker='*', s=100, zorder=4, edgecolors='white', linewidth=0.5)

        Rg_str = f'{tr["Rg [nm]"]:.1f} nm' if not np.isnan(tr['Rg [nm]']) else 'N/A'
        I0_str = f'{tr["I0 [kHz]"]:.4f} kHz'
        ax.plot([], [], color=_TOTAL_COLOR, linewidth=1.5, linestyle='--',
                label=(f'Total   $R_g$ = {Rg_str}   '
                       f'$I_0$ = {I0_str}   $R^2$ = {tr["R2"]:.4f}'))

    # per-population curves
    for idx, (pop, res) in enumerate(guinier_results.items()):
        _plot_guinier_population(ax, idx, pop, res, colors)

    # shade q²<0 extrapolation region
    xlim = ax.get_xlim()
    ax.axvspan(min(xlim[0], -xlim[1] * 0.02), 0,
               alpha=0.04, color='gray', zorder=0)
    ax.axvline(0, color='#888888', linestyle=':', linewidth=0.9, zorder=1)

    ax.set_xlabel('$q^2$ [nm$^{-2}$]', fontsize=10)
    ax.set_ylabel(r'$\ln\, I$ [kHz]', fontsize=10)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax.legend(fontsize=8.5, framealpha=0.9, edgecolor='#cccccc')
    _style_ax(ax)
    plt.tight_layout()
    plt.show()


#summary

def summarize_sls(sls_data, guinier_results, n_populations):
    """
    Summary DataFrame: I₀, Rg, qRg_max and Guinier R² per population.
    """
    summary = []
    for i in range(1, n_populations + 1):
        row = {'Population': i}
        if i in guinier_results:
            res = guinier_results[i]
            row['I0 [kHz]']   = res['I0 [kHz]']
            row['Rg [nm]']    = res['Rg [nm]']
            row['qRg_max']    = res['qRg_max']
            row['Guinier R2'] = res['R2']
        else:
            row['I0 [kHz]']   = np.nan
            row['Rg [nm]']    = np.nan
            row['qRg_max']    = np.nan
            row['Guinier R2'] = np.nan
        summary.append(row)

    return pd.DataFrame(summary)


def summarize_sls_combined(sls_data, guinier_results, total_result,
                            n_populations, rh_values, exponent=6):
    """
    Combined summary table merging intensity-weighted (Guinier) and
    number-weighted (Steinschulte) results per population, plus a total row.

    Columns
    -------
    Population      : population index
    Rh [nm]         : DLS-derived hydrodynamic radius
    I-fraction [%]  : mean intensity fraction from area-percent decomposition
    N-fraction [%]  : number/concentration fraction after Rh^exponent correction
    I0 (intensity)  : I₀ extrapolated from per-population Guinier fit
    I0 (number)     : I₀_total * N-fraction  (no fit — analytically derived)
    Rg [nm]         : radius of gyration from Guinier fit
    qRg_max         : validity indicator (should be < 1.3)
    R_squared       : Guinier fit quality
    """
    I0_total = total_result['I0 [kHz]'] if total_result is not None else np.nan

    # compute per-file number fractions from _nw columns if available,
    # otherwise recompute from I_pop and rh_values
    nw_available = all(
        f'I_pop{i}_nw [kHz]' in sls_data.columns
        for i in range(1, n_populations + 1)
    )

    rows = []
    for i in range(1, n_populations + 1):
        row = {'Population': i}

        # Rh from DLS
        row['Rh [nm]'] = rh_values.get(i, np.nan)

        # intensity fraction from Method A area columns
        i_col = f'I_pop{i} [kHz]'
        if i_col in sls_data.columns and 'MeanCR_corr [kHz]' in sls_data.columns:
            valid = sls_data['MeanCR_corr [kHz]'] > 0
            row['I-fraction [%]'] = (
                sls_data.loc[valid, i_col] /
                sls_data.loc[valid, 'MeanCR_corr [kHz]']
            ).mean() * 100
        else:
            row['I-fraction [%]'] = np.nan

        # number fraction from _nw columns
        nw_col = f'I_pop{i}_nw [kHz]'
        if nw_available and 'MeanCR_corr [kHz]' in sls_data.columns:
            valid = sls_data['MeanCR_corr [kHz]'] > 0
            row['N-fraction [%]'] = (
                sls_data.loc[valid, nw_col] /
                sls_data.loc[valid, 'MeanCR_corr [kHz]']
            ).mean() * 100
        else:
            # recompute analytically if _nw columns missing
            rh = rh_values.get(i, np.nan)
            if not np.isnan(rh) and rh > 0:
                c_vals = {
                    j: (sls_data[f'I_pop{j} [kHz]'] / (rh_values[j] ** exponent)).mean()
                    for j in range(1, n_populations + 1)
                    if f'I_pop{j} [kHz]' in sls_data.columns and rh_values.get(j, 0) > 0
                }
                c_total = sum(c_vals.values())
                row['N-fraction [%]'] = c_vals.get(i, np.nan) / c_total * 100 \
                    if c_total > 0 else np.nan
            else:
                row['N-fraction [%]'] = np.nan

        # Guinier results (intensity-weighted)
        if i in guinier_results:
            res = guinier_results[i]
            row['I0 (intensity) [kHz]'] = res['I0 [kHz]']
            row['I0 (number) [kHz]']    = I0_total * row['N-fraction [%]'] / 100
            row['Rg [nm]']              = res['Rg [nm]']
            row['qRg_max']              = res['qRg_max']
            row['R_squared']            = res['R2']
        else:
            row['I0 (intensity) [kHz]'] = np.nan
            row['I0 (number) [kHz]']    = np.nan
            row['Rg [nm]']              = np.nan
            row['qRg_max']              = np.nan
            row['R_squared']            = np.nan

        rows.append(row)

    # total row
    total_row = {
        'Population'            : 'Total',
        'Rh [nm]'               : np.nan,
        'I-fraction [%]'        : 100.0,
        'N-fraction [%]'        : 100.0,
        'I0 (intensity) [kHz]'  : I0_total,
        'I0 (number) [kHz]'     : I0_total,
        'Rg [nm]'               : total_result['Rg [nm]'] if total_result else np.nan,
        'qRg_max'               : total_result['qRg_max'] if total_result else np.nan,
        'R_squared'             : total_result['R2']      if total_result else np.nan,
    }
    rows.append(total_row)
    
    df = pd.DataFrame(rows)
    return df.style.format({
        'Rh [nm]'               : '{:.1f}',
        'I-fraction [%]'        : '{:.2f}',
        'N-fraction [%]'        : '{:.4f}',
        'I0 (intensity) [kHz]'  : '{:.5f}',
        'I0 (number) [kHz]'     : '{:.5f}',
        'Rg [nm]'               : '{:.1f}',
        'qRg_max'               : '{:.3f}',
        'R_squared'             : '{:.4f}',
        }, na_rep='—')

    return pd.DataFrame(rows)
