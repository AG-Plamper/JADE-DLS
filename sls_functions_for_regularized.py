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

compute_guinier_total(sls_data, q2_range)
    Guinier fit on the raw total intensity I_total vs q².
    Reference fit independent of any population decomposition.

compute_guinier_extrapolation(sls_data, n_populations, q2_range)
    Guinier analysis per population: ln(I_pop) vs q² linear fit on the
    intensity-weighted I_pop data. Yields I₀ and Rg per population.

compute_number_weighted_I0(guinier_results, rh_values, exponent)
    number-fraction correction applied only to the extrapolated I0:
    c_i = I0_i / Rh_i^exponent → normalised → I0_i_nw = I0_total * c_frac_i
    Guinier's law is only valid for intensity-weighted data, so the
    Rh^exponent correction is applied after extrapolation, never to the
    raw per-angle intensities feeding the fit.

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


def compute_guinier_extrapolation(sls_data, n_populations, q2_range=None):
    """
    Guinier analysis per population: ln(I_pop) vs q² linear fit.

        ln I(q) = ln I₀  −  (Rg²/3) · q²

    Always fit on the intensity-weighted I_popN [kHz] columns (Method A).
    Guinier's law describes real scattered intensity, so the Rh^exponent
    number-weighting correction must never be applied before this fit —
    see compute_number_weighted_I0 for the post-extrapolation correction.

    Parameters
    ----------
    sls_data : pd.DataFrame
    n_populations : int
    q2_range : tuple, dict, or None
        - None             : use all angles for all populations
        - (q2_min, q2_max) : same range applied to all populations
        - {1: (a, b), 2: (c, d), ...} : per-population ranges

    Returns
    -------
    dict  guinier_results[i] = {slope, intercept, R2, I0, Rg, qRg_max,
                                 grouped, grouped_fit, ln_I}
    """
    guinier_results = {}

    for i in range(1, n_populations + 1):
        i_col = f'I_pop{i} [kHz]'
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


def compute_number_weighted_I0(guinier_results, rh_values, exponent=6):
    """
    number/concentration-fraction correction applied to the extrapolated I0.

    Guinier's law describes real scattered intensity, so the Rh^exponent
    correction must be applied to the already-extrapolated, intensity-weighted
    I0 values — never to the raw per-angle intensities that feed the Guinier
    fit itself. Doing it beforehand would corrupt the q-dependence the fit
    relies on (the renormalization mixes in the q-dependence of every other
    population's area fraction, so the result no longer obeys Guinier's law).

        c_i      = I0_i  /  Rh_i ^ exponent
        c_frac_i = c_i   /  sum_j(c_j)
        I0_i_nw  = I0_total * c_frac_i

    This follows: c ~ I / R^exponent
        exponent = 5  Daoud-Cotton scaling for star polymers (M ~ R^5)
        exponent = 6  Rayleigh scattering for compact spheres (I ~ R^6)

    Parameters
    ----------
    guinier_results : dict
        Output of compute_guinier_extrapolation — guinier_results[i]['I0 [kHz]'].
    rh_values : dict {population: Rh [nm]}
    exponent : float

    Returns
    -------
    dict  nw_results[i] = {'I0_nw [kHz]': ..., 'N-fraction [%]': ...}
    """
    c_vals = {}
    for i, res in guinier_results.items():
        rh = rh_values.get(i, np.nan)
        if np.isnan(rh) or rh <= 0:
            print(f"Warning: invalid Rh for population {i} — skipped.")
            continue
        c_vals[i] = res['I0 [kHz]'] / (rh ** exponent)

    c_total  = sum(c_vals.values())
    I0_total = sum(res['I0 [kHz]'] for res in guinier_results.values())

    nw_results = {}
    print(f"\nnumber-fraction correction from extrapolated I0 (exponent={exponent})")
    print(f"{'Population':<12} {'Rh [nm]':<12} {'I0 (intensity)':<18} {'N-fraction [%]':<18}")
    for i in guinier_results:
        if i not in c_vals or c_total <= 0:
            nw_results[i] = {'I0_nw [kHz]': np.nan, 'N-fraction [%]': np.nan}
            continue
        frac = c_vals[i] / c_total
        nw_results[i] = {
            'I0_nw [kHz]'    : I0_total * frac,
            'N-fraction [%]' : frac * 100,
        }
        print(f"  Pop {i:<8} {rh_values.get(i, np.nan):<12.1f} "
              f"{guinier_results[i]['I0 [kHz]']:<18.5f} {frac * 100:<18.4f}")

    return nw_results


def summarize_sls_combined(sls_data, guinier_results, total_result,
                            n_populations, rh_values, exponent=6):
    """
    Combined summary table merging intensity-weighted (Guinier) and
    number-weighted results per population, plus a total row.

    The Rh^exponent number-weighting is applied only to the extrapolated I0
    values (see compute_number_weighted_I0), never to the raw per-angle
    intensities — Guinier's law is only valid for intensity-weighted data.

    Columns
    -------
    Population      : population index
    Rh [nm]         : DLS-derived hydrodynamic radius
    I-fraction [%]  : mean intensity fraction from area-percent decomposition (diagnostic)
    I0 (intensity)  : I₀ extrapolated from per-population Guinier fit
    N-fraction [%]  : number/concentration fraction after Rh^exponent correction of I0
    I0 (number)     : I₀_total(intensity) * N-fraction  (no fit — analytically derived)
    Rg [nm]         : radius of gyration from Guinier fit
    qRg_max         : validity indicator (should be < 1.3)
    R_squared       : Guinier fit quality
    """
    I0_total   = total_result['I0 [kHz]'] if total_result is not None else np.nan
    nw_results = compute_number_weighted_I0(guinier_results, rh_values, exponent=exponent)

    rows = []
    for i in range(1, n_populations + 1):
        row = {'Population': i, 'Rh [nm]': rh_values.get(i, np.nan)}

        # intensity fraction from Method A area columns (diagnostic only)
        i_col = f'I_pop{i} [kHz]'
        if i_col in sls_data.columns and 'MeanCR_corr [kHz]' in sls_data.columns:
            valid = sls_data['MeanCR_corr [kHz]'] > 0
            row['I-fraction [%]'] = (
                sls_data.loc[valid, i_col] /
                sls_data.loc[valid, 'MeanCR_corr [kHz]']
            ).mean() * 100
        else:
            row['I-fraction [%]'] = np.nan

        if i in guinier_results:
            res = guinier_results[i]
            nw  = nw_results.get(i, {})
            row['I0 (intensity) [kHz]'] = res['I0 [kHz]']
            row['N-fraction [%]']       = nw.get('N-fraction [%]', np.nan)
            row['I0 (number) [kHz]']    = nw.get('I0_nw [kHz]', np.nan)
            row['Rg [nm]']              = res['Rg [nm]']
            row['qRg_max']              = res['qRg_max']
            row['R_squared']            = res['R2']
        else:
            row['I0 (intensity) [kHz]'] = np.nan
            row['N-fraction [%]']       = np.nan
            row['I0 (number) [kHz]']    = np.nan
            row['Rg [nm]']              = np.nan
            row['qRg_max']              = np.nan
            row['R_squared']            = np.nan

        rows.append(row)

    # total row
    rows.append({
        'Population'            : 'Total',
        'Rh [nm]'               : np.nan,
        'I-fraction [%]'        : 100.0,
        'I0 (intensity) [kHz]'  : I0_total,
        'N-fraction [%]'        : 100.0,
        'I0 (number) [kHz]'     : I0_total,
        'Rg [nm]'               : total_result['Rg [nm]'] if total_result else np.nan,
        'qRg_max'               : total_result['qRg_max'] if total_result else np.nan,
        'R_squared'             : total_result['R2']      if total_result else np.nan,
    })

    df = pd.DataFrame(rows)
    return df.style.format({
        'Rh [nm]'               : '{:.1f}',
        'I-fraction [%]'        : '{:.2f}',
        'I0 (intensity) [kHz]'  : '{:.5f}',
        'N-fraction [%]'        : '{:.4f}',
        'I0 (number) [kHz]'     : '{:.5f}',
        'Rg [nm]'               : '{:.1f}',
        'qRg_max'               : '{:.3f}',
        'R_squared'             : '{:.4f}',
        }, na_rep='—')
