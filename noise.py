# -*- coding: utf-8 -*-
"""
noise.py
========
Baseline and noise correction for DLS autocorrelation functions.
Removes the long-time baseline offset and suppresses high-lag noise
before cumulant or inverse Laplace fitting.

Functions
---------
apply_noise_corrections(processed_correlations, ...)
    Apply baseline subtraction, tail truncation and optional smoothing
    to a dict of correlation DataFrames; returns corrected dict.

plot_correction_sample(original, corrected, df_basedata, ...)
    Overlay plot of original vs corrected correlation functions for a
    representative subset of files to verify correction quality.

Dependencies: numpy, pandas, scipy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import random

#apply baseline and/or intercept correction to correlation data.
def apply_noise_corrections(processed_correlations,
                             col='g(2)-1',
                             baseline_correction=True, baseline_pct=5,
                             intercept_correction=True, intercept_pct=2):
    corrected = {}

    for filename, df in processed_correlations.items():
        df_corr = df.copy()
        n = len(df_corr)

        n_baseline  = max(1, int(np.floor(n * baseline_pct  / 100)))
        n_intercept = max(1, int(np.floor(n * intercept_pct / 100)))

        # Baseline correction first so intercept correction works on
        # already-shifted data
        if baseline_correction:
            baseline_mean = df_corr[col].iloc[-n_baseline:].mean()
            df_corr.loc[:, col] = df_corr[col] - baseline_mean

        # Intercept correction
        if intercept_correction:
            intercept_mean = df_corr[col].iloc[:n_intercept].mean()
            df_corr.loc[df_corr.index[:n_intercept], col] = intercept_mean

        corrected[filename] = df_corr

    return corrected


def plot_correction_sample(original, corrected, df_basedata,
                            col='g(2)-1', time_col='t [s]',
                            figsize=(14, 5), title=''):
    """
    Two-panel overview of noise correction effect across all files.
    Left panel: all original curves overlaid, coloured by angle.
    Right panel: all corrected curves overlaid, coloured by angle.
    """
    #build angle lookup from basedata
    angle_lookup = dict(zip(df_basedata['filename'], df_basedata['angle [°]']))
    angles       = sorted(set(angle_lookup.values()))
    cmap         = plt.cm.viridis
    norm         = plt.Normalize(vmin=min(angles), vmax=max(angles))

    fig, (ax_orig, ax_corr) = plt.subplots(1, 2, figsize=figsize,
                                            sharey=True, sharex=True)

    for filename in original.keys():
        df_orig = original[filename]
        df_corr = corrected[filename]
        angle   = angle_lookup.get(filename, None)
        color   = cmap(norm(angle)) if angle is not None else 'grey'

        ax_orig.plot(df_orig[time_col], df_orig[col],
                     linewidth=1, color=color)
        ax_corr.plot(df_corr[time_col], df_corr[col],
                     linewidth=1, color=color)

    for ax, panel_title in zip([ax_orig, ax_corr], ['original', 'corrected']):
        ax.set_xscale('log')
        ax.set_xlabel('t [s]')
        ax.set_ylabel('g(2)-1')
        ax.set_title(panel_title, fontsize=10)
        ax.grid(True)
        ax.tick_params(axis='both', labelsize=8)
        ax.xaxis.set_major_locator(plt.LogLocator(base=10, numticks=4))
        ax.yaxis.get_major_locator().set_params(nbins=5)

    #colourbar for angle
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax_orig, ax_corr], shrink=0.8, pad=0.02)
    cbar.set_label('angle [°]', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    if title:
        fig.suptitle(title, fontsize=11)

    plt.tight_layout()
    plt.show()
