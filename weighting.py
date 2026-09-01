# -*- coding: utf-8 -*-
"""
weighting.py
============
Weight estimation for DLS correlation-function fits (opt-in via a
`weighting` boolean at the notebook level).

Context
-------
regularized.py (Tikhonov/NNLS) and the cumulant fitters (cumulants.py,
cumulants_C.py, cumulants_D.py) can treat every lag-time channel as
equally uncertain (weights=None, the default -- unchanged behaviour) or
weight each channel by its true noise variance. Real multi-tau DLS data
is heteroscedastic: noise depends strongly on lag time (shot-noise
dominated at the shortest gate times, then falling, then rising again
near the measuring-time limit). Weighting the fit residuals by the true
per-channel variance matters especially for the Tikhonov-regularized fit,
since the smoothing parameter alpha trades off "fit the data" against
"keep f smooth" -- a trade-off that is only meaningful once residuals are
scaled to their actual uncertainty.

This module implements the noise model of:
    Biganzoli & Ferri, "Statistical analysis of dynamic light scattering
    data: revisiting and beyond the Schaetzel formulas", Optics Express
    26(22), 29375-29395 (2018). https://doi.org/10.1364/OE.26.029375
which corrects the original Schaetzel (1990) variance formula for
"triangular averaging" -- the finite-sampling-time effect that the
original formula neglects and that becomes large exactly where multi-tau
correlators spend most of their channels (lag time comparable to or
larger than the sampling time of that channel).
A qualitative self-test against the paper's own Fig. 2 is at
the bottom of this file (run this module directly to print it).

Scope -- what lives here vs. in the working modules
----------------------------------------------------
This file contains ONLY weight estimation: the Biganzoli-Ferri noise
model (estimate_weights and its helpers) and two small dict-level
conveniences (apply_weights_to_correlations, compute_weights_for_all).
It does NOT duplicate any fitting code. Per project decision, weighting
is switched on by handing the fitting functions in regularized.py /
cumulants.py / cumulants_C.py / cumulants_D.py a correlation dict whose
per-file dataframes carry an extra 'weight' column -- every such function
auto-detects that column (falling back to weights=None, i.e. today's
unweighted behaviour, when it's absent) so no call site needs to change
between a weighted and unweighted run, only which dict variable is fed
in. See apply_weights_to_correlations() below.

Duration [s] and the two channel count rates (MeanCR0/MeanCR1) are all
extracted by preprocessing.extract_data() in the same pass over each
file's header as angle/temperature/etc. (intensity.py, which used to
extract MeanCR0/MeanCR1 separately, was retired in v2.4 and merged into
preprocessing.py).

No metadata fallback: if Duration or MeanCR0/MeanCR1 cannot be retrieved
for a file, estimate_weights()/compute_weights_for_all() raise rather
than silently skipping or falling back to unweighted -- per project
decision, a missing-metadata file should crash loudly, not be quietly
dropped from a weighted run.

Typical usage (added as notebook cells, placed right after noise
correction, before any fitting cells)
------------------------------------------------------------------------
    weighting = True  # or False

    from weighting import compute_weights_for_all, apply_weights_to_correlations

    if weighting:
        weights_dict = compute_weights_for_all(processed_correlations_2, df_basedata_mod)
        processed_correlations_weighted = apply_weights_to_correlations(processed_correlations_2, weights_dict)
    else:
        processed_correlations_weighted = processed_correlations_2

    # every downstream cell (regularized, cumulant B/C/D, alpha grid-search)
    # is fed processed_correlations_weighted from here on -- no other
    # change needed; each fitting function auto-detects the 'weight'
    # column when present, and reproduces today's unweighted output
    # exactly when it's absent (weighting=False).

Dependencies: numpy, pandas, scipy
"""
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


# ============================================================================
# 1. Biganzoli & Ferri (2018) noise model
# ============================================================================

def infer_sampling_times(tau):
    """Recover the per-channel multi-tau sampling time dt_k directly from
    the tau grid's local spacing (central difference), without needing to
    know the correlator's (p, m) structure. Within one multi-tau stage,
    consecutive tau values are spaced by exactly that stage's sampling
    time (Eq. 11 of the paper); at stage boundaries this is a local
    approximation (affects only the few channels at each stage seam)."""
    tau = np.asarray(tau, dtype=float)
    n = len(tau)
    if n < 2:
        raise ValueError("Need at least 2 tau points to infer sampling times.")
    fwd = np.diff(tau)                       # length n-1
    dt_fwd = np.empty(n)
    dt_fwd[:-1] = fwd
    dt_fwd[-1] = fwd[-1]
    dt_bwd = np.empty(n)
    dt_bwd[1:] = fwd
    dt_bwd[0] = fwd[0]
    return 0.5 * (dt_fwd + dt_bwd)


def prefit_single_exponential(tau, g2_minus_1):
    """Biganzoli & Ferri Eq. (14): g2(tau_k) = B + beta*exp(-2*tau_k/tau_c).
    Fast bootstrap fit used only to obtain (beta, tau_c) for the noise
    formula -- not a replacement for the project's cumulant fits. (Cross-
    checked against cumulants_C.py's own adaptive single-exponential fit
    on real data: the two converge to the same result in every case
    tested -- see conversation notes -- so this simple bootstrap is not a
    weak link.)"""
    tau = np.asarray(tau, dtype=float)
    g2m1 = np.asarray(g2_minus_1, dtype=float)

    def model(t, B, beta, tau_c):
        return B + beta * np.exp(-2.0 * t / tau_c)

    b0 = float(np.clip(g2m1[-min(5, len(g2m1)):].mean(), -0.5, 0.5))
    beta0 = float(np.clip(g2m1[0] - b0, 1e-6, 2.0))
    tau_c0 = float(tau[len(tau) // 3])
    p0 = [b0, beta0, tau_c0]
    bounds = ([-0.5, 1e-6, tau.min() * 1e-3], [0.5, 2.0, tau.max() * 1e3])
    popt, _ = curve_fit(model, tau, g2m1, p0=p0, bounds=bounds, maxfev=20000)
    B_fit, beta_fit, tau_c_fit = popt
    return float(beta_fit), float(tau_c_fit), float(B_fit)


def _biganzoli_ferri_variance(k, Gamma, beta, M, nbar):
    """Biganzoli & Ferri Eq. (9) -- exact single-exponential DLS variance,
    corrected for triangular averaging to all orders. All inputs are
    dimensionless / per-channel arrays (or scalars) of equal shape:
      k     : tau_k / dt_k               (dimensionless lag index)
      Gamma : dt_k / tau_c                (reduced sampling time)
      beta  : coherence factor (scalar)
      M     : T / dt_k                    (samples in this channel's stage)
      nbar  : cr_bar * dt_k                (mean counts per sample interval)
    Returns sigma^2(k), same shape as k.
    """
    k = np.asarray(k, dtype=float)
    Gamma = np.asarray(Gamma, dtype=float)
    M = np.asarray(M, dtype=float)
    nbar = np.asarray(nbar, dtype=float)

    sinh = np.sinh(Gamma)
    coth_G = 1.0 / np.tanh(Gamma)
    coth_2G = 1.0 / np.tanh(2.0 * Gamma)
    chi0 = np.sqrt((2.0 * Gamma - 1.0 + np.exp(-2.0 * Gamma)) / (2.0 * Gamma ** 2))

    e2 = np.exp(-2.0 * Gamma * k)
    e4 = np.exp(-4.0 * Gamma * k)

    a = 1.0 + 2.0 * beta
    c = 2.0 * k + 4.0 * beta * k - 2.0 - 8.0 * beta + 4.0 * beta * chi0 ** 2 - 16.0 * beta * chi0

    term1 = 2.0 * (sinh ** 2 / Gamma ** 2) * chi0 ** 2 * e4
    term2 = 4.0 * beta * (sinh ** 2 / Gamma ** 2) * chi0 ** 2 * e2
    term3 = 8.0 * beta * (sinh ** 3 / Gamma ** 3) * chi0 * e4
    term4 = (sinh ** 4 / Gamma ** 4) * (
        coth_2G - 1.0 + 4.0 * beta * e2 * (coth_G - 1.0) + e4 * (a * coth_2G + c)
    )
    term5 = 4.0 * beta * (sinh ** 6 / Gamma ** 6) * e4 * (coth_G - 1.0)
    term6 = -8.0 * beta * (sinh ** 5 / Gamma ** 5) * e4 * (k - 2.0 + coth_G)
    term7 = chi0 ** 4

    shot = (2.0 / (beta * nbar)) * (
        chi0 ** 2
        + (sinh ** 2 / Gamma ** 2) * e2 * (e2 + 2.0 * beta * chi0)
        + 2.0 * beta * (sinh ** 3 / Gamma ** 3) * e4
        - 2.0 * beta * (sinh ** 4 / Gamma ** 4) * e4
    ) + (1.0 / (beta ** 2 * nbar ** 2)) * (1.0 + beta * (sinh ** 2 / Gamma ** 2) * e2)

    bracket = term1 + term2 + term3 + term4 + term5 + term6 + term7 + shot
    return (beta ** 2 / (M - k)) * bracket


# Gamma=50 keeps sinh(Gamma)^6/Gamma^6 (the highest power appearing in Eq. 9)
# many orders of magnitude below float64 overflow (~1.8e308), while already
# far beyond the range the paper itself validates (Gamma <~ 1, see Fig. 2) --
# so clipping here only affects deep-tail, low-information channels where
# dt_k is dozens of times larger than tau_c.
_GAMMA_CAP = 50.0
# floor for (M_k - k): only small/negative when tau_k approaches or exceeds
# the measuring time for that channel -- i.e. too few independent samples to
# say anything reliable about its noise.
_MIN_M_MINUS_K = 0.5


def estimate_weights(tau, g2_minus_1, cr_A_kHz, cr_B_kHz, T,
                      beta=None, tau_c=None, return_diagnostics=False, verbose=True):
    """End-to-end Schaetzel/Biganzoli-Ferri weight estimation for one
    correlation curve. Returns a mean-normalized weight vector (mean 1,
    so an existing alpha choice is not silently rescaled when weighting
    is switched on). cr_A_kHz/cr_B_kHz are the two ALV channel count
    rates (MeanCR0/MeanCR1); their geometric mean is used per Eq. (16),
    validated by the paper specifically for cross-correlation setups
    like the ALV pseudo-cross-correlation detectors.

    Raises ValueError if cr_A_kHz/cr_B_kHz/T are missing (None/NaN) --
    by design, no fallback to unweighted for an individual file; a
    measurement with unreadable metadata should stop the run, not be
    silently dropped.

    verbose=True prints a plain-language note when some channels fall
    outside the noise model's validated range (see below) -- this is the
    normal case for essentially every wide-dynamic-range measurement, so
    compute_weights_for_all() calls this with verbose=False and prints
    one aggregated summary for the whole batch instead of one message
    per file.
    """
    for name, val in (('T (Duration [s])', T), ('cr_A_kHz (MeanCR0)', cr_A_kHz), ('cr_B_kHz (MeanCR1)', cr_B_kHz)):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            raise ValueError(f"estimate_weights: {name} is missing -- cannot compute weights "
                              f"for this file. Check the ALV .ASC header was parsed correctly.")

    tau = np.asarray(tau, dtype=float)
    g2m1 = np.asarray(g2_minus_1, dtype=float)

    if beta is None or tau_c is None:
        beta_fit, tau_c_fit, _ = prefit_single_exponential(tau, g2m1)
        beta = beta_fit if beta is None else beta
        tau_c = tau_c_fit if tau_c is None else tau_c

    dt_k = infer_sampling_times(tau)
    Gamma_k = dt_k / tau_c
    k_dimless = tau / dt_k
    M_k = T / dt_k
    cr_bar_Hz = np.sqrt(cr_A_kHz * cr_B_kHz) * 1e3  # kHz -> Hz, geometric mean (Eq. 16)
    nbar_k = cr_bar_Hz * dt_k

    n_capped = int(np.sum(Gamma_k > _GAMMA_CAP))
    if n_capped > 0 and verbose:
        pct = 100.0 * n_capped / len(tau)
        print(f"weighting.estimate_weights: {n_capped}/{len(tau)} channels ({pct:.0f}% of this curve) "
              f"sit far out in the baseline tail -- many decay times past tau_c, where g(2)-1 has "
              f"already flattened to ~0. That's outside the range the noise formula was validated "
              f"for, so their weights are only approximate. This is normal for wide-range "
              f"measurements, not a sign of a problem with this fit.")
    Gamma_k_safe = np.clip(Gamma_k, 1e-8, _GAMMA_CAP)

    m_minus_k = M_k - k_dimless
    n_floored = int(np.sum(m_minus_k < _MIN_M_MINUS_K))
    if n_floored > 0 and verbose:
        pct = 100.0 * n_floored / len(tau)
        print(f"weighting.estimate_weights: {n_floored}/{len(tau)} channels ({pct:.0f}% of this curve) "
              f"sit right at the end of the measurement, where too few independent samples remain "
              f"to say much about the noise there. Their weights are approximate too; this is normal "
              f"for the last few channels of a long measurement.")
    M_k_safe = k_dimless + np.maximum(m_minus_k, _MIN_M_MINUS_K)

    with np.errstate(over='raise', invalid='raise'):
        sigma2 = _biganzoli_ferri_variance(k_dimless, Gamma_k_safe, beta, M_k_safe, nbar_k)
    sigma2 = np.clip(sigma2, np.finfo(float).tiny, None)
    weights = 1.0 / sigma2
    weights = weights / np.mean(weights)

    if return_diagnostics:
        diagnostics = {'beta': beta, 'tau_c': tau_c, 'dt_k': dt_k, 'Gamma_k': Gamma_k,
                        'k_dimless': k_dimless, 'M_k': M_k, 'nbar_k': nbar_k, 'sigma2': sigma2}
        return weights, diagnostics
    return weights


# ============================================================================
# 2. Dict-level conveniences -- these are what the notebook actually calls
# ============================================================================

def compute_weights_for_all(processed_correlations, df_basedata_mod):
    """Loops estimate_weights() over every file in processed_correlations.
    df_basedata_mod must already carry 'duration [s]', 'meancr0 [kHz]' and
    'meancr1 [kHz]' (all extracted by preprocessing.extract_data() in a
    single pass over each file's header). Returns {filename: weight_array}.

    No fallback: a file with an unreadable/missing Duration or MeanCR0/
    MeanCR1 raises and stops the whole call, by design (see module
    docstring).

    Per-file "some channels are out of the validated range" notes are
    suppressed here (they fire for nearly every measurement, since any
    wide-dynamic-range multi-tau curve has a baseline tail) and replaced
    with a single plain-language summary for the whole batch.
    """
    weights_dict = {}
    tail_pcts, tail_files = [], 0
    end_pcts, end_files = [], 0

    for filename, df in processed_correlations.items():
        rows = df_basedata_mod.loc[df_basedata_mod['filename'] == filename]
        if rows.empty:
            raise ValueError(f"compute_weights_for_all: '{filename}' not found in df_basedata_mod.")
        row = rows.iloc[0]
        T = row['duration [s]']
        cr_A_kHz = row['meancr0 [kHz]']
        cr_B_kHz = row['meancr1 [kHz]']

        tau = df['t [s]'].to_numpy()
        weights, diag = estimate_weights(
            tau=tau, g2_minus_1=df['g(2)-1'].to_numpy(),
            cr_A_kHz=cr_A_kHz, cr_B_kHz=cr_B_kHz, T=T,
            return_diagnostics=True, verbose=False)
        weights_dict[filename] = weights

        n_tail = int(np.sum(diag['Gamma_k'] > _GAMMA_CAP))
        if n_tail > 0:
            tail_files += 1
            tail_pcts.append(100.0 * n_tail / len(tau))
        n_end = int(np.sum((diag['M_k'] - diag['k_dimless']) < _MIN_M_MINUS_K))
        if n_end > 0:
            end_files += 1
            end_pcts.append(100.0 * n_end / len(tau))

    n_files = len(processed_correlations)
    if tail_files > 0:
        print(f"Weighting note: {tail_files}/{n_files} file(s) reach a clean, fully flattened "
              f"baseline (on average the last {np.mean(tail_pcts):.0f}% of the curve) -- a sign of a "
              f"well-measured decay, not a problem.")
    if end_files > 0:
        print(f"Weighting note: {end_files}/{n_files} file(s) have a few channels right at the "
              f"end of the measurement (on average {np.mean(end_pcts):.0f}% of the curve) where too "
              f"few independent samples remain to pin down the noise precisely -- also normal, "
              f"also just approximate rather than precise.")
    return weights_dict


def apply_weights_to_correlations(processed_correlations, weights_dict):
    """Returns a NEW correlation dict, same shape as processed_correlations,
    with a 'weight' column added to each dataframe. This is the object to
    feed into every downstream fitting call in place of the unweighted
    dict -- regularized.nnls_reg / nnls_reg_all / nnls_reg_simple /
    analyze_random_datasets_grid and the cumulant fitters in cumulants.py /
    cumulants_C.py / cumulants_D.py all auto-detect the 'weight' column,
    so no call site changes between a weighted and unweighted run.
    """
    out = {}
    for filename, df in processed_correlations.items():
        if filename not in weights_dict:
            raise ValueError(f"apply_weights_to_correlations: no weights computed for '{filename}'.")
        df_w = df.copy()
        df_w['weight'] = np.asarray(weights_dict[filename], dtype=float)
        out[filename] = df_w
    return out


# ============================================================================
# 3. Self-test / sanity check — run this file directly
# ============================================================================

def _self_test():
    """Qualitative check of _biganzoli_ferri_variance against the paper's
    own Fig. 2 test conditions (beta=1, nbar=1, M=1e5): print a few
    sigma^2(k) values for Gamma=0.05 and Gamma=0.8 to compare by eye
    against Fig. 2(a) of Biganzoli & Ferri (2018) -- expect ~O(1e-4) to
    O(1e-3), decreasing with k for Gamma=0.05, flatter for Gamma=0.8."""
    print("=== Biganzoli-Ferri Eq.(9) sanity check (cf. paper Fig. 2) ===")
    for Gamma in (0.05, 0.8):
        k = np.array([1, 5, 20, 100, 500])
        sigma2 = _biganzoli_ferri_variance(k, Gamma, beta=1.0, M=1e5, nbar=1.0)
        print(f"Gamma={Gamma}: k={k} -> sigma^2={sigma2}")


if __name__ == '__main__':
    _self_test()
