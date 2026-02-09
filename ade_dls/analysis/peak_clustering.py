"""
Peak Clustering for DLS Analysis
Automatically groups similar peaks across different measurement angles
"""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Dict, Optional
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def cluster_peaks_across_datasets(data_df: pd.DataFrame,
                                   tau_prefix: str = 'tau_',
                                   method: str = 'dbscan',
                                   eps_factor: float = 0.3,
                                   min_samples: int = 2) -> Tuple[pd.DataFrame, Dict]:
    """
    Cluster tau values across all datasets to identify common modes

    IMPORTANT: This function clusters based on diffusion coefficients (D),
    not raw tau values. This is physically correct because:
    - τ = 1/(D × q²), so τ depends on scattering angle (q²)
    - D should be constant for a given particle species across all angles
    - Clustering on D groups peaks with the same physical origin

    Args:
        data_df: DataFrame with tau columns (tau_1, tau_2, etc.) AND q^2 column
        tau_prefix: Prefix for tau columns
        method: Clustering method ('dbscan' or 'log_proximity')
        eps_factor: DBSCAN epsilon as fraction of log-space range
        min_samples: Minimum samples per cluster for DBSCAN

    Returns:
        Tuple of (reassigned DataFrame, clustering info dict)
    """
    # Find all tau columns
    tau_cols = [col for col in data_df.columns if col.startswith(tau_prefix)]

    if not tau_cols:
        raise ValueError(f"No columns starting with '{tau_prefix}' found")

    # Check if q^2 column exists (required for physics-based clustering)
    if 'q^2' not in data_df.columns:
        raise ValueError("DataFrame must contain 'q^2' column for physics-based clustering")

    print(f"\n[Peak Clustering] Found {len(tau_cols)} tau columns")
    print(f"[Peak Clustering] Method: {method} (physics-based: clusters by D = Γ/q²)")

    # Collect all tau values with their source information
    all_tau_values = []
    all_D_values = []  # Diffusion coefficients
    tau_metadata = []

    for idx, row in data_df.iterrows():
        q_squared = row.get('q^2', None)

        if q_squared is None or q_squared <= 0:
            print(f"[Peak Clustering] Warning: Invalid q^2 value for row {idx}, skipping")
            continue

        for tau_col in tau_cols:
            tau_val = row[tau_col]
            if pd.notna(tau_val) and tau_val > 0:
                # Calculate diffusion coefficient: D = Γ/q² = 1/(τ × q²)
                # This is the physically meaningful quantity that should be constant across angles
                gamma = 1.0 / tau_val  # Relaxation rate
                D = gamma / q_squared   # Diffusion coefficient (units: nm²/s if q² in nm⁻²)

                all_tau_values.append(tau_val)
                all_D_values.append(D)
                tau_metadata.append({
                    'row_idx': idx,
                    'original_col': tau_col,
                    'filename': row.get('filename', f'Row_{idx}'),
                    'angle': row.get('angle', None),
                    'q^2': q_squared,
                    'tau': tau_val,
                    'D': D
                })

    if len(all_tau_values) == 0:
        raise ValueError("No valid tau values found")

    print(f"[Peak Clustering] Total tau values: {len(all_tau_values)}")
    print(f"[Peak Clustering] D range: {min(all_D_values):.3e} to {max(all_D_values):.3e} nm²/s")

    # Convert D values to log space for clustering (D values span orders of magnitude)
    # We cluster on D, not τ, because D is constant across angles for the same mode
    log_D = np.log10(all_D_values).reshape(-1, 1)

    # Perform clustering on diffusion coefficients
    if method == 'dbscan':
        labels = _cluster_dbscan(log_D, eps_factor, min_samples)
    elif method == 'log_proximity':
        labels = _cluster_log_proximity(log_D, eps_factor)
    else:
        raise ValueError(f"Unknown clustering method: {method}")

    # Analyze clusters
    unique_labels = set(labels)
    n_clusters = len(unique_labels - {-1})  # Exclude noise label (-1)
    n_noise = list(labels).count(-1)

    print(f"[Peak Clustering] Found {n_clusters} clusters")
    if n_noise > 0:
        print(f"[Peak Clustering] Warning: {n_noise} outlier points (not assigned to any cluster)")

    # Create new DataFrame with reassigned tau columns
    reassigned_df = data_df.copy()

    # Remove old tau columns
    reassigned_df = reassigned_df.drop(columns=tau_cols)

    # Also remove intensity and percentage columns if they exist
    for tau_col in tau_cols:
        tau_num = tau_col.replace(tau_prefix, '')
        intensity_col = f'intensity_{tau_num}'
        percent_col = f'normalized_sum_percent_{tau_num}'
        if intensity_col in reassigned_df.columns:
            reassigned_df = reassigned_df.drop(columns=[intensity_col])
        if percent_col in reassigned_df.columns:
            reassigned_df = reassigned_df.drop(columns=[percent_col])

    # Assign each tau value to its cluster
    for i, (tau_val, meta, label) in enumerate(zip(all_tau_values, tau_metadata, labels)):
        if label == -1:
            continue  # Skip noise points

        row_idx = meta['row_idx']
        cluster_col = f'{tau_prefix}{label + 1}'  # +1 to make 1-indexed

        # Store the tau value in the appropriate cluster column
        if cluster_col not in reassigned_df.columns:
            reassigned_df[cluster_col] = np.nan

        reassigned_df.at[row_idx, cluster_col] = tau_val

    # Calculate cluster statistics (based on D values, the physically relevant quantity)
    cluster_info = {}
    for cluster_id in range(n_clusters):
        cluster_label = cluster_id  # 0-indexed in labels array
        cluster_tau_values = [all_tau_values[i] for i, label in enumerate(labels) if label == cluster_label]
        cluster_D_values = [all_D_values[i] for i, label in enumerate(labels) if label == cluster_label]

        # Calculate statistics for both tau and D
        mean_D = np.mean(cluster_D_values)
        std_D = np.std(cluster_D_values)
        cv_D = std_D / mean_D if mean_D > 0 else 0

        cluster_info[f'mode_{cluster_id + 1}'] = {
            'n_points': len(cluster_tau_values),
            # Tau statistics (these vary with q²)
            'mean_tau': np.mean(cluster_tau_values),
            'std_tau': np.std(cluster_tau_values),
            'min_tau': np.min(cluster_tau_values),
            'max_tau': np.max(cluster_tau_values),
            'cv_tau': np.std(cluster_tau_values) / np.mean(cluster_tau_values) if np.mean(cluster_tau_values) > 0 else 0,
            # D statistics (these should be relatively constant)
            'mean_D': mean_D,
            'std_D': std_D,
            'min_D': np.min(cluster_D_values),
            'max_D': np.max(cluster_D_values),
            'cv_D': cv_D  # Coefficient of variation for D (should be small!)
        }

        print(f"  Mode {cluster_id + 1}: n={len(cluster_tau_values)}, "
              f"D_mean={mean_D:.3e} nm²/s (CV_D={cv_D:.2%}), "
              f"τ_range=[{np.min(cluster_tau_values):.3e}, {np.max(cluster_tau_values):.3e}] s")

    return reassigned_df, cluster_info


def _cluster_dbscan(log_D: np.ndarray, eps_factor: float, min_samples: int) -> np.ndarray:
    """
    Cluster using DBSCAN in log-space

    Args:
        log_D: Log10-transformed diffusion coefficient values
        eps_factor: Epsilon as fraction of data range
        min_samples: Minimum samples per cluster

    Returns:
        Cluster labels (-1 for noise)
    """
    # Calculate epsilon based on data range
    data_range = log_D.max() - log_D.min()
    eps = eps_factor * data_range

    print(f"[DBSCAN] eps={eps:.3f} (log-D scale), min_samples={min_samples}")

    # Perform DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    labels = clustering.fit_predict(log_D)

    return labels


def _cluster_log_proximity(log_D: np.ndarray, threshold_factor: float) -> np.ndarray:
    """
    Simple clustering based on log-space proximity
    Groups values that are within threshold_factor * range of each other

    Args:
        log_D: Log10-transformed diffusion coefficient values
        threshold_factor: Proximity threshold as fraction of range

    Returns:
        Cluster labels
    """
    # Sort values with their original indices
    sorted_indices = np.argsort(log_D.flatten())
    sorted_log_D = log_D.flatten()[sorted_indices]

    # Calculate threshold
    data_range = sorted_log_D.max() - sorted_log_D.min()
    threshold = threshold_factor * data_range

    print(f"[Log Proximity] threshold={threshold:.3f} (log-D scale)")

    # Assign clusters
    labels = np.full(len(log_D), -1)
    current_cluster = 0

    for i in range(len(sorted_log_D)):
        original_idx = sorted_indices[i]

        if labels[original_idx] != -1:
            continue  # Already assigned

        # Start new cluster
        labels[original_idx] = current_cluster

        # Find all nearby points
        for j in range(i + 1, len(sorted_log_D)):
            original_j = sorted_indices[j]

            if sorted_log_D[j] - sorted_log_D[i] <= threshold:
                labels[original_j] = current_cluster
            else:
                break  # No more nearby points

        current_cluster += 1

    return labels


def robust_linear_regression(X: np.ndarray, Y: np.ndarray,
                            method: str = 'ransac',
                            **kwargs) -> Tuple[float, float, np.ndarray, Dict]:
    """
    Perform robust linear regression to handle outliers

    Args:
        X: Independent variable (q²)
        Y: Dependent variable (Γ)
        method: 'ransac', 'theil-sen', or 'huber'
        **kwargs: Additional parameters for the regression method

    Returns:
        Tuple of (slope, intercept, inlier_mask, stats_dict)
    """
    import statsmodels.api as sm
    from sklearn.linear_model import RANSACRegressor, TheilSenRegressor, HuberRegressor
    from sklearn.linear_model import LinearRegression

    X_2d = X.reshape(-1, 1)

    if method == 'ransac':
        # RANSAC regression
        min_samples = kwargs.get('min_samples', max(2, int(0.5 * len(X))))
        residual_threshold = kwargs.get('residual_threshold', None)

        ransac = RANSACRegressor(
            estimator=LinearRegression(),
            min_samples=min_samples,
            residual_threshold=residual_threshold,
            random_state=42
        )
        ransac.fit(X_2d, Y)

        slope = ransac.estimator_.coef_[0]
        intercept = ransac.estimator_.intercept_
        inlier_mask = ransac.inlier_mask_

        print(f"[RANSAC] Inliers: {np.sum(inlier_mask)}/{len(X)} points")

    elif method == 'theil-sen':
        # Theil-Sen regression (median-based, very robust)
        ts = TheilSenRegressor(random_state=42)
        ts.fit(X_2d, Y)

        slope = ts.coef_[0]
        intercept = ts.intercept_

        # Calculate residuals and identify inliers (within 2 std)
        Y_pred = slope * X + intercept
        residuals = Y - Y_pred
        threshold = 2 * np.std(residuals)
        inlier_mask = np.abs(residuals) <= threshold

        print(f"[Theil-Sen] Inliers: {np.sum(inlier_mask)}/{len(X)} points (±2σ)")

    elif method == 'huber':
        # Huber regression (robust to outliers)
        huber = HuberRegressor(epsilon=kwargs.get('epsilon', 1.35))
        huber.fit(X_2d, Y)

        slope = huber.coef_[0]
        intercept = huber.intercept_

        # Calculate residuals and identify inliers
        Y_pred = slope * X + intercept
        residuals = Y - Y_pred
        threshold = 2 * np.std(residuals)
        inlier_mask = np.abs(residuals) <= threshold

        print(f"[Huber] Inliers: {np.sum(inlier_mask)}/{len(X)} points (±2σ)")

    else:
        raise ValueError(f"Unknown method: {method}")

    # Calculate statistics using statsmodels for consistency
    X_inliers = X[inlier_mask]
    Y_inliers = Y[inlier_mask]

    if len(X_inliers) < 2:
        raise ValueError("Not enough inliers for regression")

    X_with_const = sm.add_constant(X_inliers)
    model = sm.OLS(Y_inliers, X_with_const).fit()

    stats_dict = {
        'slope': slope,
        'intercept': intercept,
        'slope_se': model.bse[1],
        'intercept_se': model.bse[0],
        'r_squared': model.rsquared,
        'n_inliers': np.sum(inlier_mask),
        'n_outliers': len(X) - np.sum(inlier_mask),
        'outlier_fraction': (len(X) - np.sum(inlier_mask)) / len(X)
    }

    return slope, intercept, inlier_mask, stats_dict


def analyze_diffusion_coefficient_robust(data_df: pd.DataFrame,
                                         q_squared_col: str,
                                         gamma_cols: List[str],
                                         method_names: Optional[List[str]] = None,
                                         robust_method: str = 'ransac',
                                         x_range: Optional[Tuple[float, float]] = None,
                                         show_plots: bool = True) -> pd.DataFrame:
    """
    Analyze diffusion coefficients with robust regression and outlier detection

    This is an enhanced version of cumulants.analyze_diffusion_coefficient
    that uses robust regression methods to handle outliers.

    Args:
        data_df: DataFrame with q² and gamma columns
        q_squared_col: Name of q² column
        gamma_cols: List of gamma column names
        method_names: Optional display names for each method
        robust_method: 'ransac', 'theil-sen', or 'huber'
        x_range: Optional (min, max) range for q² values
        show_plots: Whether to show plots

    Returns:
        DataFrame with regression results
    """
    import matplotlib.pyplot as plt
    import statsmodels.api as sm

    if not isinstance(gamma_cols, list):
        gamma_cols = [gamma_cols]

    if method_names is None:
        method_names = gamma_cols
    elif not isinstance(method_names, list):
        method_names = [method_names]

    if len(method_names) < len(gamma_cols):
        method_names.extend(['' for _ in range(len(gamma_cols) - len(method_names))])

    if q_squared_col not in data_df.columns:
        raise ValueError(f"Column '{q_squared_col}' not found in the DataFrame")

    X_full = data_df[q_squared_col].values
    all_results = []

    print(f"\n[Robust Regression] Using {robust_method.upper()} method")

    for i, gamma_col in enumerate(gamma_cols):
        if gamma_col not in data_df.columns:
            print(f"Column '{gamma_col}' not found. Skipping...")
            continue

        method_name = method_names[i] if i < len(method_names) else ''
        Y_full = data_df[gamma_col].values

        # First, filter out NaN values
        valid_mask = ~(np.isnan(X_full) | np.isnan(Y_full))
        X_valid = X_full[valid_mask]
        Y_valid = Y_full[valid_mask]

        if len(X_valid) < 3:
            print(f"Warning: Only {len(X_valid)} valid (non-NaN) points for {gamma_col}. Need at least 3.")
            continue

        # Filter by x_range if specified
        if x_range is not None:
            min_x, max_x = x_range
            mask = (X_valid >= min_x) & (X_valid <= max_x)
            X_fit = X_valid[mask]
            Y_fit = Y_valid[mask]
            fit_range_text = f" (fit range: {min_x:.3f} to {max_x:.3f})"
        else:
            X_fit = X_valid
            Y_fit = Y_valid
            fit_range_text = ""

        if len(X_fit) < 3:
            print(f"Warning: Only {len(X_fit)} points for {gamma_col}. Need at least 3.")
            continue

        # Perform robust regression
        try:
            slope, intercept, inlier_mask, stats = robust_linear_regression(
                X_fit, Y_fit, method=robust_method
            )
        except Exception as e:
            print(f"Error in robust regression for {gamma_col}: {e}")
            continue

        # Plot if requested
        if show_plots:
            plt.figure(figsize=(10, 6))

            # Plot all valid data (non-NaN)
            plt.scatter(X_valid, Y_valid, alpha=0.3, s=30, label='All valid data', color='lightgray')

            # Highlight fitting range
            if x_range is not None:
                plt.scatter(X_fit, Y_fit, alpha=0.5, s=50, label='Fitting range', color='blue')

            # Highlight inliers vs outliers
            X_inliers = X_fit[inlier_mask]
            Y_inliers = Y_fit[inlier_mask]
            X_outliers = X_fit[~inlier_mask]
            Y_outliers = Y_fit[~inlier_mask]

            plt.scatter(X_inliers, Y_inliers, s=60, label='Inliers', color='green', marker='o', edgecolors='black')
            if len(X_outliers) > 0:
                plt.scatter(X_outliers, Y_outliers, s=60, label='Outliers (excluded)',
                           color='red', marker='x', linewidths=2)

            # Plot regression line
            X_line = np.linspace(X_valid.min(), X_valid.max(), 100)
            Y_line = slope * X_line + intercept
            plt.plot(X_line, Y_line, 'r-', linewidth=2,
                    label=f'{robust_method.upper()} fit{fit_range_text}')

            plt.xlabel(r'q$^2$ [nm$^{-2}$]')
            plt.ylabel(r'$\Gamma$ [1/s]')

            title = f'Robust Regression: q² vs. Γ ({robust_method.upper()})'
            if method_name:
                title += f' - {method_name}'
            plt.title(title)

            # Add statistics box
            stats_text = (
                f"R² = {stats['r_squared']:.4f}\n"
                f"Slope = {slope:.2e} ± {stats['slope_se']:.2e}\n"
                f"Inliers: {stats['n_inliers']}/{len(X_fit)} "
                f"({(1-stats['outlier_fraction'])*100:.1f}%)"
            )
            plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round',
                    facecolor='white', alpha=0.8))

            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

        # Store results
        results_dict = {
            'Column': gamma_col,
            'Method': method_name,
            'R_squared': stats['r_squared'],
            'q^2_coef': slope,
            'q^2_se': stats['slope_se'],
            'const_coef': intercept,
            'const_se': stats['intercept_se'],
            'N_points_fit': len(X_fit),
            'N_inliers': stats['n_inliers'],
            'N_outliers': stats['n_outliers'],
            'Outlier_fraction': stats['outlier_fraction'],
            'Robust_method': robust_method,
            'X_range_min': X_fit.min(),
            'X_range_max': X_fit.max()
        }

        all_results.append(results_dict)

        print(f"\n--- Robust Regression Results {method_name}{fit_range_text} ---")
        print(f"Method: {robust_method.upper()}")
        print(f"Slope (D): {slope:.4e} ± {stats['slope_se']:.4e} nm²/s")
        print(f"R²: {stats['r_squared']:.4f}")
        print(f"Inliers: {stats['n_inliers']}/{len(X_fit)} ({(1-stats['outlier_fraction'])*100:.1f}%)")
        print(f"Outliers removed: {stats['n_outliers']}")

    if not all_results:
        print("No valid columns were analyzed.")
        return pd.DataFrame()

    return pd.DataFrame(all_results)


def plot_peak_clustering(data_df: pd.DataFrame,
                         tau_prefix: str = 'tau_',
                         cluster_info: Optional[Dict] = None,
                         show_plot: bool = True) -> Figure:
    """
    Visualize the peak clustering results

    Shows diffusion coefficients (D), which is what was actually clustered.
    This is the physically meaningful quantity that should be constant across angles.

    Args:
        data_df: DataFrame with tau columns (after clustering) AND q^2 column
        tau_prefix: Prefix for tau columns
        cluster_info: Clustering information dict (from cluster_peaks_across_datasets)
        show_plot: Whether to display the plot

    Returns:
        Figure object
    """
    # Find all tau columns
    tau_cols = sorted([col for col in data_df.columns if col.startswith(tau_prefix)])

    if not tau_cols:
        raise ValueError(f"No columns starting with '{tau_prefix}' found")

    # Check for q^2 column
    if 'q^2' not in data_df.columns:
        raise ValueError("DataFrame must contain 'q^2' column for D calculation")

    # Collect all tau and D values with their cluster assignment
    cluster_tau_data = {col: [] for col in tau_cols}
    cluster_D_data = {col: [] for col in tau_cols}

    for tau_col in tau_cols:
        # Get tau values and corresponding q² values
        for idx, row in data_df.iterrows():
            tau_val = row[tau_col]
            q_squared = row.get('q^2', None)

            if pd.notna(tau_val) and tau_val > 0 and q_squared is not None and q_squared > 0:
                # Calculate D = 1/(τ × q²)
                gamma = 1.0 / tau_val
                D = gamma / q_squared

                cluster_tau_data[tau_col].append(tau_val)
                cluster_D_data[tau_col].append(D)

    # Create figure with two subplots
    fig = Figure(figsize=(14, 6))

    # Subplot 1: Diffusion coefficients (D) in log space - this is what was clustered!
    ax1 = fig.add_subplot(121)

    colors = plt.cm.tab10(np.linspace(0, 1, len(tau_cols)))

    for i, tau_col in enumerate(tau_cols):
        D_values = cluster_D_data[tau_col]
        if len(D_values) > 0:
            cluster_num = int(tau_col.replace(tau_prefix, ''))
            log_D = np.log10(D_values)

            # Plot as scatter with jitter for visibility
            y_jitter = np.random.normal(0, 0.02, len(log_D))
            ax1.scatter(log_D, y_jitter + i, c=[colors[i]], s=60,
                       alpha=0.7, label=f'Mode {cluster_num} (n={len(D_values)})',
                       edgecolors='black', linewidths=0.5)

            # Add mean line
            mean_log_D = np.mean(log_D)
            ax1.axvline(mean_log_D, color=colors[i], linestyle='--',
                       alpha=0.5, linewidth=1.5)

    ax1.set_xlabel(r'log$_{10}$(D) [log(nm²/s)]', fontsize=12)
    ax1.set_ylabel('Mode Number', fontsize=12)
    ax1.set_title('Clustering based on Diffusion Coefficient (D = Γ/q²)', fontsize=14, fontweight='bold')
    ax1.set_yticks(range(len(tau_cols)))
    ax1.set_yticklabels([int(col.replace(tau_prefix, '')) for col in tau_cols])
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.legend(loc='best', fontsize=9)

    # Subplot 2: Distribution of D values per cluster (should be tight!)
    ax2 = fig.add_subplot(122)

    positions = []
    data_for_box = []
    labels_list = []

    for i, tau_col in enumerate(tau_cols):
        D_values = cluster_D_data[tau_col]
        if len(D_values) > 0:
            positions.append(i)
            data_for_box.append(np.log10(D_values))
            cluster_num = int(tau_col.replace(tau_prefix, ''))
            labels_list.append(f'Mode {cluster_num}')

    bp = ax2.boxplot(data_for_box, positions=positions, labels=labels_list,
                     patch_artist=True, widths=0.6)

    # Color the boxes
    for patch, color in zip(bp['boxes'], colors[:len(positions)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax2.set_ylabel(r'log$_{10}$(D) [log(nm²/s)]', fontsize=12)
    ax2.set_xlabel('Mode', fontsize=12)
    ax2.set_title('Distribution of D per Mode (should be constant!)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add statistics text if available
    if cluster_info:
        stats_text = "Cluster Statistics (D-based):\n"
        for mode_key, info in cluster_info.items():
            mode_num = mode_key.replace('mode_', '')
            stats_text += f"Mode {mode_num}: n={info['n_points']}, "
            stats_text += f"D={info['mean_D']:.2e} nm²/s, "
            stats_text += f"CV_D={info['cv_D']:.1%}\n"
            if info['cv_D'] > 0.15:  # Warn if CV is too high
                stats_text += f"  ⚠ High variability!\n"

        fig.text(0.02, 0.98, stats_text, transform=fig.transFigure,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if show_plot:
        plt.show()

    return fig
