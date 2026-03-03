"""
clustering.py
=============
Population clustering and statistics for multi-angle DLS decay rate data.
Clusters Γ or D = Γ/q² values across angles into discrete populations,
assigns reliability flags and exports population-level summaries.

Functions
---------
cluster_all_gammas(data_df, gamma_cols, q_squared_col, ...)
    Main entry point. Clusters decay rates into populations using
    hierarchical or silhouette-refined agglomerative clustering,
    applies min_abundance filtering, remaps cluster IDs to sequential
    population numbers and returns a clustered DataFrame + cluster_info.

get_reliable_gamma_cols(cluster_info, prefix)
    Return the list of gamma_popN column names for reliable populations.

aggregate_peak_stats(cluster_info, data_df, stat_prefixes)
    Map per-peak statistics (skewness, kurtosis) from regularized fit
    results onto the correct sequential population numbers, using the
    cluster_id_to_pop mapping stored in cluster_info.

_hierarchical_clustering(gammas_df, n_clusters, ...)        [internal]
    Agglomerative clustering with automatic or fixed cluster count.

_simple_threshold_clustering(gammas_df, distance_threshold) [internal]
    Threshold-based clustering in log Γ space.

_refine_clusters_with_silhouette(gammas_df, threshold, ...) [internal]
    Iteratively merge clusters to maximise silhouette score.

_calculate_silhouette_scores(gammas_df, cluster_labels)     [internal]
    Compute per-cluster and mean silhouette scores.

_plot_population_distribution(gammas_df, stats_df, ...)     [internal]
    Scatter and histogram plots of population distributions per angle.

_show_uncertainty_removal_preview(gammas_df, ...)           [internal]
    Preview effect of removing uncertain points before committing.

Dependencies: numpy, pandas, scipy, sklearn, matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.metrics import silhouette_score

#cluster gamma values across all files into physical populations
def cluster_all_gammas(data_df, gamma_cols, q_squared_col=None,
                       enable_clustering=True, normalize_by_q2=False, 
                       method='hierarchical', n_clusters='auto', 
                       distance_threshold=0.3, min_abundance=0.3,
                       clustering_strategy='simple', silhouette_threshold=0.3,
                       uncertainty_flags=False, uncertainty_threshold=0.5,
                       plot=True, experiment_name=''):
    """
    Can operate in two modes:
    1. Clustering enabled: Groups gammas across files by similarity
    2. Clustering disabled: Pass-through mode (returns data unchanged)
    """
    
    #PASS-THROUGH MODE
    if not enable_clustering:
        print("CLUSTERING DISABLED - Pass-through mode")
        print(f"Using original gamma columns as-is: {gamma_cols}")
        
        cluster_info = {
            'n_populations': len(gamma_cols),
            'clustering_enabled': False,
            'reliable_populations': list(range(1, len(gamma_cols) + 1)),
            'original_gamma_cols': gamma_cols  #store for get_gamma_cols_for_analysis
        }
        
        return data_df, cluster_info
    
    #CLUSTERING MODE
    # Validate inputs
    if 'filename' not in data_df.columns:
        raise ValueError("DataFrame must have 'filename' column")
    
    for col in gamma_cols:
        if col not in data_df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")
    
    if normalize_by_q2:
        if q_squared_col is None:
            raise ValueError("q_squared_col must be provided when normalize_by_q2=True")
        if q_squared_col not in data_df.columns:
            raise ValueError(f"Column '{q_squared_col}' not found in DataFrame")
    
    print("CROSS-FILE GAMMA CLUSTERING")
    if normalize_by_q2:
        print("Mode: Clustering on D = Γ/q² (angle-independent)")
    else:
        print("Mode: Clustering on Γ (raw values)")
    print(f"Method: {method}")
    print(f"Strategy: {clustering_strategy}")
    if uncertainty_flags:
        print(f"Uncertainty flags: enabled (threshold={uncertainty_threshold})")
    print(f"Input: {len(data_df)} files, {len(gamma_cols)} gamma columns")
    
    # Step 1: Collect all gammas with their metadata
    all_gammas = []
    for idx, row in data_df.iterrows():
        filename = row['filename']
        q_squared = row[q_squared_col] if normalize_by_q2 else None
        
        for gamma_col in gamma_cols:
            gamma_val = row[gamma_col]
            if pd.notna(gamma_val) and gamma_val > 0:
                if normalize_by_q2:
                    if pd.notna(q_squared) and q_squared > 0:
                        # Calculate D = Γ/q²
                        D_val = gamma_val / q_squared
                        log_val = np.log10(D_val)
                    else:
                        continue  # Skip if q² is invalid
                else:
                    log_val = np.log10(gamma_val)
                
                all_gammas.append({
                    'file_idx': idx,
                    'filename': filename,
                    'original_col': gamma_col,
                    'gamma': gamma_val,
                    'q_squared': q_squared if normalize_by_q2 else np.nan,
                    'D': gamma_val / q_squared if normalize_by_q2 else np.nan,
                    'log_clustering_val': log_val  # This is what we cluster on
                })
    
    if len(all_gammas) == 0:
        raise ValueError("No valid gamma values found!")
    
    gammas_df = pd.DataFrame(all_gammas)
    n_total_gammas = len(gammas_df)
    
    print(f"Total non-NaN gammas: {n_total_gammas}")
    print(f"Gamma range: {gammas_df['gamma'].min():.1f} to {gammas_df['gamma'].max():.1f} s⁻¹")
    
    if normalize_by_q2:
        print(f"D range: {gammas_df['D'].min():.1e} to {gammas_df['D'].max():.1e} nm²/s")
        print(f"Log(D) range: {gammas_df['log_clustering_val'].min():.2f} to {gammas_df['log_clustering_val'].max():.2f}")
    else:
        print(f"Log(Γ) range: {gammas_df['log_clustering_val'].min():.2f} to {gammas_df['log_clustering_val'].max():.2f}")
    
    # Step 2: Perform clustering
    if method == 'hierarchical':
        cluster_labels, linkage_matrix = _hierarchical_clustering(
            gammas_df, 
            n_clusters=n_clusters,
            distance_threshold=distance_threshold,
            normalize_by_q2=normalize_by_q2,
            plot=plot
        )
    elif method == 'simple':
        cluster_labels = _simple_threshold_clustering(
            gammas_df,
            distance_threshold=distance_threshold
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'hierarchical' or 'simple'")
    
    gammas_df['cluster'] = cluster_labels
    n_populations = len(np.unique(cluster_labels))
    
    print(f"\n→ Found {n_populations} population(s)")
    
    # Step 3: Calculate population statistics and identify outlier populations
    population_stats = []
    for cluster_id in sorted(np.unique(cluster_labels)):
        cluster_gammas = gammas_df[gammas_df['cluster'] == cluster_id]['gamma']
        cluster_files  = gammas_df[gammas_df['cluster'] == cluster_id]['filename'].unique()

        abundance  = len(cluster_files) / len(data_df)
        mean_gamma = np.exp(np.mean(np.log(cluster_gammas)))
        std_gamma  = np.std(cluster_gammas)
        is_reliable = abundance >= min_abundance
        status = "RELIABLE" if is_reliable else "OUTLIER (below min_abundance)"

        stats_dict = {
            'cluster_id' : cluster_id,
            'n_gammas'   : len(cluster_gammas),
            'n_files'    : len(cluster_files),
            'abundance'  : abundance,
            'mean_gamma' : mean_gamma,
            'std_gamma'  : std_gamma,
            'reliable'   : is_reliable,
            'status'     : status
        }

        if normalize_by_q2:
            cluster_D = gammas_df[gammas_df['cluster'] == cluster_id]['D']
            mean_D = np.exp(np.mean(np.log(cluster_D)))
            std_D  = np.std(cluster_D)
            stats_dict['mean_D'] = mean_D
            stats_dict['std_D']  = std_D

        population_stats.append(stats_dict)

        print(f"\nPopulation {cluster_id + 1}:")
        print(f"  Mean Γ: {mean_gamma:.1f} s⁻¹ (±{std_gamma:.1f})")
        if normalize_by_q2:
            print(f"  Mean D: {mean_D:.2e} nm²/s (±{std_D:.2e})")
        print(f"  Abundance: {len(cluster_files)}/{len(data_df)} files ({abundance*100:.1f}%)")
        print(f"  Status: {status}")

    stats_df = pd.DataFrame(population_stats)

    #flag points belonging to outlier populations
    outlier_cluster_ids = set(stats_df[~stats_df['reliable']]['cluster_id'].tolist())
    gammas_df['outlier'] = gammas_df['cluster'].isin(outlier_cluster_ids)
    n_outlier_points = gammas_df['outlier'].sum()
    if n_outlier_points > 0:
        print(f"\n→ {n_outlier_points} point(s) flagged as outliers "
              f"(belong to population(s) below min_abundance={min_abundance})")

    #work on reliable points only for all further analysis
    gammas_df_reliable = gammas_df[~gammas_df['outlier']].copy()
    reliable_cluster_ids = sorted(stats_df[stats_df['reliable']]['cluster_id'].tolist())
    n_reliable_populations = len(reliable_cluster_ids)

    if n_reliable_populations == 0:
        print("WARNING: No reliable populations found. "
              "Consider lowering min_abundance or distance_threshold.")
        stats_df_reliable = pd.DataFrame()
    else:
        stats_df_reliable = stats_df[stats_df['reliable']].reset_index(drop=True)
    
    # Step 3.5: Silhouette / z-score analysis on reliable populations only
    silhouette_scores = None
    need_silhouette = (clustering_strategy == 'silhouette_refined' or uncertainty_flags)

    if need_silhouette and n_reliable_populations > 0:

        if n_reliable_populations == 1:
            # --- Monomodal case: z-score outlier detection ---
            print("OUTLIER DETECTION (monomodal — z-score)")
            vals = gammas_df_reliable['log_clustering_val'].values
            mean_val = np.mean(vals)
            std_val  = np.std(vals)
            if std_val > 0:
                z_scores = np.abs((vals - mean_val) / std_val)
            else:
                z_scores = np.zeros(len(vals))
            gammas_df_reliable = gammas_df_reliable.copy()
            gammas_df_reliable['z_score']  = z_scores
            gammas_df_reliable['uncertain'] = z_scores > uncertainty_threshold
            val_label = "log(D)" if normalize_by_q2 else "log(Γ)"
            print(f"Mean {val_label}: {mean_val:.3f}  Std: {std_val:.3f}")
            print(f"Flagging points with |z| > {uncertainty_threshold} as uncertain")
            n_uncertain = gammas_df_reliable['uncertain'].sum()
            print(f"→ {n_uncertain}/{len(gammas_df_reliable)} uncertain point(s) in single population")
            if n_uncertain > 0:
                pct_uncertain = n_uncertain / len(gammas_df_reliable) * 100
                _show_uncertainty_removal_preview(gammas_df_reliable, n_uncertain, normalize_by_q2)
                if pct_uncertain > 30.0:
                    print("\nWARNING: >30% uncertain — consider adjusting distance_threshold.")
                answer = input("\nRemove outlier points? (y/n): ").strip().lower()
                if answer == 'y':
                    gammas_df_reliable = gammas_df_reliable[
                        ~gammas_df_reliable['uncertain']].copy()
                    print(f"→ Removed {n_uncertain} outlier points.")
                else:
                    print("\n→ Keeping all data (outliers flagged but not removed)")

        else:
            # --- Multimodal case: silhouette uncertainty on reliable points ---
            print("SILHOUETTE ANALYSIS (reliable populations only)")
            rel_labels = gammas_df_reliable['cluster'].values
            silhouette_scores = _calculate_silhouette_scores(gammas_df_reliable, rel_labels)
            gammas_df_reliable = gammas_df_reliable.copy()
            gammas_df_reliable['silhouette'] = silhouette_scores
            avg_silhouette = np.mean(silhouette_scores)
            print(f"Average silhouette score: {avg_silhouette:.3f}")

            # Step 3.6: Silhouette-based refinement
            if clustering_strategy == 'silhouette_refined':
                print("SILHOUETTE-BASED REFINEMENT")
                n_reassigned = _refine_clusters_with_silhouette(
                    gammas_df_reliable, silhouette_threshold, normalize_by_q2
                )
                if n_reassigned > 0:
                    print(f"→ Reassigned {n_reassigned} points")
                    rel_labels = gammas_df_reliable['cluster'].values
                    silhouette_scores = _calculate_silhouette_scores(gammas_df_reliable, rel_labels)
                    gammas_df_reliable['silhouette'] = silhouette_scores
                    print(f"Silhouette after refinement: {np.mean(silhouette_scores):.3f}")
                else:
                    print("→ No reassignments needed")

            # Step 3.7: Uncertainty flagging on reliable points
            if uncertainty_flags:
                print("UNCERTAINTY FLAGGING")
                gammas_df_reliable['uncertain'] = (
                    gammas_df_reliable['silhouette'] < uncertainty_threshold
                )
                n_uncertain = gammas_df_reliable['uncertain'].sum()
                pct_uncertain = n_uncertain / len(gammas_df_reliable) * 100
                print(f"→ Flagged {n_uncertain}/{len(gammas_df_reliable)} "
                      f"({pct_uncertain:.1f}%) as uncertain")

                for cluster_id in sorted(gammas_df_reliable['cluster'].unique()):
                    cd = gammas_df_reliable[gammas_df_reliable['cluster'] == cluster_id]
                    n_u = cd['uncertain'].sum()
                    pct = n_u / len(cd) * 100 if len(cd) > 0 else 0
                    print(f"  Population {cluster_id + 1}: {n_u}/{len(cd)} uncertain ({pct:.1f}%)")

                # Step 3.8: Interactive decision
                remove_uncertain_points = False
                
                _show_uncertainty_removal_preview(gammas_df_reliable, n_uncertain, normalize_by_q2)
                if pct_uncertain > 30.0:
                    print("\nWARNING: >30% uncertain — consider adjusting distance_threshold.")
                answer = input("\nRemove uncertain points and re-cluster? (y/n): ").strip().lower()
                remove_uncertain_points = (answer == 'y')

                # Step 3.9: Re-cluster after removal
                if remove_uncertain_points:
                    print("\n" + "="*70)
                    print("RE-CLUSTERING AFTER UNCERTAINTY REMOVAL")
                    print("="*70)
                    gammas_df_clean = gammas_df_reliable[
                        ~gammas_df_reliable['uncertain']
                    ].copy()
                    print(f"Removed {n_uncertain} uncertain points, "
                          f"re-clustering {len(gammas_df_clean)} points...")

                    if method == 'hierarchical':
                        new_labels, _ = _hierarchical_clustering(
                            gammas_df_clean, n_clusters=n_clusters,
                            distance_threshold=distance_threshold,
                            normalize_by_q2=normalize_by_q2, plot=False
                        )
                    else:
                        new_labels = _simple_threshold_clustering(
                            gammas_df_clean, distance_threshold=distance_threshold
                        )
                    gammas_df_clean['cluster'] = new_labels

                    # Recalculate stats
                    population_stats_clean = []
                    for cluster_id in sorted(gammas_df_clean['cluster'].unique()):
                        cg = gammas_df_clean[gammas_df_clean['cluster'] == cluster_id]['gamma']
                        cf = gammas_df_clean[gammas_df_clean['cluster'] == cluster_id]['filename'].unique()
                        ab = len(cf) / len(data_df)
                        mg = np.exp(np.mean(np.log(cg)))
                        sd = np.std(cg)
                        sd_clean = {
                            'cluster_id': cluster_id, 'n_gammas': len(cg),
                            'n_files': len(cf), 'abundance': ab,
                            'mean_gamma': mg, 'std_gamma': sd,
                            'reliable': ab >= min_abundance,
                            'status': "RELIABLE" if ab >= min_abundance else "OUTLIER"
                        }
                        if normalize_by_q2:
                            cd2 = gammas_df_clean[gammas_df_clean['cluster'] == cluster_id]['D']
                            sd_clean['mean_D'] = np.exp(np.mean(np.log(cd2)))
                            sd_clean['std_D']  = np.std(cd2)
                        population_stats_clean.append(sd_clean)

                    gammas_df_reliable = gammas_df_clean
                    stats_df_reliable  = pd.DataFrame(population_stats_clean)
                    n_reliable_populations = len(stats_df_reliable)
                    reliable_cluster_ids = sorted(
                        stats_df_reliable['cluster_id'].tolist()
                    )

                    #recalculate silhouette
                    sc = _calculate_silhouette_scores(
                        gammas_df_reliable, gammas_df_reliable['cluster'].values
                    )
                    gammas_df_reliable['silhouette'] = sc
                    silhouette_scores = sc
                    print(f"\nRe-clustering complete — {n_reliable_populations} populations, "
                          f"silhouette: {np.mean(sc):.3f}")
                else:
                    print("\n→ Keeping all data (uncertain points flagged but not removed)")
    
    # Step 4: Reassign reliable gammas to population columns
    # Remap reliable cluster_ids to sequential 1, 2, 3, ... population numbers
    cluster_id_to_pop = {cid: i + 1 for i, cid in enumerate(reliable_cluster_ids)}
    clustered_df = data_df.copy()

    for pop_num in cluster_id_to_pop.values():
        clustered_df[f'gamma_pop{pop_num}'] = np.nan
        if uncertainty_flags:
            clustered_df[f'uncertain_pop{pop_num}'] = np.nan
        clustered_df[f'outlier_pop{pop_num}'] = np.nan

    #fill reliable points
    for _, row in gammas_df_reliable.iterrows():
        file_idx   = row['file_idx']
        cluster_id = row['cluster']
        if cluster_id not in cluster_id_to_pop:
            continue
        pop_num  = cluster_id_to_pop[cluster_id]
        pop_col  = f'gamma_pop{pop_num}'
        if pd.isna(clustered_df.loc[file_idx, pop_col]):
            clustered_df.loc[file_idx, pop_col] = row['gamma']
            if uncertainty_flags and 'uncertain' in row:
                clustered_df.loc[file_idx, f'uncertain_pop{pop_num}'] = row['uncertain']
            clustered_df.loc[file_idx, f'outlier_pop{pop_num}'] = False

    #mark outlier points in output
    for _, row in gammas_df[gammas_df['outlier']].iterrows():
        file_idx = row['file_idx']
        # Outliers don't belong to a reliable population — mark in all pop columns
        for pop_num in cluster_id_to_pop.values():
            out_col = f'outlier_pop{pop_num}'
            if out_col in clustered_df.columns and pd.isna(clustered_df.loc[file_idx, out_col]):
                clustered_df.loc[file_idx, out_col] = True

    # Step 5: Overall silhouette score on reliable populations
    avg_silhouette = None
    if silhouette_scores is not None and n_reliable_populations > 1:
        try:
            log_vals = gammas_df_reliable['log_clustering_val'].values.reshape(-1, 1)
            avg_silhouette = silhouette_score(
                log_vals, gammas_df_reliable['cluster'].values
            )
        except Exception:
            avg_silhouette = None

    # Step 6: Plot population distribution
    if plot:
        _plot_population_distribution(gammas_df, stats_df, min_abundance, normalize_by_q2, experiment_name=experiment_name)

    # Prepare cluster_info
    if stats_df_reliable.empty:
        reliable_abundances = []
        reliable_means      = []
        reliable_stds       = []
    else:
        reliable_abundances = stats_df_reliable['abundance'].tolist()
        reliable_means      = stats_df_reliable['mean_gamma'].tolist()
        reliable_stds       = stats_df_reliable['std_gamma'].tolist()

    cluster_info = {
        'n_populations'           : n_reliable_populations,
        'population_abundances'   : reliable_abundances,
        'population_means'        : reliable_means,
        'population_stds'         : reliable_stds,
        'silhouette_score'        : avg_silhouette,
        'reliable_populations'    : list(cluster_id_to_pop.values()),
        'population_stats'        : stats_df_reliable,
        'clustering_enabled'      : True,
        'clustering_strategy'     : clustering_strategy,
        'uncertainty_flags_enabled': uncertainty_flags,
        'n_outlier_points'        : int(gammas_df['outlier'].sum()),
        'gammas_df'               : gammas_df_reliable,
        'cluster_id_to_pop'       : cluster_id_to_pop,
    }

    if normalize_by_q2 and not stats_df_reliable.empty and 'mean_D' in stats_df_reliable.columns:
        cluster_info['population_D_means'] = stats_df_reliable['mean_D'].tolist()
        cluster_info['population_D_stds']  = stats_df_reliable['std_D'].tolist()

    print("\n" + "="*70)
    print("In SUMMARY:")
    print(f"Reliable populations: {n_reliable_populations}")
    n_out = int(gammas_df['outlier'].sum())
    if n_out > 0:
        print(f"Outlier points (below min_abundance): {n_out}")
    if avg_silhouette is not None:
        print(f"Clustering quality (silhouette): {avg_silhouette:.3f}")
    print("="*70)

    return clustered_df, cluster_info

#perform hierarchical clustering on gamma values
def _hierarchical_clustering(gammas_df, n_clusters='auto', 
                             distance_threshold=0.3, normalize_by_q2=False, plot=True):    
    #prepare data for clustering (use log_clustering_val which is either log(Γ) or log(D))
    X = gammas_df['log_clustering_val'].values.reshape(-1, 1)
    
    #perform hierarchical clustering
    linkage_matrix = linkage(X, method='ward')
    
    #determine number of clusters
    if n_clusters == 'auto':
        #use distance threshold to cut dendrogram
        cluster_labels = fcluster(linkage_matrix, t=distance_threshold, 
                                  criterion='distance')
        n_found = len(np.unique(cluster_labels))
        print(f"\nAuto-detected {n_found} clusters (distance threshold={distance_threshold})")
    else:
        #use specified number
        cluster_labels = fcluster(linkage_matrix, t=n_clusters, 
                                  criterion='maxclust')
        print(f"\nForcing {n_clusters} clusters")
    
    #convert to 0-indexed
    cluster_labels = cluster_labels - 1
    
    #plot dendrogram
    if plot:
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        dendrogram(linkage_matrix, color_threshold=distance_threshold)
        plt.axhline(y=distance_threshold, c='red', linestyle='--', 
                   label=f'Threshold={distance_threshold}')
        plt.xlabel('Data point index')
        plt.ylabel('Distance (log-space)')
        
        if normalize_by_q2:
            plt.title('Hierarchical Clustering Dendrogram (on D = Γ/q²)')
        else:
            plt.title('Hierarchical Clustering Dendrogram (on Γ)')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        for cluster_id in np.unique(cluster_labels):
            mask = cluster_labels == cluster_id
            plt.hist(gammas_df.loc[mask, 'log_clustering_val'], 
                    alpha=0.6, label=f'Pop {cluster_id + 1}', bins=20)
        
        if normalize_by_q2:
            plt.xlabel('log₁₀(D) [log(nm²/s)]')
            plt.title('D Distribution by Population')
        else:
            plt.xlabel('log₁₀(Γ) [log(s⁻¹)]')
            plt.title('Gamma Distribution by Population')
        
        plt.ylabel('Count')
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    return cluster_labels, linkage_matrix

#simple threshold-based clustering (alternative to hierarchical)
def _simple_threshold_clustering(gammas_df, distance_threshold=0.3):
    #sort values
    sorted_vals = np.sort(gammas_df['log_clustering_val'].values)
    
    #find gaps
    cluster_labels = np.zeros(len(sorted_vals), dtype=int)
    current_cluster = 0
    
    for i in range(1, len(sorted_vals)):
        gap = sorted_vals[i] - sorted_vals[i-1]
        if gap > distance_threshold:
            current_cluster += 1
        cluster_labels[i] = current_cluster
    
    #map back to original order
    sort_indices = np.argsort(gammas_df['log_clustering_val'].values)
    reverse_indices = np.argsort(sort_indices)
    cluster_labels = cluster_labels[reverse_indices]
    
    return cluster_labels

#plot gamma distribution with cluster assignments
def _plot_population_distribution(gammas_df, stats_df, min_abundance, normalize_by_q2=False, experiment_name=''):
   
    if normalize_by_q2:
        # Special plot for D-based clustering: D vs q²
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        suptitle = 'Population Clustering'
        if experiment_name:
            suptitle += f' — {experiment_name}'
        fig.suptitle(suptitle, fontsize=12)
        
        # Plot 1: D vs q² (should show horizontal bands)
        ax1 = axes[0]
        for cluster_id in sorted(gammas_df['cluster'].unique()):
            cluster_data = gammas_df[gammas_df['cluster'] == cluster_id]
            abundance = stats_df[stats_df['cluster_id'] == cluster_id]['abundance'].values[0]
            
            marker = 'o' if abundance >= min_abundance else 'x'
            alpha = 0.8 if abundance >= min_abundance else 0.4
            
            ax1.scatter(cluster_data['q_squared'], cluster_data['D'],
                       marker=marker, alpha=alpha, s=50,
                       label=f'Pop {cluster_id + 1} ({abundance*100:.0f}%)')
        
        ax1.set_xlabel('q² [nm⁻²]')
        ax1.set_ylabel('D = Γ/q² [nm²/s]')
        ax1.set_title('D vs q² (should be horizontal for each population)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Plot 2: Log(D) histogram
        ax2 = axes[1]
        for cluster_id in sorted(gammas_df['cluster'].unique()):
            cluster_data = gammas_df[gammas_df['cluster'] == cluster_id]
            abundance = stats_df[stats_df['cluster_id'] == cluster_id]['abundance'].values[0]
            
            alpha = 0.7 if abundance >= min_abundance else 0.3
            
            ax2.hist(cluster_data['log_clustering_val'], alpha=alpha, bins=15,
                    label=f'Pop {cluster_id + 1}')
        
        ax2.set_xlabel('log₁₀(D) [log(nm²/s)]')
        ax2.set_ylabel('Count')
        ax2.set_title('D Distribution by Population')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Original Γ vs q² (for reference)
        ax3 = axes[2]
        for cluster_id in sorted(gammas_df['cluster'].unique()):
            cluster_data = gammas_df[gammas_df['cluster'] == cluster_id]
            abundance = stats_df[stats_df['cluster_id'] == cluster_id]['abundance'].values[0]
            
            marker = 'o' if abundance >= min_abundance else 'x'
            alpha = 0.8 if abundance >= min_abundance else 0.4
            
            ax3.scatter(cluster_data['q_squared'], cluster_data['gamma'],
                       marker=marker, alpha=alpha, s=50,
                       label=f'Pop {cluster_id + 1}')
        
        ax3.set_xlabel('q² [nm⁻²]')
        ax3.set_ylabel('Γ [s⁻¹]')
        ax3.set_title('Γ vs q² (grouped by D-based clustering)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
    else:
        #original plots for Γ-based clustering
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        suptitle = 'Population Clustering'
        if experiment_name:
            suptitle += f' — {experiment_name}'
        fig.suptitle(suptitle, fontsize=12)
    
        # Plot 1: Scatter plot in log-space
        ax1 = axes[0]
        for cluster_id in sorted(gammas_df['cluster'].unique()):
            cluster_data = gammas_df[gammas_df['cluster'] == cluster_id]
            abundance = stats_df[stats_df['cluster_id'] == cluster_id]['abundance'].values[0]
            
            marker = 'o' if abundance >= min_abundance else 'x'
            alpha = 0.8 if abundance >= min_abundance else 0.4
            
            ax1.scatter(cluster_data.index, cluster_data['log_clustering_val'],
                       marker=marker, alpha=alpha, s=50,
                       label=f'Pop {cluster_id + 1} ({abundance*100:.0f}%)')
        
        ax1.set_xlabel('Gamma index')
        ax1.set_ylabel('log₁₀(Γ) [log(s⁻¹)]')
        ax1.set_title('Cluster Assignments')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Histogram
        ax2 = axes[1]
        for cluster_id in sorted(gammas_df['cluster'].unique()):
            cluster_data = gammas_df[gammas_df['cluster'] == cluster_id]
            abundance = stats_df[stats_df['cluster_id'] == cluster_id]['abundance'].values[0]
            
            alpha = 0.7 if abundance >= min_abundance else 0.3
            
            ax2.hist(cluster_data['gamma'], alpha=alpha, bins=15,
                    label=f'Pop {cluster_id + 1}')
        
        ax2.set_xlabel('Γ [s⁻¹]')
        ax2.set_ylabel('Count')
        ax2.set_title('Gamma Distribution by Population')
        ax2.set_xscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

#show preview of what would happen if uncertain points are removed
def _show_uncertainty_removal_preview(gammas_df, n_uncertain, normalize_by_q2):
    print("\n" + "="*70)
    print("UNCERTAINTY REMOVAL PREVIEW:")
    print("What would happen if uncertain points are removed?\n")
    
    #current stats
    n_total = len(gammas_df)
    n_clean = n_total - n_uncertain
    avg_silhouette_current = gammas_df['silhouette'].mean()
    
    #preview stats (estimated)
    clean_data = gammas_df[~gammas_df['uncertain']]
    avg_silhouette_clean = clean_data['silhouette'].mean()
    
    print("Current (with all data):")
    print(f"  - Total points: {n_total}")
    print(f"  - Average silhouette: {avg_silhouette_current:.3f}")
    
    #show per-population breakdown
    for cluster_id in sorted(gammas_df['cluster'].unique()):
        cluster_data = gammas_df[gammas_df['cluster'] == cluster_id]
        n_in_cluster = len(cluster_data)
        print(f"  - Population {cluster_id + 1}: {n_in_cluster} points")
    
    print("\nAfter removal (estimated):")
    print(f"  - Total points: {n_clean} ({n_uncertain} removed)")
    print(f"  - Average silhouette: {avg_silhouette_clean:.3f}", end="")
    
    improvement = ((avg_silhouette_clean - avg_silhouette_current) / avg_silhouette_current * 100)
    if improvement > 0:
        print(f" ← +{improvement:.1f}% improvement")
    else:
        print()
    
    #show per-population after removal
    for cluster_id in sorted(clean_data['cluster'].unique()):
        cluster_clean = clean_data[clean_data['cluster'] == cluster_id]
        cluster_all = gammas_df[gammas_df['cluster'] == cluster_id]
        n_removed = len(cluster_all) - len(cluster_clean)
        print(f"  - Population {cluster_id + 1}: {len(cluster_clean)} points ({n_removed} removed)")

#calculate silhouette score for each point
def _calculate_silhouette_scores(gammas_df, cluster_labels):
    """
    silhouette measures how well each point fits in its assigned cluster:
    - Score near +1: Point is well-clustered
    - Score near 0: Point is on border between clusters
    - Score near -1: Point is probably in wrong cluster
    """
    from sklearn.metrics import silhouette_samples
    
    #use log_clustering_val for distance calculation
    X = gammas_df['log_clustering_val'].values.reshape(-1, 1)
    
    #calculate silhouette for each sample
    try:
        silhouette_vals = silhouette_samples(X, cluster_labels)
        return silhouette_vals
    except:
        #if calculation fails (e.g., only 1 cluster), return zeros
        return np.zeros(len(gammas_df))

#try to reassign points with low silhouette scores to better clusters
def _refine_clusters_with_silhouette(gammas_df, threshold, normalize_by_q2):
    n_reassigned = 0
    
    #get unique cluster IDs
    cluster_ids = sorted(gammas_df['cluster'].unique())
    
    if len(cluster_ids) < 2:
        #can't reassign if only 1 cluster
        return 0
    
    #find points with low silhouette
    low_silhouette_mask = gammas_df['silhouette'] < threshold
    low_silhouette_points = gammas_df[low_silhouette_mask]
    
    for idx in low_silhouette_points.index:
        current_cluster = gammas_df.loc[idx, 'cluster']
        current_val = gammas_df.loc[idx, 'log_clustering_val']
        
        #calculate distance to each cluster center
        best_cluster = current_cluster
        best_distance = float('inf')
        
        for cluster_id in cluster_ids:
            #get mean of cluster
            cluster_points = gammas_df[gammas_df['cluster'] == cluster_id]
            cluster_mean = cluster_points['log_clustering_val'].mean()
            
            distance = abs(current_val - cluster_mean)
            
            if distance < best_distance:
                best_distance = distance
                best_cluster = cluster_id
        
        #reassign if found a better cluster
        if best_cluster != current_cluster:
            gammas_df.loc[idx, 'cluster'] = best_cluster
            n_reassigned += 1
    
    return n_reassigned


def get_reliable_gamma_cols(cluster_info, prefix='gamma_pop'):
    """
    Helper function to get list of gamma column names for analysis.
    """
    #check if clustering was enabled
    if not cluster_info.get('clustering_enabled', True):
        #pass-through mode: return original column names
        return cluster_info.get('original_gamma_cols', [])
    else:
        #clustering mode: return gamma_pop columns
        return [f'{prefix}{i}' for i in cluster_info['reliable_populations']]


#alias for better clarity (both names work)
get_gamma_cols_for_analysis = get_reliable_gamma_cols


def aggregate_peak_stats(cluster_info, data_df, stat_prefixes=('skewness', 'kurtosis')):
    """
    Aggregate per-peak statistics (skewness, kurtosis) per population after clustering.
    Only activates when per-peak stat columns (e.g. skewness_1) exist in data_df.
    """
    #condition: at least one stat column must exist in data_df
    if not any(f'{prefix}_1' in data_df.columns for prefix in stat_prefixes):
        return cluster_info

    #pass-through mode: gamma_1 → skewness_1 directly, average per column
    if not cluster_info.get('clustering_enabled', False):
        gamma_cols = cluster_info.get('original_gamma_cols', [])
        for prefix in stat_prefixes:
            if f'{prefix}_1' not in data_df.columns:
                continue
            pop_means = {}
            pop_stds  = {}
            for i, col in enumerate(gamma_cols):
                try:
                    peak_idx = int(col.split('_')[-1])
                except (ValueError, IndexError):
                    continue
                stat_col = f'{prefix}_{peak_idx}'
                if stat_col in data_df.columns:
                    vals = data_df[stat_col].dropna()
                    pop_means[i + 1] = vals.mean() if len(vals) > 0 else np.nan
                    pop_stds[i + 1]  = vals.std()  if len(vals) > 1 else np.nan
            cluster_info[f'population_{prefix}_mean'] = pop_means
            cluster_info[f'population_{prefix}_std']  = pop_stds

            print(f"\nPer-population {prefix} (pass-through):")
            for pop_id, mean_val in pop_means.items():
                std_val = pop_stds[pop_id]
                if np.isnan(mean_val):
                    print(f"  Population {pop_id}: insufficient data")
                else:
                    std_str = f" ± {std_val:.3f}" if not np.isnan(std_val) else ""
                    print(f"  Population {pop_id}: {mean_val:.3f}{std_str}")
        return cluster_info

    if 'gammas_df' not in cluster_info:
        return cluster_info

    gammas_df = cluster_info['gammas_df']
    
    cluster_id_to_pop = cluster_info.get('cluster_id_to_pop', None)
    if cluster_id_to_pop is None:
        reliable_ids = sorted(gammas_df['cluster'].unique())
        cluster_id_to_pop = {cid: i + 1 for i, cid in enumerate(reliable_ids)}

    #for each stat prefix, collect values per population
    for prefix in stat_prefixes:
        #check this specific stat exists
        if f'{prefix}_1' not in data_df.columns:
            continue

        pop_means = {}
        pop_stds  = {}

        for cluster_id in sorted(gammas_df['cluster'].unique()):
            cluster_rows = gammas_df[gammas_df['cluster'] == cluster_id]
            values = []

            for _, row in cluster_rows.iterrows():
                file_idx    = row['file_idx']
                original_col = row['original_col']  # e.g. 'gamma_1', 'gamma_2'

                # Derive peak index from original gamma column name
                try:
                    peak_idx = int(original_col.split('_')[-1])
                except (ValueError, IndexError):
                    continue

                stat_col = f'{prefix}_{peak_idx}'
                if stat_col in data_df.columns and file_idx in data_df.index:
                    val = data_df.loc[file_idx, stat_col]
                    if pd.notna(val):
                        values.append(val)

                pop_num            = cluster_id_to_pop.get(cluster_id, cluster_id + 1)
                pop_means[pop_num] = np.mean(values) if values else np.nan
                pop_stds[pop_num]  = np.std(values)  if len(values) > 1 else np.nan

        cluster_info[f'population_{prefix}_mean'] = pop_means
        cluster_info[f'population_{prefix}_std']  = pop_stds

        #print summary
        print(f"\nPer-population {prefix}:")
        for pop_id, mean_val in pop_means.items():
            std_val = pop_stds[pop_id]
            if np.isnan(mean_val):
                print(f"  Population {pop_id}: insufficient data")
            else:
                std_str = f" ± {std_val:.3f}" if not np.isnan(std_val) else ""
                print(f"  Population {pop_id}: {mean_val:.3f}{std_str}")

    return cluster_info
