"""
Alpha Analysis Dialog for Regularized NNLS
Interactive 3D visualization to find optimal alpha parameter
"""

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGroupBox,
                             QLabel, QPushButton, QSpinBox, QDoubleSpinBox,
                             QSlider, QWidget, QMessageBox, QSizePolicy,
                             QCheckBox, QTextEdit)
from PyQt5.QtCore import Qt
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class AlphaAnalysisDialog(QDialog):
    """
    Dialog for Alpha parameter analysis

    Shows 3D visualization of how distributions change with different alpha values
    Helps user find optimal regularization parameter
    """

    def __init__(self, laplace_analyzer, parent=None):
        super().__init__(parent)
        self.laplace_analyzer = laplace_analyzer

        # Default parameters
        self.alpha_min = 0.001
        self.alpha_max = 10.0
        self.num_alphas = 5
        self.num_datasets = 3
        self.recommended_alpha = None

        self.init_ui()
        self.setWindowTitle("Alpha Parameter Analysis")
        self.resize(1200, 900)

    def init_ui(self):
        """Initialize the UI"""
        layout = QVBoxLayout()

        # Title
        title = QLabel("Alpha Parameter Analysis for Regularized NNLS")
        title.setStyleSheet("font-size: 14pt; font-weight: bold;")
        layout.addWidget(title)

        info = QLabel(
            "Alpha controls the strength of regularization:\n"
            "• Low alpha (< 0.1): Minimal smoothing, more peaks (may be noisy)\n"
            "• Medium alpha (0.1 - 1): Balanced smoothing\n"
            "• High alpha (> 1): Strong smoothing, fewer peaks (may oversimplify)"
        )
        info.setStyleSheet("color: #666; font-style: italic; padding: 10px; background-color: #f0f0f0; border-radius: 5px;")
        info.setWordWrap(True)
        layout.addWidget(info)

        # Parameters group
        params_group = self.create_parameters_group()
        layout.addWidget(params_group)

        # Run button
        run_layout = QHBoxLayout()
        self.run_btn = QPushButton("🔬 Run Alpha Analysis")
        self.run_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-weight: bold;
                padding: 10px 20px;
                border-radius: 5px;
                font-size: 12pt;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        self.run_btn.clicked.connect(self.run_analysis)
        run_layout.addStretch()
        run_layout.addWidget(self.run_btn)
        run_layout.addStretch()
        layout.addLayout(run_layout)

        # 3D Plot canvas
        self.canvas = FigureCanvas(Figure(figsize=(12, 8)))
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        # Recommendation text
        self.recommendation_group = QGroupBox("Analysis Results & Recommendation")
        rec_layout = QVBoxLayout()
        self.recommendation_text = QTextEdit()
        self.recommendation_text.setReadOnly(True)
        self.recommendation_text.setMaximumHeight(150)
        self.recommendation_text.setStyleSheet("background-color: #FFFACD; padding: 10px;")
        rec_layout.addWidget(self.recommendation_text)
        self.recommendation_group.setLayout(rec_layout)
        self.recommendation_group.setVisible(False)
        layout.addWidget(self.recommendation_group)

        # Bottom buttons
        button_layout = QHBoxLayout()

        self.use_recommended_btn = QPushButton("✓ Use Recommended Alpha")
        self.use_recommended_btn.setEnabled(False)
        self.use_recommended_btn.clicked.connect(self.accept)
        button_layout.addWidget(self.use_recommended_btn)

        button_layout.addStretch()

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        button_layout.addWidget(close_btn)

        layout.addLayout(button_layout)

        self.setLayout(layout)

    def create_parameters_group(self):
        """Create parameters configuration group"""
        group = QGroupBox("Analysis Parameters")
        layout = QVBoxLayout()

        # Alpha range
        alpha_layout = QHBoxLayout()
        alpha_layout.addWidget(QLabel("Alpha Range:"))

        alpha_layout.addWidget(QLabel("Min:"))
        self.alpha_min_spin = QDoubleSpinBox()
        self.alpha_min_spin.setDecimals(4)
        self.alpha_min_spin.setRange(0.0001, 1.0)
        self.alpha_min_spin.setValue(0.001)
        self.alpha_min_spin.setSingleStep(0.001)
        alpha_layout.addWidget(self.alpha_min_spin)

        alpha_layout.addWidget(QLabel("Max:"))
        self.alpha_max_spin = QDoubleSpinBox()
        self.alpha_max_spin.setDecimals(2)
        self.alpha_max_spin.setRange(0.1, 100.0)
        self.alpha_max_spin.setValue(10.0)
        self.alpha_max_spin.setSingleStep(1.0)
        alpha_layout.addWidget(self.alpha_max_spin)

        alpha_layout.addStretch()
        layout.addLayout(alpha_layout)

        # Number of alpha values
        num_alpha_layout = QHBoxLayout()
        num_alpha_layout.addWidget(QLabel("Number of alpha values to test:"))
        self.num_alphas_spin = QSpinBox()
        self.num_alphas_spin.setRange(3, 10)
        self.num_alphas_spin.setValue(5)
        num_alpha_layout.addWidget(self.num_alphas_spin)
        num_alpha_layout.addWidget(QLabel("(logarithmically spaced)"))
        num_alpha_layout.addStretch()
        layout.addLayout(num_alpha_layout)

        # Number of datasets
        num_data_layout = QHBoxLayout()
        num_data_layout.addWidget(QLabel("Random datasets to analyze:"))
        self.num_datasets_spin = QSpinBox()
        self.num_datasets_spin.setRange(1, 10)
        self.num_datasets_spin.setValue(3)
        num_data_layout.addWidget(self.num_datasets_spin)
        num_data_layout.addWidget(QLabel("(for speed)"))
        num_data_layout.addStretch()
        layout.addLayout(num_data_layout)

        # Options
        self.show_peaks_check = QCheckBox("Show detected peaks")
        self.show_peaks_check.setChecked(True)
        layout.addWidget(self.show_peaks_check)

        # Two-stage optimization option
        self.two_stage_check = QCheckBox("Use two-stage optimization (more accurate)")
        self.two_stage_check.setChecked(True)
        self.two_stage_check.setToolTip(
            "Stage 1: Coarse search with few alpha values\n"
            "Stage 2: Fine search around best alpha from Stage 1 (more accurate)"
        )
        layout.addWidget(self.two_stage_check)

        # Stage 2 parameters (only used if two-stage is enabled)
        stage2_layout = QHBoxLayout()
        stage2_layout.addWidget(QLabel("Stage 2 refinement:"))
        self.stage2_num_spin = QSpinBox()
        self.stage2_num_spin.setRange(5, 20)
        self.stage2_num_spin.setValue(10)
        self.stage2_num_spin.setToolTip("Number of alpha values in refinement stage")
        stage2_layout.addWidget(self.stage2_num_spin)
        stage2_layout.addWidget(QLabel("values"))
        stage2_layout.addStretch()
        layout.addLayout(stage2_layout)

        group.setLayout(layout)
        return group

    def run_analysis(self):
        """Run alpha parameter analysis with optional two-stage optimization"""
        try:
            # Get parameters
            self.alpha_min = self.alpha_min_spin.value()
            self.alpha_max = self.alpha_max_spin.value()
            self.num_alphas = self.num_alphas_spin.value()
            self.num_datasets = self.num_datasets_spin.value()
            use_two_stage = self.two_stage_check.isChecked()
            stage2_num = self.stage2_num_spin.value()

            if self.alpha_min >= self.alpha_max:
                QMessageBox.warning(self, "Invalid Range",
                                   "Alpha min must be less than alpha max")
                return

            # Update status
            self.run_btn.setEnabled(False)
            self.run_btn.setText("⏳ Running analysis...")

            # ===== STAGE 1: Coarse search =====
            if use_two_stage:
                print(f"[Alpha Analysis] Two-stage optimization enabled")
                print(f"[Alpha Analysis] Stage 1: Coarse search with {self.num_alphas} alpha values")
            else:
                print(f"[Alpha Analysis] Single-stage optimization with {self.num_alphas} alpha values")

            # Create alpha values for Stage 1 (logarithmic spacing)
            alphas_stage1 = np.logspace(np.log10(self.alpha_min),
                                        np.log10(self.alpha_max),
                                        self.num_alphas)

            print(f"[Alpha Analysis] Testing {len(alphas_stage1)} alpha values: {alphas_stage1}")

            # Get random datasets
            import random
            all_keys = list(self.laplace_analyzer.processed_correlations.keys())
            num_to_select = min(self.num_datasets, len(all_keys))
            selected_keys = random.sample(all_keys, num_to_select)

            print(f"[Alpha Analysis] Selected {len(selected_keys)} datasets: {selected_keys}")

            # Run regularized fits for all combinations
            from regularized_optimized import regularized_nnls_optimized, create_exponential_matrix

            # Create decay times (shared across all)
            decay_times = np.logspace(-8, 1, 200)

            # Pre-compute T matrices for each dataset (major speedup!)
            print("[Alpha Analysis] Pre-computing T matrices...")
            T_matrices = {}
            for key in selected_keys:
                df = self.laplace_analyzer.processed_correlations[key]
                tau = df['t (s)'].to_numpy()
                T_matrices[key] = create_exponential_matrix(tau, decay_times)
            print("[Alpha Analysis] T matrices cached")

            # Stage 1: Coarse search
            results_stage1 = self._run_alpha_fits(alphas_stage1, selected_keys, T_matrices,
                                                   decay_times, stage_name="Stage 1")

            # Analyze Stage 1 results
            best_alpha_stage1 = self._analyze_results(results_stage1, alphas_stage1, selected_keys)
            print(f"[Alpha Analysis] Stage 1 best alpha: {best_alpha_stage1:.4f}")

            # ===== STAGE 2: Fine search (if enabled) =====
            if use_two_stage:
                self.run_btn.setText("⏳ Running Stage 2...")

                # Define refined search range around best alpha from Stage 1
                # Use +/- 0.5 log units (about 3x range on each side)
                log_center = np.log10(best_alpha_stage1)
                log_range = 0.5  # Half a decade on each side
                alpha_min_stage2 = max(10**(log_center - log_range), self.alpha_min)
                alpha_max_stage2 = min(10**(log_center + log_range), self.alpha_max)

                print(f"[Alpha Analysis] Stage 2: Fine search with {stage2_num} values")
                print(f"[Alpha Analysis] Stage 2 range: {alpha_min_stage2:.4f} - {alpha_max_stage2:.4f}")

                alphas_stage2 = np.logspace(np.log10(alpha_min_stage2),
                                            np.log10(alpha_max_stage2),
                                            stage2_num)

                results_stage2 = self._run_alpha_fits(alphas_stage2, selected_keys, T_matrices,
                                                       decay_times, stage_name="Stage 2")

                # Combine results from both stages for visualization
                alphas_combined = np.concatenate([alphas_stage1, alphas_stage2])
                results_combined = {}
                for key in selected_keys:
                    results_combined[key] = {**results_stage1[key], **results_stage2[key]}

                # Analyze combined results for final recommendation
                self.recommended_alpha = self._analyze_results(results_combined, alphas_combined, selected_keys)

                # Use combined results for plotting
                alphas_final = alphas_combined
                results_final = results_combined

                print(f"[Alpha Analysis] Final recommended alpha: {self.recommended_alpha:.4f}")
            else:
                # Single-stage: use Stage 1 results
                alphas_final = alphas_stage1
                results_final = results_stage1
                self.recommended_alpha = best_alpha_stage1

            # Create 3D visualization
            self._create_3d_plot(results_final, alphas_final, selected_keys, decay_times)

            # Show recommendation
            self._show_recommendation(results_final, alphas_final, selected_keys)

            # Enable accept button
            self.use_recommended_btn.setEnabled(True)

            # Re-enable run button
            self.run_btn.setEnabled(True)
            self.run_btn.setText("🔬 Run Alpha Analysis")

        except Exception as e:
            QMessageBox.critical(self, "Analysis Error",
                               f"Error during alpha analysis:\n\n{str(e)}")
            import traceback
            traceback.print_exc()

            self.run_btn.setEnabled(True)
            self.run_btn.setText("🔬 Run Alpha Analysis")

    def _run_alpha_fits(self, alphas, selected_keys, T_matrices, decay_times, stage_name=""):
        """
        Run regularized NNLS fits for given alpha values

        Args:
            alphas: Array of alpha values to test
            selected_keys: List of dataset keys
            T_matrices: Pre-computed T matrices for each dataset
            decay_times: Decay time grid
            stage_name: Name of the stage (for logging)

        Returns:
            Dictionary with results for each dataset and alpha
        """
        from regularized_optimized import regularized_nnls_optimized

        results = {}
        total_fits = len(selected_keys) * len(alphas)
        completed = 0

        for key in selected_keys:
            df = self.laplace_analyzer.processed_correlations[key]
            results[key] = {}
            T_matrix = T_matrices[key]  # Use pre-computed matrix

            for alpha in alphas:
                # Create parameters dict for regularized_nnls_optimized
                params = {
                    'decay_times': decay_times,
                    'prominence': 0.05,
                    'distance': 1,
                    'alpha': alpha
                }

                # Run optimized regularized fit with cached T matrix
                try:
                    _, f_optimized, _, _, peaks = regularized_nnls_optimized(
                        df, key, params, plot_number=1, T_matrix=T_matrix
                    )
                    results[key][alpha] = {
                        'distribution': f_optimized,
                        'num_peaks': len(peaks),
                        'peak_indices': peaks
                    }
                except Exception as e:
                    print(f"[Alpha Analysis] {stage_name} Error for {key}, alpha={alpha}: {e}")
                    results[key][alpha] = {
                        'distribution': np.zeros_like(decay_times),
                        'num_peaks': 0,
                        'peak_indices': np.array([])
                    }

                completed += 1
                if completed % 5 == 0 or completed == total_fits:
                    print(f"[Alpha Analysis] {stage_name} Progress: {completed}/{total_fits} fits completed")

        return results

    def _analyze_results(self, results, alphas, selected_keys):
        """
        Analyze results and recommend optimal alpha

        Strategy: Find alpha where peak count becomes stable
        """
        # Count peaks for each alpha (averaged across datasets)
        peak_counts = []
        for alpha in alphas:
            counts = [results[key][alpha]['num_peaks'] for key in selected_keys]
            avg_count = np.mean(counts)
            peak_counts.append(avg_count)

        # Find where peak count stabilizes (derivative is smallest)
        if len(peak_counts) > 2:
            derivatives = np.abs(np.diff(peak_counts))
            stable_idx = np.argmin(derivatives) + 1  # +1 because diff reduces array by 1

            # Prefer slightly higher alpha for stability
            recommended_idx = min(stable_idx + 1, len(alphas) - 1)
        else:
            # If only few alphas, pick middle one
            recommended_idx = len(alphas) // 2

        recommended = alphas[recommended_idx]
        print(f"[Alpha Analysis] Recommended alpha: {recommended:.4f} (index {recommended_idx})")

        return recommended

    def _create_3d_plot(self, results, alphas, selected_keys, decay_times):
        """Create 3D visualization of distributions vs alpha"""
        self.canvas.figure.clear()

        # Create 3D plot for each dataset
        num_datasets = len(selected_keys)
        if num_datasets == 1:
            ax = self.canvas.figure.add_subplot(111, projection='3d')
            axes = [ax]
        elif num_datasets == 2:
            ax1 = self.canvas.figure.add_subplot(121, projection='3d')
            ax2 = self.canvas.figure.add_subplot(122, projection='3d')
            axes = [ax1, ax2]
        else:
            # Create grid for multiple datasets
            cols = min(3, num_datasets)
            rows = (num_datasets + cols - 1) // cols
            axes = []
            for i in range(num_datasets):
                ax = self.canvas.figure.add_subplot(rows, cols, i+1, projection='3d')
                axes.append(ax)

        # Plot each dataset
        for idx, key in enumerate(selected_keys):
            ax = axes[idx]

            # Prepare data for 3D surface
            X, Y = np.meshgrid(np.log10(decay_times), np.log10(alphas))
            Z = np.zeros_like(X)

            for i, alpha in enumerate(alphas):
                distribution = results[key][alpha]['distribution']
                Z[i, :] = distribution

            # Plot surface
            surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8,
                                  linewidth=0, antialiased=True)

            # Mark recommended alpha
            if self.recommended_alpha is not None:
                rec_idx = np.argmin(np.abs(alphas - self.recommended_alpha))
                rec_dist = results[key][alphas[rec_idx]]['distribution']
                ax.plot(np.log10(decay_times),
                       np.full_like(decay_times, np.log10(self.recommended_alpha)),
                       rec_dist, 'r-', linewidth=3, label=f'α={self.recommended_alpha:.3f}')

            ax.set_xlabel('log₁₀(Decay Time [s])')
            ax.set_ylabel('log₁₀(Alpha)')
            ax.set_zlabel('Intensity')
            ax.set_title(f'{key}')
            ax.legend()

            # Add colorbar (only for first plot)
            if idx == 0:
                self.canvas.figure.colorbar(surf, ax=ax, shrink=0.5)

        self.canvas.figure.suptitle(
            f'Distribution vs Alpha Parameter\n'
            f'Recommended α = {self.recommended_alpha:.4f}',
            fontsize=14, fontweight='bold'
        )

        self.canvas.figure.tight_layout()
        self.canvas.draw()

    def _show_recommendation(self, results, alphas, selected_keys):
        """Show recommendation text"""
        # Calculate statistics
        peak_counts = {}
        for alpha in alphas:
            counts = [results[key][alpha]['num_peaks'] for key in selected_keys]
            peak_counts[alpha] = {
                'mean': np.mean(counts),
                'std': np.std(counts),
                'counts': counts
            }

        # Generate recommendation text
        text = f"<h3>Recommended Alpha: {self.recommended_alpha:.4f}</h3>\n\n"

        text += "<b>Analysis Summary:</b><br>"
        text += f"• Tested {len(alphas)} alpha values from {alphas[0]:.4f} to {alphas[-1]:.4f}<br>"
        text += f"• Analyzed {len(selected_keys)} random datasets<br><br>"

        text += "<b>Peak Count Stability:</b><br>"
        text += "<table border='1' cellpadding='5' style='border-collapse: collapse;'>"
        text += "<tr><th>Alpha</th><th>Avg Peaks</th><th>Std Dev</th></tr>"

        for alpha in alphas:
            stats = peak_counts[alpha]
            # Highlight recommended
            if np.isclose(alpha, self.recommended_alpha, rtol=0.01):
                text += f"<tr style='background-color: #90EE90;'>"
            else:
                text += "<tr>"
            text += f"<td>{alpha:.4f}</td>"
            text += f"<td>{stats['mean']:.1f}</td>"
            text += f"<td>{stats['std']:.2f}</td>"
            text += "</tr>"

        text += "</table><br>"

        text += "<b>Recommendation Rationale:</b><br>"
        rec_stats = peak_counts[self.recommended_alpha]
        text += f"• This alpha value shows stable peak detection across datasets<br>"
        text += f"• Average of {rec_stats['mean']:.1f} peaks detected<br>"
        text += f"• Low variation (σ = {rec_stats['std']:.2f}) indicates consistent results<br>"

        self.recommendation_text.setHtml(text)
        self.recommendation_group.setVisible(True)

    def get_recommended_alpha(self):
        """Get the recommended alpha value"""
        return self.recommended_alpha if self.recommended_alpha is not None else 1.0
