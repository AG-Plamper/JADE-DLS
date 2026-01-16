"""
NNLS Parameter Dialog with Interactive Preview
Allows users to tune peak detection parameters with live feedback
"""

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGroupBox,
                             QLabel, QLineEdit, QPushButton, QCheckBox,
                             QSlider, QSpinBox, QDoubleSpinBox, QTabWidget,
                             QWidget, QMessageBox, QSizePolicy, QScrollArea)
from PyQt5.QtCore import Qt, pyqtSignal
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


class NNLSDialog(QDialog):
    """
    Dialog for NNLS parameter configuration with live preview

    Features:
        - Configure decay time range
        - Tune peak detection parameters (prominence, distance)
        - Live preview on random datasets
        - Interactive parameter adjustment
    """

    parameters_accepted = pyqtSignal(dict)

    def __init__(self, laplace_analyzer, parent=None):
        super().__init__(parent)
        self.laplace_analyzer = laplace_analyzer

        # Default parameters
        self.params = {
            'decay_times': np.logspace(-8, 1, 200),
            'prominence': 0.05,
            'distance': 1,
            'num_preview': 5,
            'eps_factor': 0.3,  # Clustering parameter for automatic mode detection
            'use_clustering': True  # Enable automatic peak clustering
        }

        # Storage for preview
        self.preview_figure = None
        self.preview_datasets = []

        self.init_ui()
        self.setWindowTitle("NNLS Configuration")
        self.resize(1200, 800)

    def init_ui(self):
        """Initialize the UI"""
        layout = QVBoxLayout()

        # Title
        title = QLabel("NNLS (Non-Negative Least Squares) Configuration")
        title.setStyleSheet("font-size: 14pt; font-weight: bold;")
        layout.addWidget(title)

        # Tabs for different sections
        tabs = QTabWidget()

        # Tab 1: Basic Parameters
        basic_tab = self.create_basic_parameters_tab()
        tabs.addTab(basic_tab, "Basic Parameters")

        # Tab 2: Peak Detection
        peak_tab = self.create_peak_detection_tab()
        tabs.addTab(peak_tab, "Peak Detection")

        # Tab 3: Preview
        preview_tab = self.create_preview_tab()
        tabs.addTab(preview_tab, "Interactive Preview")

        layout.addWidget(tabs)

        # Bottom buttons
        button_layout = QHBoxLayout()

        self.run_btn = QPushButton("Run NNLS Analysis")
        self.run_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.run_btn.clicked.connect(self.accept_parameters)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)

        button_layout.addStretch()
        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(self.run_btn)

        layout.addLayout(button_layout)

        self.setLayout(layout)

    def create_basic_parameters_tab(self):
        """Create basic parameters tab"""
        widget = QWidget()
        layout = QVBoxLayout()

        # Decay Times Group
        decay_group = QGroupBox("Decay Time Range")
        decay_layout = QVBoxLayout()

        # Start
        start_layout = QHBoxLayout()
        start_layout.addWidget(QLabel("Start (seconds):"))
        self.decay_start_spin = QDoubleSpinBox()
        self.decay_start_spin.setDecimals(10)
        self.decay_start_spin.setRange(1e-10, 1)
        self.decay_start_spin.setValue(1e-8)
        self.decay_start_spin.setSingleStep(1e-9)
        self.decay_start_spin.setStyleSheet("QDoubleSpinBox { min-width: 150px; }")
        start_layout.addWidget(self.decay_start_spin)
        start_layout.addWidget(QLabel("(e.g., 1e-8)"))
        start_layout.addStretch()
        decay_layout.addLayout(start_layout)

        # End
        end_layout = QHBoxLayout()
        end_layout.addWidget(QLabel("End (seconds):"))
        self.decay_end_spin = QDoubleSpinBox()
        self.decay_end_spin.setDecimals(2)
        self.decay_end_spin.setRange(1e-6, 100)
        self.decay_end_spin.setValue(10)
        self.decay_end_spin.setSingleStep(1)
        self.decay_end_spin.setStyleSheet("QDoubleSpinBox { min-width: 150px; }")
        end_layout.addWidget(self.decay_end_spin)
        end_layout.addWidget(QLabel("(e.g., 10)"))
        end_layout.addStretch()
        decay_layout.addLayout(end_layout)

        # Number of points
        num_layout = QHBoxLayout()
        num_layout.addWidget(QLabel("Number of points:"))
        self.decay_num_spin = QSpinBox()
        self.decay_num_spin.setRange(50, 1000)
        self.decay_num_spin.setValue(200)
        self.decay_num_spin.setSingleStep(10)
        num_layout.addWidget(self.decay_num_spin)
        num_layout.addWidget(QLabel("(logarithmic spacing)"))
        num_layout.addStretch()
        decay_layout.addLayout(num_layout)

        # Info
        info_label = QLabel("💡 Tip: The decay time range should cover the expected relaxation times.\n"
                           "   Typical DLS measurements: 1e-8 to 10 seconds")
        info_label.setStyleSheet("color: #666; font-style: italic; padding: 10px;")
        decay_layout.addWidget(info_label)

        decay_group.setLayout(decay_layout)
        layout.addWidget(decay_group)

        # Processing Options
        options_group = QGroupBox("Processing Options")
        options_layout = QVBoxLayout()

        self.multiprocessing_check = QCheckBox("Use multiprocessing (faster for large datasets)")
        self.multiprocessing_check.setChecked(False)
        self.multiprocessing_check.setToolTip("Enable parallel processing for faster analysis (uses joblib for cross-platform support)")
        options_layout.addWidget(self.multiprocessing_check)

        self.show_plots_check = QCheckBox("Show individual plots during processing")
        self.show_plots_check.setChecked(True)
        options_layout.addWidget(self.show_plots_check)

        options_group.setLayout(options_layout)
        layout.addWidget(options_group)

        layout.addStretch()
        widget.setLayout(layout)
        return widget

    def create_peak_detection_tab(self):
        """Create peak detection parameters tab with interactive sliders"""
        widget = QWidget()
        layout = QVBoxLayout()

        # Prominence
        prom_group = QGroupBox("Peak Prominence")
        prom_layout = QVBoxLayout()

        prom_info = QLabel("Prominence: Minimum height of peaks relative to baseline.\n"
                          "Lower values = more sensitive (finds smaller peaks)")
        prom_info.setWordWrap(True)
        prom_info.setStyleSheet("color: #666; font-style: italic;")
        prom_layout.addWidget(prom_info)

        prom_slider_layout = QHBoxLayout()
        prom_slider_layout.addWidget(QLabel("Prominence:"))

        self.prominence_slider = QSlider(Qt.Horizontal)
        self.prominence_slider.setRange(1, 500)  # 0.001 to 0.5 (scaled by 1000)
        self.prominence_slider.setValue(50)  # 0.05
        self.prominence_slider.setTickPosition(QSlider.TicksBelow)
        self.prominence_slider.setTickInterval(50)
        self.prominence_slider.valueChanged.connect(self.on_prominence_slider_changed)
        prom_slider_layout.addWidget(self.prominence_slider)

        # Manual input field
        self.prominence_input = QDoubleSpinBox()
        self.prominence_input.setDecimals(3)
        self.prominence_input.setRange(0.001, 0.5)
        self.prominence_input.setValue(0.05)
        self.prominence_input.setSingleStep(0.001)
        self.prominence_input.setMinimumWidth(80)
        self.prominence_input.valueChanged.connect(self.on_prominence_input_changed)
        prom_slider_layout.addWidget(self.prominence_input)

        prom_layout.addLayout(prom_slider_layout)
        prom_group.setLayout(prom_layout)
        layout.addWidget(prom_group)

        # Distance
        dist_group = QGroupBox("Peak Distance")
        dist_layout = QVBoxLayout()

        dist_info = QLabel("Distance: Minimum separation between peaks (in index points).\n"
                          "Higher values = fewer peaks (forces peaks to be further apart)")
        dist_info.setWordWrap(True)
        dist_info.setStyleSheet("color: #666; font-style: italic;")
        dist_layout.addWidget(dist_info)

        dist_slider_layout = QHBoxLayout()
        dist_slider_layout.addWidget(QLabel("Distance:"))

        self.distance_slider = QSlider(Qt.Horizontal)
        self.distance_slider.setRange(1, 50)
        self.distance_slider.setValue(1)
        self.distance_slider.setTickPosition(QSlider.TicksBelow)
        self.distance_slider.setTickInterval(5)
        self.distance_slider.valueChanged.connect(self.on_distance_slider_changed)
        dist_slider_layout.addWidget(self.distance_slider)

        # Manual input field
        self.distance_input = QSpinBox()
        self.distance_input.setRange(1, 50)
        self.distance_input.setValue(1)
        self.distance_input.setSingleStep(1)
        self.distance_input.setMinimumWidth(80)
        self.distance_input.valueChanged.connect(self.on_distance_input_changed)
        dist_slider_layout.addWidget(self.distance_input)

        dist_layout.addLayout(dist_slider_layout)
        dist_group.setLayout(dist_layout)
        layout.addWidget(dist_group)

        # Clustering Parameters
        cluster_group = QGroupBox("Automatic Mode Clustering")
        cluster_layout = QVBoxLayout()

        cluster_info = QLabel("Clustering groups similar peaks across different angles into modes.\n"
                             "Lower eps_factor = stricter clustering (more modes)\n"
                             "Higher eps_factor = looser clustering (fewer modes)")
        cluster_info.setWordWrap(True)
        cluster_info.setStyleSheet("color: #666; font-style: italic;")
        cluster_layout.addWidget(cluster_info)

        # Enable clustering checkbox
        self.clustering_enabled_check = QCheckBox("Enable automatic peak clustering")
        self.clustering_enabled_check.setChecked(True)
        self.clustering_enabled_check.setToolTip("Groups peaks with similar diffusion coefficients across angles")
        cluster_layout.addWidget(self.clustering_enabled_check)

        # eps_factor slider
        eps_slider_layout = QHBoxLayout()
        eps_slider_layout.addWidget(QLabel("eps_factor:"))

        self.eps_factor_slider = QSlider(Qt.Horizontal)
        self.eps_factor_slider.setRange(5, 100)  # 0.05 to 1.0 (scaled by 100)
        self.eps_factor_slider.setValue(30)  # 0.3
        self.eps_factor_slider.setTickPosition(QSlider.TicksBelow)
        self.eps_factor_slider.setTickInterval(10)
        self.eps_factor_slider.valueChanged.connect(self.on_eps_factor_slider_changed)
        eps_slider_layout.addWidget(self.eps_factor_slider)

        # Manual input field
        self.eps_factor_input = QDoubleSpinBox()
        self.eps_factor_input.setDecimals(2)
        self.eps_factor_input.setRange(0.05, 1.0)
        self.eps_factor_input.setValue(0.3)
        self.eps_factor_input.setSingleStep(0.05)
        self.eps_factor_input.setMinimumWidth(80)
        self.eps_factor_input.valueChanged.connect(self.on_eps_factor_input_changed)
        eps_slider_layout.addWidget(self.eps_factor_input)

        cluster_layout.addLayout(eps_slider_layout)

        # Mode detection info label
        self.detected_modes_label = QLabel("Detected modes: N/A (run preview to see)")
        self.detected_modes_label.setStyleSheet("font-weight: bold; color: #2196F3; padding: 5px;")
        cluster_layout.addWidget(self.detected_modes_label)

        cluster_group.setLayout(cluster_layout)
        layout.addWidget(cluster_group)

        # Peak Position Method
        position_group = QGroupBox("Peak Position Method")
        position_layout = QVBoxLayout()

        position_info = QLabel("Choose how to calculate the characteristic decay time for each peak:\n"
                              "• Maximum: Use the tau value at the peak maximum (traditional method)\n"
                              "• Centroid: Use the center of mass in log-space (compensates for log-distortion)")
        position_info.setWordWrap(True)
        position_info.setStyleSheet("color: #666; font-style: italic;")
        position_layout.addWidget(position_info)

        # Centroid checkbox
        self.use_centroid_check = QCheckBox("Use centroid (center of mass) instead of maximum")
        self.use_centroid_check.setChecked(False)
        self.use_centroid_check.setToolTip("Calculate peak position as weighted average in log-space.\n"
                                           "Recommended for distributions with asymmetric or log-distorted peaks.")
        position_layout.addWidget(self.use_centroid_check)

        position_group.setLayout(position_layout)
        layout.addWidget(position_group)

        # Quick presets
        preset_group = QGroupBox("Quick Presets")
        preset_layout = QHBoxLayout()

        monodisperse_btn = QPushButton("Monodisperse\n(1 peak)")
        monodisperse_btn.clicked.connect(lambda: self.apply_preset('monodisperse'))
        preset_layout.addWidget(monodisperse_btn)

        bidisperse_btn = QPushButton("Bidisperse\n(2 peaks)")
        bidisperse_btn.clicked.connect(lambda: self.apply_preset('bidisperse'))
        preset_layout.addWidget(bidisperse_btn)

        polydisperse_btn = QPushButton("Polydisperse\n(multiple peaks)")
        polydisperse_btn.clicked.connect(lambda: self.apply_preset('polydisperse'))
        preset_layout.addWidget(polydisperse_btn)

        sensitive_btn = QPushButton("Very Sensitive\n(find all peaks)")
        sensitive_btn.clicked.connect(lambda: self.apply_preset('sensitive'))
        preset_layout.addWidget(sensitive_btn)

        preset_group.setLayout(preset_layout)
        layout.addWidget(preset_group)

        layout.addStretch()
        widget.setLayout(layout)
        return widget

    def create_preview_tab(self):
        """Create interactive preview tab"""
        widget = QWidget()
        layout = QVBoxLayout()

        # Controls
        controls_layout = QHBoxLayout()

        controls_layout.addWidget(QLabel("Preview datasets:"))
        self.preview_num_spin = QSpinBox()
        self.preview_num_spin.setRange(1, 10)
        self.preview_num_spin.setValue(5)
        controls_layout.addWidget(self.preview_num_spin)

        self.preview_btn = QPushButton("🔍 Generate Preview")
        self.preview_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-weight: bold;
                padding: 8px 15px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        self.preview_btn.clicked.connect(self.generate_preview)
        controls_layout.addWidget(self.preview_btn)

        self.update_preview_btn = QPushButton("↻ Update Preview")
        self.update_preview_btn.setEnabled(False)
        self.update_preview_btn.clicked.connect(self.update_preview)
        controls_layout.addWidget(self.update_preview_btn)

        controls_layout.addStretch()
        layout.addLayout(controls_layout)

        # Info label
        self.preview_info_label = QLabel("Click 'Generate Preview' to see NNLS results on random datasets")
        self.preview_info_label.setStyleSheet("color: #666; font-style: italic; padding: 5px;")
        layout.addWidget(self.preview_info_label)

        # Matplotlib canvas with larger figure size for better visibility
        # Use much larger size to ensure plots are clearly visible
        self.canvas = FigureCanvas(Figure(figsize=(18, 20)))
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setMinimumSize(800, 1000)

        # Navigation toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)

        # Add scroll area to make plots scrollable
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.canvas)
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumHeight(400)
        layout.addWidget(scroll_area)

        widget.setLayout(layout)
        return widget

    def on_prominence_slider_changed(self, value):
        """Handle prominence slider change"""
        prominence = value / 1000.0
        self.prominence_input.blockSignals(True)
        self.prominence_input.setValue(prominence)
        self.prominence_input.blockSignals(False)
        self.params['prominence'] = prominence

    def on_prominence_input_changed(self, value):
        """Handle prominence manual input change"""
        slider_value = int(value * 1000)
        self.prominence_slider.blockSignals(True)
        self.prominence_slider.setValue(slider_value)
        self.prominence_slider.blockSignals(False)
        self.params['prominence'] = value

    def on_distance_slider_changed(self, value):
        """Handle distance slider change"""
        self.distance_input.blockSignals(True)
        self.distance_input.setValue(value)
        self.distance_input.blockSignals(False)
        self.params['distance'] = value

    def on_distance_input_changed(self, value):
        """Handle distance manual input change"""
        self.distance_slider.blockSignals(True)
        self.distance_slider.setValue(value)
        self.distance_slider.blockSignals(False)
        self.params['distance'] = value

    def on_eps_factor_slider_changed(self, value):
        """Handle eps_factor slider change"""
        eps_factor = value / 100.0
        self.eps_factor_input.blockSignals(True)
        self.eps_factor_input.setValue(eps_factor)
        self.eps_factor_input.blockSignals(False)
        self.params['eps_factor'] = eps_factor

    def on_eps_factor_input_changed(self, value):
        """Handle eps_factor manual input change"""
        slider_value = int(value * 100)
        self.eps_factor_slider.blockSignals(True)
        self.eps_factor_slider.setValue(slider_value)
        self.eps_factor_slider.blockSignals(False)
        self.params['eps_factor'] = value

    def apply_preset(self, preset_name):
        """Apply parameter preset"""
        presets = {
            'monodisperse': {'prominence': 0.1, 'distance': 20},
            'bidisperse': {'prominence': 0.05, 'distance': 10},
            'polydisperse': {'prominence': 0.02, 'distance': 3},
            'sensitive': {'prominence': 0.01, 'distance': 1}
        }

        if preset_name in presets:
            prom = presets[preset_name]['prominence']
            dist = presets[preset_name]['distance']

            # Update sliders and inputs (this will trigger the change handlers)
            self.prominence_slider.setValue(int(prom * 1000))
            self.distance_slider.setValue(dist)

            QMessageBox.information(self, "Preset Applied",
                                   f"Applied '{preset_name}' preset:\n"
                                   f"Prominence: {prom}\n"
                                   f"Distance: {dist}\n\n"
                                   f"Click 'Update Preview' to see changes.")

    def generate_preview(self):
        """Generate preview with current parameters"""
        # Update parameters from inputs
        self.update_params_from_inputs()

        # Generate preview
        try:
            self.preview_info_label.setText("Generating preview... Please wait.")
            self.preview_info_label.setStyleSheet("color: orange; font-weight: bold;")

            fig, selected_datasets, preview_results = self.laplace_analyzer.preview_nnls_parameters(
                self.params,
                num_datasets=self.preview_num_spin.value()
            )

            self.preview_datasets = selected_datasets
            self.preview_results = preview_results

            # Perform mock clustering to estimate mode count
            self._estimate_mode_count_from_preview(preview_results, selected_datasets)

            # Create log-plot showing Γ vs q² for preview peaks
            gamma_q2_fig = self._create_gamma_q2_logplot(preview_results, selected_datasets)

            # Close the matplotlib figure and embed in canvas
            plt.close(fig)
            if gamma_q2_fig is not None:
                plt.close(gamma_q2_fig)

            # Clear canvas and draw new figure
            self.canvas.figure.clear()

            # Determine layout: if we have gamma_q2 plot, use GridSpec for better control
            if gamma_q2_fig is not None:
                from matplotlib.gridspec import GridSpec

                # Create layout: top section for preview plots, bottom for gamma-q² plot
                # Give more space to preview plots (height ratio 3:1 instead of 2:1)
                gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.4,
                             figure=self.canvas.figure)

                # We'll create a nested grid for the preview plots
                num_preview_plots = len(fig.get_axes())
                # Use 2 columns instead of 3 for more space per plot
                cols = min(2, num_preview_plots)
                rows = (num_preview_plots + cols - 1) // cols

                # Add more spacing between preview subplots
                gs_preview = gs[0].subgridspec(rows, cols, hspace=0.5, wspace=0.35)

                for i, ax in enumerate(fig.get_axes()):
                    row = i // cols
                    col = i % cols
                    new_ax = self.canvas.figure.add_subplot(gs_preview[row, col])

                    # Copy all artists
                    for line in ax.get_lines():
                        new_ax.plot(line.get_xdata(), line.get_ydata(),
                                   color=line.get_color(),
                                   linewidth=line.get_linewidth(),
                                   linestyle=line.get_linestyle(),
                                   marker=line.get_marker(),
                                   markersize=line.get_markersize(),
                                   label=line.get_label())

                    # Copy annotations
                    for text in ax.texts:
                        bbox_dict = None
                        if text.get_bbox_patch():
                            bbox_patch = text.get_bbox_patch()
                            bbox_dict = dict(
                                boxstyle='round,pad=0.3',
                                facecolor=bbox_patch.get_facecolor(),
                                alpha=bbox_patch.get_alpha() or 0.7
                            )

                        new_ax.text(text.get_position()[0], text.get_position()[1],
                                   text.get_text(),
                                   fontsize=text.get_fontsize(),
                                   bbox=bbox_dict)

                    # Copy properties with larger font sizes for better readability
                    new_ax.set_xlabel(ax.get_xlabel(), fontsize=11)
                    new_ax.set_ylabel(ax.get_ylabel(), fontsize=11)
                    new_ax.set_title(ax.get_title(), fontsize=12, fontweight='bold')
                    new_ax.set_xscale(ax.get_xscale())
                    new_ax.set_yscale(ax.get_yscale())
                    new_ax.grid(True, which='both', alpha=0.3)
                    # Increase tick label sizes
                    new_ax.tick_params(axis='both', which='major', labelsize=10)
                    if ax.get_legend():
                        new_ax.legend(fontsize=9)

                # Bottom section: copy gamma-q² plot
                gamma_ax = gamma_q2_fig.get_axes()[0]
                new_gamma_ax = self.canvas.figure.add_subplot(gs[1])

                # Copy gamma-q² plot
                for line in gamma_ax.get_lines():
                    new_gamma_ax.plot(line.get_xdata(), line.get_ydata(),
                                     color=line.get_color(),
                                     linewidth=line.get_linewidth(),
                                     linestyle=line.get_linestyle(),
                                     marker=line.get_marker(),
                                     markersize=line.get_markersize(),
                                     label=line.get_label(),
                                     alpha=line.get_alpha())

                # Copy text annotations
                for text in gamma_ax.texts:
                    bbox_dict = None
                    if text.get_bbox_patch():
                        bbox_patch = text.get_bbox_patch()
                        bbox_dict = dict(
                            boxstyle='round,pad=0.3',
                            facecolor=bbox_patch.get_facecolor(),
                            alpha=bbox_patch.get_alpha() or 0.7
                        )

                    new_gamma_ax.text(text.get_position()[0], text.get_position()[1],
                                     text.get_text(),
                                     fontsize=text.get_fontsize(),
                                     bbox=bbox_dict)

                new_gamma_ax.set_xlabel(gamma_ax.get_xlabel())
                new_gamma_ax.set_ylabel(gamma_ax.get_ylabel())
                new_gamma_ax.set_title(gamma_ax.get_title())
                new_gamma_ax.set_xscale(gamma_ax.get_xscale())
                new_gamma_ax.set_yscale(gamma_ax.get_yscale())
                new_gamma_ax.grid(True, which='both', alpha=0.3, linestyle='--')
                if gamma_ax.get_legend():
                    new_gamma_ax.legend(fontsize=8)

            else:
                # No gamma-q² plot, use GridSpec for better layout
                from matplotlib.gridspec import GridSpec

                num_preview_plots = len(fig.get_axes())
                # Use 2 columns for more space per plot
                cols = min(2, num_preview_plots)
                rows = (num_preview_plots + cols - 1) // cols

                # Create GridSpec with good spacing
                gs = GridSpec(rows, cols, hspace=0.5, wspace=0.35,
                             figure=self.canvas.figure)

                for i, ax in enumerate(fig.get_axes()):
                    row = i // cols
                    col = i % cols
                    new_ax = self.canvas.figure.add_subplot(gs[row, col])

                    # Copy all artists
                    for line in ax.get_lines():
                        new_ax.plot(line.get_xdata(), line.get_ydata(),
                                   color=line.get_color(),
                                   linewidth=line.get_linewidth(),
                                   linestyle=line.get_linestyle(),
                                   marker=line.get_marker(),
                                   markersize=line.get_markersize(),
                                   label=line.get_label())

                    # Copy annotations
                    for text in ax.texts:
                        bbox_dict = None
                        if text.get_bbox_patch():
                            bbox_patch = text.get_bbox_patch()
                            bbox_dict = dict(
                                boxstyle='round,pad=0.3',
                                facecolor=bbox_patch.get_facecolor(),
                                alpha=bbox_patch.get_alpha() or 0.7
                            )

                        new_ax.text(text.get_position()[0], text.get_position()[1],
                                   text.get_text(),
                                   fontsize=text.get_fontsize(),
                                   bbox=bbox_dict)

                    # Copy properties with larger font sizes
                    new_ax.set_xlabel(ax.get_xlabel(), fontsize=11)
                    new_ax.set_ylabel(ax.get_ylabel(), fontsize=11)
                    new_ax.set_title(ax.get_title(), fontsize=12, fontweight='bold')
                    new_ax.set_xscale(ax.get_xscale())
                    new_ax.set_yscale(ax.get_yscale())
                    new_ax.grid(True, which='both', alpha=0.3)
                    new_ax.tick_params(axis='both', which='major', labelsize=10)
                    if ax.get_legend():
                        new_ax.legend(fontsize=9)

            # Add overall title
            self.canvas.figure.suptitle(
                f'NNLS Preview - Prominence: {self.params["prominence"]:.3f}, Distance: {self.params["distance"]}, '
                f'eps_factor: {self.params["eps_factor"]:.2f}',
                fontsize=12, fontweight='bold'
            )

            self.canvas.figure.tight_layout()
            self.canvas.draw()

            self.preview_info_label.setText(
                f"Preview generated for {len(selected_datasets)} datasets. "
                f"Adjust parameters and click 'Update Preview' to see changes."
            )
            self.preview_info_label.setStyleSheet("color: green; font-weight: bold;")

            self.update_preview_btn.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "Preview Error",
                               f"Error generating preview:\n{str(e)}")
            self.preview_info_label.setText("Error generating preview")
            self.preview_info_label.setStyleSheet("color: red; font-weight: bold;")

    def update_preview(self):
        """Update preview with current parameter values"""
        self.generate_preview()

    def _estimate_mode_count_from_preview(self, preview_results, selected_datasets):
        """
        Estimate the number of modes from preview results using clustering

        Args:
            preview_results: List of result dicts from NNLS preview
            selected_datasets: List of dataset names used in preview
        """
        try:
            # Check if clustering is enabled
            if not self.params.get('use_clustering', True):
                self.detected_modes_label.setText("Detected modes: N/A (clustering disabled)")
                self.detected_modes_label.setStyleSheet("font-weight: bold; color: gray; padding: 5px;")
                return

            # Create a DataFrame from preview results
            preview_df = pd.DataFrame(preview_results)

            # Get q² values from basedata for the selected datasets
            basedata = self.laplace_analyzer.df_basedata

            # Match filenames to get q² values
            q_squared_values = []
            for filename in selected_datasets:
                # Find matching row in basedata
                matching_rows = basedata[basedata['filename'] == filename]
                if len(matching_rows) > 0:
                    q_squared_values.append(matching_rows.iloc[0]['q^2'])
                else:
                    # If not found, try to calculate from angle
                    print(f"[Preview Clustering] Warning: {filename} not found in basedata")
                    q_squared_values.append(None)

            # Add q² column to preview_df
            preview_df['q^2'] = q_squared_values

            # Remove rows with missing q² values
            preview_df = preview_df.dropna(subset=['q^2'])

            if len(preview_df) < 2:
                self.detected_modes_label.setText("Detected modes: N/A (insufficient data)")
                self.detected_modes_label.setStyleSheet("font-weight: bold; color: orange; padding: 5px;")
                return

            # Perform mock clustering
            from peak_clustering import cluster_peaks_across_datasets

            eps_factor = self.params.get('eps_factor', 0.3)

            clustered_df, cluster_info = cluster_peaks_across_datasets(
                preview_df,
                tau_prefix='tau_',
                method='dbscan',
                eps_factor=eps_factor,
                min_samples=max(2, int(0.2 * len(preview_df)))
            )

            # Count modes
            n_modes = len(cluster_info)

            # Update label
            mode_text = f"Detected modes: {n_modes} mode{'s' if n_modes != 1 else ''}"
            self.detected_modes_label.setText(mode_text)

            # Color based on mode count
            if n_modes == 0:
                color = "#f44336"  # Red - no modes
            elif n_modes == 1:
                color = "#4CAF50"  # Green - monodisperse
            elif n_modes <= 3:
                color = "#2196F3"  # Blue - few modes
            else:
                color = "#FF9800"  # Orange - many modes

            self.detected_modes_label.setStyleSheet(f"font-weight: bold; color: {color}; padding: 5px;")

            print(f"[Preview Clustering] Estimated {n_modes} modes with eps_factor={eps_factor:.2f}")

        except Exception as e:
            print(f"[Preview Clustering] Error estimating mode count: {e}")
            import traceback
            traceback.print_exc()
            self.detected_modes_label.setText(f"Detected modes: Error ({str(e)[:30]}...)")
            self.detected_modes_label.setStyleSheet("font-weight: bold; color: red; padding: 5px;")

    def _create_gamma_q2_logplot(self, preview_results, selected_datasets):
        """
        Create a log-plot showing Γ vs q² for all preview peaks

        This helps visualize the clustering behavior and identify if modes
        are being correctly grouped across different scattering angles.

        Args:
            preview_results: List of result dicts from NNLS preview
            selected_datasets: List of dataset names used in preview

        Returns:
            matplotlib Figure or None if error
        """
        try:
            # Create a DataFrame from preview results
            preview_df = pd.DataFrame(preview_results)

            # Get q² values from basedata
            basedata = self.laplace_analyzer.df_basedata

            q_squared_values = []
            for filename in selected_datasets:
                matching_rows = basedata[basedata['filename'] == filename]
                if len(matching_rows) > 0:
                    q_squared_values.append(matching_rows.iloc[0]['q^2'])
                else:
                    q_squared_values.append(None)

            preview_df['q^2'] = q_squared_values
            preview_df = preview_df.dropna(subset=['q^2'])

            if len(preview_df) < 1:
                return None

            # Find all tau columns
            tau_cols = [col for col in preview_df.columns if col.startswith('tau_')]
            if len(tau_cols) == 0:
                return None

            # Calculate Γ values from tau values
            # Γ = 1/τ
            from regularized_optimized import calculate_decay_rates
            preview_df = calculate_decay_rates(preview_df, tau_cols)

            # Get gamma columns
            gamma_cols = [col.replace('tau', 'gamma') for col in tau_cols]

            # Create figure
            fig = Figure(figsize=(10, 6))
            ax = fig.add_subplot(111)

            # Plot each peak as a separate series
            colors = plt.cm.tab10(np.linspace(0, 1, len(gamma_cols)))

            for i, (gamma_col, tau_col) in enumerate(zip(gamma_cols, tau_cols)):
                if gamma_col not in preview_df.columns:
                    continue

                # Extract valid data points
                valid_mask = preview_df[gamma_col].notna()
                q2_values = preview_df.loc[valid_mask, 'q^2'].values
                gamma_values = preview_df.loc[valid_mask, gamma_col].values

                if len(q2_values) == 0:
                    continue

                # Plot in log-log scale
                ax.loglog(q2_values, gamma_values, 'o', color=colors[i],
                         markersize=8, alpha=0.7, label=f'Peak {i+1}',
                         markeredgecolor='black', markeredgewidth=0.5)

                # Try to fit a line if we have enough points
                if len(q2_values) >= 2:
                    # Linear fit in log-space: log(Γ) = log(D) + log(q²)
                    # So Γ = D × q²
                    log_q2 = np.log10(q2_values)
                    log_gamma = np.log10(gamma_values)

                    # Simple linear fit
                    coeffs = np.polyfit(log_q2, log_gamma, 1)
                    slope = coeffs[0]
                    intercept = coeffs[1]

                    # Plot fit line
                    q2_fit = np.logspace(np.log10(q2_values.min()), np.log10(q2_values.max()), 50)
                    gamma_fit = 10**(slope * np.log10(q2_fit) + intercept)

                    ax.loglog(q2_fit, gamma_fit, '--', color=colors[i],
                             alpha=0.5, linewidth=1.5)

                    # Calculate D from slope
                    # If slope ≈ 1, then D ≈ 10^intercept
                    D_estimate = 10**intercept if abs(slope - 1.0) < 0.2 else None

                    if D_estimate is not None:
                        # Add label with D estimate
                        mid_idx = len(q2_fit) // 2
                        ax.text(q2_fit[mid_idx], gamma_fit[mid_idx],
                               f'D≈{D_estimate:.2e}',
                               fontsize=8, color=colors[i],
                               bbox=dict(boxstyle='round,pad=0.3',
                                       facecolor='white', alpha=0.7))

            ax.set_xlabel(r'q² [nm$^{-2}$]', fontsize=12, fontweight='bold')
            ax.set_ylabel(r'$\Gamma$ [1/s]', fontsize=12, fontweight='bold')
            ax.set_title('Relaxation Rate vs Scattering Vector (Log-Log Plot)\n'
                        'Helps visualize mode clustering across angles',
                        fontsize=12, fontweight='bold')
            ax.grid(True, which='both', alpha=0.3, linestyle='--')
            ax.legend(loc='best', fontsize=9)

            # Add reference line for Γ ∝ q² (slope = 1)
            q2_range = ax.get_xlim()
            gamma_ref = np.array(q2_range)
            ax.loglog(q2_range, gamma_ref * np.median(gamma_ref) / np.median(q2_range),
                     'k:', linewidth=1, alpha=0.3, label='Γ ∝ q² (reference)')

            fig.tight_layout()

            # Store figure for later display
            self.gamma_q2_figure = fig

            return fig

        except Exception as e:
            print(f"[Gamma-q² Plot] Error creating log-plot: {e}")
            import traceback
            traceback.print_exc()
            return None

    def update_params_from_inputs(self):
        """Update parameters dictionary from input widgets"""
        # Decay times
        start = self.decay_start_spin.value()
        end = self.decay_end_spin.value()
        num = self.decay_num_spin.value()

        self.params['decay_times'] = np.logspace(np.log10(start), np.log10(end), num)

        # Peak detection - use values from input fields (they're synced with sliders)
        self.params['prominence'] = self.prominence_input.value()
        self.params['distance'] = self.distance_input.value()

        # Clustering parameters
        self.params['use_clustering'] = self.clustering_enabled_check.isChecked()
        self.params['eps_factor'] = self.eps_factor_input.value()

        # Peak position method
        self.params['use_centroid'] = self.use_centroid_check.isChecked()

        # Processing options
        self.params['use_multiprocessing'] = self.multiprocessing_check.isChecked()
        self.params['show_plots'] = self.show_plots_check.isChecked()

    def accept_parameters(self):
        """Accept parameters and emit signal"""
        self.update_params_from_inputs()

        # Emit parameters
        self.parameters_accepted.emit(self.params)
        self.accept()

    def get_parameters(self):
        """Get current parameters"""
        self.update_params_from_inputs()
        return self.params
