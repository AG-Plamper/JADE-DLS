"""
Post-Fit Refinement Dialog for Laplace Methods (NNLS and Regularized)

Allows refinement of NNLS/Regularized analysis results after initial fitting:
- Adjust q² range for diffusion coefficient calculation
- Visual Γ vs q² plots with interactive range selection
- Exclude individual distribution plots based on quality
"""

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QDoubleSpinBox, QFormLayout, QTabWidget,
                             QWidget, QGroupBox, QMessageBox, QSizePolicy,
                             QListWidget, QListWidgetItem, QCheckBox, QScrollArea,
                             QSplitter, QSpinBox)
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


class InteractivePlotWidget(QWidget):
    """
    Widget with interactive Γ vs q² plot for range selection
    """

    def __init__(self, data, gamma_cols, q_squared_col, method_name, parent=None):
        super().__init__(parent)
        self.data = data
        self.gamma_cols = gamma_cols if isinstance(gamma_cols, list) else [gamma_cols]
        self.q_squared_col = q_squared_col
        self.method_name = method_name
        self.parent_dialog = parent

        # Range selection
        self.q_min = None
        self.q_max = None
        self.selection_rect = None
        self.click_start = None

        self.init_ui()

    def init_ui(self):
        """Initialize UI with matplotlib plot"""
        layout = QVBoxLayout()

        # Create matplotlib figure
        self.figure, self.ax = plt.subplots(figsize=(8, 5))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Add navigation toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)

        # Plot initial data
        self.plot_data()

        # Connect mouse events for range selection
        self.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.canvas.mpl_connect('button_release_event', self.on_mouse_release)

        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        # Instructions
        instructions = QLabel(
            "📌 <b>Interactive Selection:</b> Click and drag on the plot to select q² range. "
            "The spinboxes below will update automatically."
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("padding: 5px; background-color: #FFF3E0; border-radius: 3px;")
        layout.addWidget(instructions)

        self.setLayout(layout)

    def plot_data(self):
        """Plot Γ vs q² data for all peaks"""
        self.ax.clear()

        # Extract data
        q_squared = self.data[self.q_squared_col].values

        # Color cycle
        colors = plt.cm.tab10(range(len(self.gamma_cols)))

        # Plot each gamma column (each peak)
        for i, gamma_col in enumerate(self.gamma_cols):
            if gamma_col in self.data.columns:
                gamma = self.data[gamma_col].values

                # Sort by q²
                sort_idx = np.argsort(q_squared)
                q_sorted = q_squared[sort_idx]
                gamma_sorted = gamma[sort_idx]

                # Plot data points
                self.ax.plot(q_sorted, gamma_sorted, 'o', markersize=8,
                            label=f'{gamma_col}', color=colors[i], alpha=0.7)

                # Simple linear regression for visualization
                if len(q_sorted) > 1:
                    coeffs = np.polyfit(q_sorted, gamma_sorted, 1)
                    fit_line = np.poly1d(coeffs)
                    q_fit = np.linspace(q_sorted.min(), q_sorted.max(), 100)
                    self.ax.plot(q_fit, fit_line(q_fit), '--', color=colors[i],
                                linewidth=2, alpha=0.6)

        self.ax.set_xlabel('q² [nm⁻²]', fontsize=11, fontweight='bold')
        self.ax.set_ylabel('Γ [s⁻¹]', fontsize=11, fontweight='bold')
        self.ax.set_title(f'{self.method_name}: Diffusion Coefficient Fit', fontsize=12, fontweight='bold')
        self.ax.legend(loc='best')
        self.ax.grid(True, alpha=0.3)

        self.canvas.draw()

    def on_mouse_press(self, event):
        """Handle mouse press event"""
        if event.inaxes == self.ax and event.button == 1:  # Left click
            self.click_start = event.xdata
            # Remove previous selection rectangle if exists
            if self.selection_rect:
                self.selection_rect.remove()
                self.selection_rect = None

    def on_mouse_move(self, event):
        """Handle mouse move event - show preview of selection"""
        if self.click_start is not None and event.inaxes == self.ax and event.xdata is not None:
            # Remove old rectangle
            if self.selection_rect:
                self.selection_rect.remove()

            # Draw new rectangle
            x_min = min(self.click_start, event.xdata)
            x_max = max(self.click_start, event.xdata)
            y_limits = self.ax.get_ylim()

            self.selection_rect = Rectangle(
                (x_min, y_limits[0]),
                x_max - x_min,
                y_limits[1] - y_limits[0],
                fill=True,
                facecolor='green',
                alpha=0.2,
                edgecolor='green',
                linewidth=2
            )
            self.ax.add_patch(self.selection_rect)
            self.canvas.draw()

    def on_mouse_release(self, event):
        """Handle mouse release event - finalize selection"""
        if self.click_start is not None and event.inaxes == self.ax and event.xdata is not None:
            # Calculate selected range
            self.q_min = min(self.click_start, event.xdata)
            self.q_max = max(self.click_start, event.xdata)

            # Update parent dialog spinboxes
            if hasattr(self.parent_dialog, 'update_q_range_from_plot'):
                self.parent_dialog.update_q_range_from_plot(self.q_min, self.q_max)

        self.click_start = None


class DistributionInspectorWidget(QWidget):
    """
    Widget to display and select distributions to exclude
    """

    def __init__(self, plots_dict, method_name, parent=None):
        super().__init__(parent)
        self.plots_dict = plots_dict  # {filename: (fig, data)}
        self.method_name = method_name
        self.excluded_files = []

        self.init_ui()

    def init_ui(self):
        """Initialize the distribution inspector UI"""
        layout = QVBoxLayout()

        # Info
        info_label = QLabel(
            f"<b>Select distributions to exclude</b><br>"
            f"Use arrow keys or mouse to navigate. Check the boxes to exclude distributions from analysis.<br>"
            f"Press Space to toggle exclusion for the selected distribution."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Create horizontal split: List on left, Plot on right
        splitter = QSplitter(Qt.Horizontal)

        # Left side: List with checkboxes
        left_widget = QWidget()
        left_layout = QVBoxLayout()

        # Create QListWidget for distribution list (better keyboard navigation)
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QListWidget.SingleSelection)

        # Filter and add items
        filenames = sorted(self.plots_dict.keys())
        self.valid_filenames = []

        # Debug: Print available plots
        print(f"[DistributionInspector] Available plots: {len(filenames)}")

        for filename in filenames:
            # Skip summary/diffusion analysis plots but keep individual distribution plots
            if 'Summary' in filename or 'Diffusion Analysis' in filename or 'Analysis' in filename:
                print(f"[DistributionInspector] Skipping summary plot: {filename}")
                continue  # Skip summary plots

            self.valid_filenames.append(filename)

            # Create list item with checkbox
            item = QListWidgetItem(filename)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.list_widget.addItem(item)

        print(f"[DistributionInspector] Created {len(self.valid_filenames)} items")

        # Connect selection changed signal
        self.list_widget.currentItemChanged.connect(self._on_selection_changed)
        self.list_widget.itemChanged.connect(self._update_count)

        left_layout.addWidget(self.list_widget)

        # Selection controls
        control_layout = QHBoxLayout()

        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(self._select_all)
        control_layout.addWidget(select_all_btn)

        deselect_all_btn = QPushButton("Deselect All")
        deselect_all_btn.clicked.connect(self._deselect_all)
        control_layout.addWidget(deselect_all_btn)

        control_layout.addStretch()

        # Count label
        self.count_label = QLabel("0 distributions selected for exclusion")
        control_layout.addWidget(self.count_label)

        left_layout.addLayout(control_layout)
        left_widget.setLayout(left_layout)

        # Right side: Plot display
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

        right_widget = QWidget()
        right_layout = QVBoxLayout()

        # Placeholder figure
        from matplotlib.figure import Figure
        self.current_figure = Figure(figsize=(8, 6))
        ax = self.current_figure.add_subplot(111)
        ax.text(0.5, 0.5, 'Click on a distribution to view its plot',
                ha='center', va='center', fontsize=12, color='gray')
        ax.set_xticks([])
        ax.set_yticks([])

        self.canvas = FigureCanvas(self.current_figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        right_layout.addWidget(self.toolbar)
        right_layout.addWidget(self.canvas)
        right_widget.setLayout(right_layout)

        # Add both sides to splitter
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 1)  # Left: 1 part
        splitter.setStretchFactor(1, 2)  # Right: 2 parts

        layout.addWidget(splitter)

        # Select first item by default
        if self.list_widget.count() > 0:
            self.list_widget.setCurrentRow(0)

        self.setLayout(layout)

    def _on_selection_changed(self, current, previous):
        """Handle selection change in list widget"""
        if current is not None:
            filename = current.text()
            self._show_plot(filename)

    def _select_all(self):
        """Select all distributions for exclusion"""
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            item.setCheckState(Qt.Checked)

    def _deselect_all(self):
        """Deselect all distributions"""
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            item.setCheckState(Qt.Unchecked)

    def _show_plot(self, filename):
        """Display the plot for the selected distribution"""
        if filename not in self.plots_dict:
            print(f"[DistributionInspector] Plot not found for: {filename}")
            return

        # Clear current figure
        self.current_figure.clear()

        # Get the stored figure (plots_dict contains tuples: (fig, data))
        plot_entry = self.plots_dict[filename]
        if isinstance(plot_entry, tuple):
            stored_fig, _ = plot_entry  # Unpack tuple
        else:
            stored_fig = plot_entry  # Handle case where it's just a figure

        # Copy axes from stored figure to current figure
        if hasattr(stored_fig, 'axes') and len(stored_fig.axes) > 0:
            for i, ax_src in enumerate(stored_fig.axes):
                # Create subplot in same position
                ax_dest = self.current_figure.add_subplot(len(stored_fig.axes), 1, i+1)

                # Copy plot contents
                for line in ax_src.get_lines():
                    ax_dest.plot(line.get_xdata(), line.get_ydata(),
                               color=line.get_color(),
                               linestyle=line.get_linestyle(),
                               linewidth=line.get_linewidth(),
                               label=line.get_label(),
                               marker=line.get_marker(),
                               markersize=line.get_markersize())

                # Copy axis labels and title
                ax_dest.set_xlabel(ax_src.get_xlabel())
                ax_dest.set_ylabel(ax_src.get_ylabel())
                ax_dest.set_title(ax_src.get_title())

                # Copy scale
                ax_dest.set_xscale(ax_src.get_xscale())
                ax_dest.set_yscale(ax_src.get_yscale())

                # Copy grid settings
                # Check if grid is enabled by inspecting gridlines visibility
                grid_visible = (len(ax_src.xaxis.get_gridlines()) > 0 and
                              ax_src.xaxis.get_gridlines()[0].get_visible())
                ax_dest.grid(grid_visible)

                # Copy legend if exists
                if ax_src.get_legend() is not None:
                    ax_dest.legend()

                # Copy text annotations
                for text in ax_src.texts:
                    # Properly handle bbox patch
                    bbox_dict = None
                    if text.get_bbox_patch():
                        bbox_patch = text.get_bbox_patch()
                        bbox_dict = dict(
                            boxstyle='round,pad=0.3',
                            facecolor=bbox_patch.get_facecolor(),
                            edgecolor=bbox_patch.get_edgecolor(),
                            alpha=bbox_patch.get_alpha() or 0.7
                        )

                    ax_dest.text(text.get_position()[0], text.get_position()[1],
                               text.get_text(),
                               transform=ax_dest.transData if text.get_transform() == ax_src.transData else ax_dest.transAxes,
                               fontsize=text.get_fontsize(),
                               color=text.get_color(),
                               ha=text.get_ha(),
                               va=text.get_va(),
                               bbox=bbox_dict)

        else:
            # Fallback: show message
            ax = self.current_figure.add_subplot(111)
            ax.text(0.5, 0.5, f'Plot loaded: {filename}',
                   ha='center', va='center', fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

        self.current_figure.tight_layout()
        self.canvas.draw()

        print(f"[DistributionInspector] Displayed plot: {filename}")

    def _update_count(self, item=None):
        """Update the count of excluded distributions"""
        count = 0
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.checkState() == Qt.Checked:
                count += 1
        self.count_label.setText(f"{count} distribution(s) selected for exclusion")

    def get_excluded_files(self):
        """Get list of files to exclude"""
        excluded = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.checkState() == Qt.Checked:
                excluded.append(item.text())
        return excluded


class LaplacePostFitRefinementDialog(QDialog):
    """
    Dialog for post-fit refinement of Laplace analysis results (NNLS or Regularized)

    Provides two-stage refinement:
    - Tab 1: Q² range adjustment with interactive plot
    - Tab 2: Distribution exclusion
    """

    def __init__(self, analyzer, method_name, parent=None):
        """
        Initialize the post-fit refinement dialog

        Args:
            analyzer: LaplaceAnalyzer instance with existing results
            method_name: 'NNLS' or 'Regularized'
            parent: Parent widget
        """
        super().__init__(parent)
        self.analyzer = analyzer
        self.method_name = method_name
        self.is_nnls = (method_name == 'NNLS')

        # Store refinement parameters
        self.q_range = None
        self.excluded_files = []

        self.setWindowTitle(f"{method_name} Post-Fit Refinement")
        self.setModal(True)
        self.resize(900, 700)

        self.init_ui()

    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout()

        # Title and description
        title = QLabel(f"{self.method_name} Post-Fit Refinement")
        title.setStyleSheet("font-size: 14pt; font-weight: bold;")
        layout.addWidget(title)

        description = QLabel(
            f"Refine your {self.method_name} analysis results by adjusting the q² range for "
            "diffusion coefficient calculation and/or excluding individual distributions."
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        # Create tabs
        self.tabs = QTabWidget()

        # Tab 1: Q² range adjustment
        self.tabs.addTab(self.create_q_range_tab(), "1. Fitting Range")

        # Tab 2: Distribution exclusion
        self.tabs.addTab(self.create_exclusion_tab(), "2. Distribution Selection")

        layout.addWidget(self.tabs)

        # Buttons
        button_layout = QHBoxLayout()

        self.reset_btn = QPushButton("Reset All")
        self.reset_btn.clicked.connect(self.reset_all)
        button_layout.addWidget(self.reset_btn)

        button_layout.addStretch()

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)

        self.apply_btn = QPushButton("Apply")
        self.apply_btn.setDefault(True)
        self.apply_btn.clicked.connect(self.apply_refinement)
        self.apply_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 6px 16px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        button_layout.addWidget(self.apply_btn)

        layout.addLayout(button_layout)

        self.setLayout(layout)

    def create_q_range_tab(self):
        """Create q² range adjustment tab with interactive plot"""
        tab = QWidget()
        layout = QVBoxLayout()

        # Get data for plotting
        if self.is_nnls:
            data = self.analyzer.nnls_data
            gamma_cols = [col for col in data.columns if col.startswith('gamma_')]
        else:
            data = self.analyzer.regularized_data
            gamma_cols = [col for col in data.columns if col.startswith('gamma_')]

        # Add interactive Γ vs q² plot
        if data is not None and gamma_cols:
            self.plot_widget = InteractivePlotWidget(
                data,
                gamma_cols,
                'q^2',
                self.method_name,
                self
            )
            layout.addWidget(self.plot_widget)

        # Q² range group
        q_range_group = QGroupBox("Q² Range for Diffusion Coefficient Fit")
        q_range_layout = QFormLayout()

        # Get current q² range from data
        if data is not None and 'q^2' in data.columns:
            q_squared_values = data['q^2'].values
            min_q = float(q_squared_values.min())
            max_q = float(q_squared_values.max())
        else:
            min_q = 0.0
            max_q = 0.01

        # Min Q²
        self.q_min = QDoubleSpinBox()
        self.q_min.setRange(0, 1)
        self.q_min.setValue(min_q)
        self.q_min.setDecimals(6)
        self.q_min.setSingleStep(0.0001)
        self.q_min.setSuffix(" nm⁻²")
        q_range_layout.addRow("Min Q²:", self.q_min)

        # Max Q²
        self.q_max = QDoubleSpinBox()
        self.q_max.setRange(0, 1)
        self.q_max.setValue(max_q)
        self.q_max.setDecimals(6)
        self.q_max.setSingleStep(0.0001)
        self.q_max.setSuffix(" nm⁻²")
        q_range_layout.addRow("Max Q²:", self.q_max)

        q_range_group.setLayout(q_range_layout)
        layout.addWidget(q_range_group)

        # Info
        info = QLabel(
            "<b>Effect:</b> Re-calculate diffusion coefficient using only data points "
            "within the specified q² range. Individual distribution fits remain unchanged."
        )
        info.setWordWrap(True)
        info.setStyleSheet("padding: 10px; background-color: #e3f2fd; border-radius: 4px;")
        layout.addWidget(info)

        tab.setLayout(layout)
        return tab

    def create_exclusion_tab(self):
        """Create distribution exclusion tab"""
        tab = QWidget()
        layout = QVBoxLayout()

        # Get plots
        if self.is_nnls:
            plots_dict = self.analyzer.nnls_plots if hasattr(self.analyzer, 'nnls_plots') else {}
        else:
            plots_dict = self.analyzer.regularized_plots if hasattr(self.analyzer, 'regularized_plots') else {}

        if plots_dict:
            self.exclusion_widget = DistributionInspectorWidget(plots_dict, self.method_name, self)
            layout.addWidget(self.exclusion_widget)
        else:
            layout.addWidget(QLabel("No distribution plots available for inspection."))

        # Info
        info = QLabel(
            "<b>Effect:</b> Excluded distributions will be removed from the diffusion "
            "coefficient analysis. The q² vs Γ regression will be recalculated without them."
        )
        info.setWordWrap(True)
        info.setStyleSheet("padding: 10px; background-color: #e3f2fd; border-radius: 4px;")
        layout.addWidget(info)

        tab.setLayout(layout)
        return tab

    def update_q_range_from_plot(self, q_min, q_max):
        """
        Update spinboxes when user selects range in plot

        Args:
            q_min: Minimum q² value
            q_max: Maximum q² value
        """
        self.q_min.setValue(q_min)
        self.q_max.setValue(q_max)

    def reset_all(self):
        """Reset all refinement parameters to defaults"""
        reply = QMessageBox.question(
            self,
            "Reset All",
            "Reset all refinement parameters to their default values?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            # Get default q² range
            if self.is_nnls:
                data = self.analyzer.nnls_data
            else:
                data = self.analyzer.regularized_data

            if data is not None and 'q^2' in data.columns:
                q_squared_values = data['q^2'].values
                min_q = float(q_squared_values.min())
                max_q = float(q_squared_values.max())
            else:
                min_q = 0.0
                max_q = 0.01

            self.q_min.setValue(min_q)
            self.q_max.setValue(max_q)

            # Reset exclusions
            if hasattr(self, 'exclusion_widget'):
                self.exclusion_widget._deselect_all()

    def apply_refinement(self):
        """Apply refinement and recalculate results"""
        # Validate inputs
        if self.q_min.value() >= self.q_max.value():
            QMessageBox.warning(
                self,
                "Invalid Range",
                f"{self.method_name}: Minimum q² must be less than maximum q²."
            )
            return

        self.q_range = (self.q_min.value(), self.q_max.value())

        # Get excluded files
        if hasattr(self, 'exclusion_widget'):
            self.excluded_files = self.exclusion_widget.get_excluded_files()

        # Accept dialog
        self.accept()

    def get_refinement_params(self):
        """
        Get refinement parameters

        Returns:
            dict: Dictionary with refinement parameters
        """
        return {
            'q_range': self.q_range,
            'excluded_files': self.excluded_files
        }
