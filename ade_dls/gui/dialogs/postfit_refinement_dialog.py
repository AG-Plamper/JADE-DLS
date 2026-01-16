"""
Post-Fit Refinement Dialog with Interactive Plots

Allows refinement of cumulant analysis results after initial fitting:
- Adjust q² range for diffusion coefficient calculation (all methods)
- Visual Γ vs q² plots with interactive range selection (Methods A & B)
- Exclude individual fits (Method C only)
"""

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QDoubleSpinBox, QFormLayout, QTabWidget,
                             QWidget, QGroupBox, QMessageBox, QSizePolicy)
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.patches import Rectangle
import numpy as np


class InteractivePlotWidget(QWidget):
    """
    Widget with interactive Γ vs q² plot for range selection
    """

    def __init__(self, data, gamma_col, q_squared_col, method_name, parent=None):
        super().__init__(parent)
        self.data = data
        self.gamma_col = gamma_col
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
        """Plot Γ vs q² data"""
        self.ax.clear()

        # Extract data
        q_squared = self.data[self.q_squared_col].values
        gamma = self.data[self.gamma_col].values

        # Sort by q²
        sort_idx = np.argsort(q_squared)
        q_squared = q_squared[sort_idx]
        gamma = gamma[sort_idx]

        # Plot data points
        self.ax.plot(q_squared, gamma, 'o', markersize=8, label='Data', color='#2196F3', alpha=0.7)

        # If we have a linear fit, show it
        if len(q_squared) > 1:
            # Simple linear regression for visualization
            coeffs = np.polyfit(q_squared, gamma, 1)
            fit_line = np.poly1d(coeffs)
            q_fit = np.linspace(q_squared.min(), q_squared.max(), 100)
            self.ax.plot(q_fit, fit_line(q_fit), '--', color='#FF5722',
                        linewidth=2, label=f'Linear fit (slope={coeffs[0]:.2e})', alpha=0.8)

        self.ax.set_xlabel('q² [nm⁻²]', fontsize=11, fontweight='bold')
        self.ax.set_ylabel('Γ [ms⁻¹]', fontsize=11, fontweight='bold')
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
                self.parent_dialog.update_q_range_from_plot(self.q_min, self.q_max, self.method_name)

        self.click_start = None


class PostFitRefinementDialog(QDialog):
    """
    Dialog for post-fit refinement of cumulant analysis results

    Provides method-specific refinement options:
    - Method A: Q² range adjustment
    - Method B: Q² range adjustment
    - Method C: Q² range adjustment + fit exclusion
    """

    def __init__(self, analyzer, available_methods, parent=None):
        """
        Initialize the post-fit refinement dialog

        Args:
            analyzer: CumulantAnalyzer instance with existing results
            available_methods: List of methods with results ['A', 'B', 'C']
            parent: Parent widget
        """
        super().__init__(parent)
        self.analyzer = analyzer
        self.available_methods = available_methods

        # Store current q-ranges for each method
        self.q_ranges = {
            'A': None,
            'B': None,
            'C': None
        }

        # Store excluded fits for Method C
        self.excluded_fits_c = []

        self.setWindowTitle("Post-Fit Refinement")
        self.setModal(True)
        self.resize(900, 700)  # Larger size to accommodate plots

        self.init_ui()

    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout()

        # Title and description
        title = QLabel("Post-Fit Refinement")
        title.setStyleSheet("font-size: 14pt; font-weight: bold;")
        layout.addWidget(title)

        description = QLabel(
            "Refine your cumulant analysis results by adjusting the q² range for "
            "diffusion coefficient calculation and/or excluding individual fits."
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        # Create tabs for each method
        self.tabs = QTabWidget()

        if 'A' in self.available_methods:
            self.tabs.addTab(self.create_method_a_tab(), "Method A")

        if 'B' in self.available_methods:
            self.tabs.addTab(self.create_method_b_tab(), "Method B")

        if 'C' in self.available_methods:
            self.tabs.addTab(self.create_method_c_tab(), "Method C")

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

    def create_method_a_tab(self):
        """Create refinement tab for Method A with interactive plot"""
        tab = QWidget()
        layout = QVBoxLayout()

        # Add interactive Γ vs q² plot if we have Method A data
        if hasattr(self.analyzer, 'method_a_data') and self.analyzer.method_a_data is not None:
            self.plot_widget_a = InteractivePlotWidget(
                self.analyzer.method_a_data,
                '1st order frequency [1/ms]',  # Gamma column for Method A
                'q^2',
                'Method A',
                self
            )
            layout.addWidget(self.plot_widget_a)

        # Q² range group
        q_range_group = QGroupBox("Q² Range for Diffusion Coefficient Fit")
        q_range_layout = QFormLayout()

        # Get current q² range from basedata
        if hasattr(self.analyzer, 'df_basedata') and self.analyzer.df_basedata is not None:
            q_squared_values = self.analyzer.df_basedata['q^2'].values
            min_q = float(q_squared_values.min())
            max_q = float(q_squared_values.max())
        else:
            min_q = 0.0
            max_q = 0.01

        # Min Q²
        self.a_q_min = QDoubleSpinBox()
        self.a_q_min.setRange(0, 1)
        self.a_q_min.setValue(min_q)
        self.a_q_min.setDecimals(6)
        self.a_q_min.setSingleStep(0.0001)
        self.a_q_min.setSuffix(" nm⁻²")
        q_range_layout.addRow("Min Q²:", self.a_q_min)

        # Max Q²
        self.a_q_max = QDoubleSpinBox()
        self.a_q_max.setRange(0, 1)
        self.a_q_max.setValue(max_q)
        self.a_q_max.setDecimals(6)
        self.a_q_max.setSingleStep(0.0001)
        self.a_q_max.setSuffix(" nm⁻²")
        q_range_layout.addRow("Max Q²:", self.a_q_max)

        q_range_group.setLayout(q_range_layout)
        layout.addWidget(q_range_group)

        # Info
        info = QLabel(
            "<b>Effect:</b> Re-calculate diffusion coefficient using only data points "
            "within the specified q² range. Individual cumulant fits remain unchanged."
        )
        info.setWordWrap(True)
        info.setStyleSheet("padding: 10px; background-color: #e3f2fd; border-radius: 4px;")
        layout.addWidget(info)

        tab.setLayout(layout)
        return tab

    def create_method_b_tab(self):
        """Create refinement tab for Method B with interactive plot"""
        tab = QWidget()
        layout = QVBoxLayout()

        # Add interactive Γ vs q² plot if we have Method B data
        if hasattr(self.analyzer, 'method_b_data') and self.analyzer.method_b_data is not None:
            self.plot_widget_b = InteractivePlotWidget(
                self.analyzer.method_b_data,
                'b',  # Gamma column for Method B
                'q^2',
                'Method B',
                self
            )
            layout.addWidget(self.plot_widget_b)

        # Q² range group
        q_range_group = QGroupBox("Q² Range for Diffusion Coefficient Fit")
        q_range_layout = QFormLayout()

        # Get current q² range from basedata
        if hasattr(self.analyzer, 'df_basedata') and self.analyzer.df_basedata is not None:
            q_squared_values = self.analyzer.df_basedata['q^2'].values
            min_q = float(q_squared_values.min())
            max_q = float(q_squared_values.max())
        else:
            min_q = 0.0
            max_q = 0.01

        # Min Q²
        self.b_q_min = QDoubleSpinBox()
        self.b_q_min.setRange(0, 1)
        self.b_q_min.setValue(min_q)
        self.b_q_min.setDecimals(6)
        self.b_q_min.setSingleStep(0.0001)
        self.b_q_min.setSuffix(" nm⁻²")
        q_range_layout.addRow("Min Q²:", self.b_q_min)

        # Max Q²
        self.b_q_max = QDoubleSpinBox()
        self.b_q_max.setRange(0, 1)
        self.b_q_max.setValue(max_q)
        self.b_q_max.setDecimals(6)
        self.b_q_max.setSingleStep(0.0001)
        self.b_q_max.setSuffix(" nm⁻²")
        q_range_layout.addRow("Max Q²:", self.b_q_max)

        q_range_group.setLayout(q_range_layout)
        layout.addWidget(q_range_group)

        # Info
        info = QLabel(
            "<b>Effect:</b> Re-calculate diffusion coefficient using only data points "
            "within the specified q² range. Individual linear fits remain unchanged."
        )
        info.setWordWrap(True)
        info.setStyleSheet("padding: 10px; background-color: #e3f2fd; border-radius: 4px;")
        layout.addWidget(info)

        tab.setLayout(layout)
        return tab

    def create_method_c_tab(self):
        """Create refinement tab for Method C with interactive plot"""
        tab = QWidget()
        layout = QVBoxLayout()

        # Add interactive Γ vs q² plot if we have Method C data
        if hasattr(self.analyzer, 'method_c_data') and self.analyzer.method_c_data is not None:
            self.plot_widget_c = InteractivePlotWidget(
                self.analyzer.method_c_data,
                'best_b',  # Gamma column for Method C (from non-linear fit)
                'q^2',
                'Method C',
                self
            )
            layout.addWidget(self.plot_widget_c)

        # Q² range group
        q_range_group = QGroupBox("Q² Range for Diffusion Coefficient Fit")
        q_range_layout = QFormLayout()

        # Get current q² range from basedata
        if hasattr(self.analyzer, 'df_basedata') and self.analyzer.df_basedata is not None:
            q_squared_values = self.analyzer.df_basedata['q^2'].values
            min_q = float(q_squared_values.min())
            max_q = float(q_squared_values.max())
        else:
            min_q = 0.0
            max_q = 0.01

        # Min Q²
        self.c_q_min = QDoubleSpinBox()
        self.c_q_min.setRange(0, 1)
        self.c_q_min.setValue(min_q)
        self.c_q_min.setDecimals(6)
        self.c_q_min.setSingleStep(0.0001)
        self.c_q_min.setSuffix(" nm⁻²")
        q_range_layout.addRow("Min Q²:", self.c_q_min)

        # Max Q²
        self.c_q_max = QDoubleSpinBox()
        self.c_q_max.setRange(0, 1)
        self.c_q_max.setValue(max_q)
        self.c_q_max.setDecimals(6)
        self.c_q_max.setSingleStep(0.0001)
        self.c_q_max.setSuffix(" nm⁻²")
        q_range_layout.addRow("Max Q²:", self.c_q_max)

        q_range_group.setLayout(q_range_layout)
        layout.addWidget(q_range_group)

        # Fit exclusion group
        exclusion_group = QGroupBox("Exclude Individual Fits")
        exclusion_layout = QVBoxLayout()

        self.inspect_fits_btn = QPushButton("Inspect & Exclude Fits...")
        self.inspect_fits_btn.clicked.connect(self.open_fit_inspector_c)
        exclusion_layout.addWidget(self.inspect_fits_btn)

        self.excluded_label_c = QLabel("No fits excluded")
        self.excluded_label_c.setStyleSheet("font-style: italic; color: #666;")
        exclusion_layout.addWidget(self.excluded_label_c)

        exclusion_group.setLayout(exclusion_layout)
        layout.addWidget(exclusion_group)

        # Info
        info = QLabel(
            "<b>Effect:</b> Re-calculate diffusion coefficient using only data points "
            "within the specified q² range AND excluding selected fits. "
            "Individual non-linear fits remain unchanged."
        )
        info.setWordWrap(True)
        info.setStyleSheet("padding: 10px; background-color: #e3f2fd; border-radius: 4px;")
        layout.addWidget(info)

        layout.addStretch()
        tab.setLayout(layout)
        return tab

    def open_fit_inspector_c(self):
        """Open fit inspection dialog for Method C"""
        from gui.dialogs.method_c_postfit_dialog import MethodCPostFitDialog

        if not hasattr(self.analyzer, 'method_c_plots') or not self.analyzer.method_c_plots:
            QMessageBox.warning(
                self,
                "No Plots Available",
                "No Method C fit plots are available for inspection."
            )
            return

        dialog = MethodCPostFitDialog(
            self.analyzer.method_c_plots,
            self.analyzer.method_c_fit_quality,
            self.excluded_fits_c,
            self
        )

        if dialog.exec_() == QDialog.Accepted:
            self.excluded_fits_c = dialog.get_excluded_files()

            # Update label
            if self.excluded_fits_c:
                self.excluded_label_c.setText(f"{len(self.excluded_fits_c)} fits excluded")
                self.excluded_label_c.setStyleSheet("font-weight: bold; color: #d32f2f;")
            else:
                self.excluded_label_c.setText("No fits excluded")
                self.excluded_label_c.setStyleSheet("font-style: italic; color: #666;")

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
            if hasattr(self.analyzer, 'df_basedata') and self.analyzer.df_basedata is not None:
                q_squared_values = self.analyzer.df_basedata['q^2'].values
                min_q = float(q_squared_values.min())
                max_q = float(q_squared_values.max())
            else:
                min_q = 0.0
                max_q = 0.01

            # Reset Method A
            if 'A' in self.available_methods:
                self.a_q_min.setValue(min_q)
                self.a_q_max.setValue(max_q)

            # Reset Method B
            if 'B' in self.available_methods:
                self.b_q_min.setValue(min_q)
                self.b_q_max.setValue(max_q)

            # Reset Method C
            if 'C' in self.available_methods:
                self.c_q_min.setValue(min_q)
                self.c_q_max.setValue(max_q)
                self.excluded_fits_c = []
                self.excluded_label_c.setText("No fits excluded")
                self.excluded_label_c.setStyleSheet("font-style: italic; color: #666;")

    def update_q_range_from_plot(self, q_min, q_max, method_name):
        """
        Update spinboxes when user selects range in plot

        Args:
            q_min: Minimum q² value
            q_max: Maximum q² value
            method_name: Which method's plot was used ('Method A', 'Method B', 'Method C')
        """
        if method_name == 'Method A':
            self.a_q_min.setValue(q_min)
            self.a_q_max.setValue(q_max)
        elif method_name == 'Method B':
            self.b_q_min.setValue(q_min)
            self.b_q_max.setValue(q_max)
        elif method_name == 'Method C':
            self.c_q_min.setValue(q_min)
            self.c_q_max.setValue(q_max)

    def apply_refinement(self):
        """Apply refinement and recalculate results"""
        # Validate inputs
        for method in self.available_methods:
            if method == 'A':
                if self.a_q_min.value() >= self.a_q_max.value():
                    QMessageBox.warning(
                        self,
                        "Invalid Range",
                        "Method A: Minimum q² must be less than maximum q²."
                    )
                    return
                self.q_ranges['A'] = (self.a_q_min.value(), self.a_q_max.value())

            elif method == 'B':
                if self.b_q_min.value() >= self.b_q_max.value():
                    QMessageBox.warning(
                        self,
                        "Invalid Range",
                        "Method B: Minimum q² must be less than maximum q²."
                    )
                    return
                self.q_ranges['B'] = (self.b_q_min.value(), self.b_q_max.value())

            elif method == 'C':
                if self.c_q_min.value() >= self.c_q_max.value():
                    QMessageBox.warning(
                        self,
                        "Invalid Range",
                        "Method C: Minimum q² must be less than maximum q²."
                    )
                    return
                self.q_ranges['C'] = (self.c_q_min.value(), self.c_q_max.value())

        # Accept dialog
        self.accept()

    def get_refinement_params(self):
        """
        Get refinement parameters

        Returns:
            dict: Dictionary with refinement parameters for each method
        """
        return {
            'q_ranges': self.q_ranges,
            'excluded_fits_c': self.excluded_fits_c
        }
