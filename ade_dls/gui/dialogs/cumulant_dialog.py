"""
Cumulant Analysis Dialog
Allows user to select which cumulant methods to perform and configure parameters
"""

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGroupBox,
                             QCheckBox, QLabel, QLineEdit, QPushButton,
                             QComboBox, QDoubleSpinBox, QSpinBox, QFormLayout,
                             QTabWidget, QWidget, QRadioButton, QButtonGroup,
                             QMessageBox)
from PyQt5.QtCore import Qt
import numpy as np


class CumulantAnalysisDialog(QDialog):
    """
    Dialog for configuring Cumulant Analysis methods

    Three methods available:
    - Method A: Uses ALV software cumulant fit data
    - Method B: Simple linear fit method
    - Method C: Iterative non-linear fit method
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cumulant Analysis Configuration")
        self.setMinimumWidth(600)
        self.setMinimumHeight(500)

        # Results
        self.selected_methods = []
        self.method_a_params = {}
        self.method_b_params = {}
        self.method_c_params = {}

        self.init_ui()

    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout()

        # Method selection group
        method_group = QGroupBox("Select Cumulant Methods")
        method_layout = QVBoxLayout()

        # Method checkboxes
        self.method_a_check = QCheckBox("Method A: ALV Software Cumulant Data")
        self.method_a_check.setToolTip(
            "Uses cumulant fit data directly from ALV correlator software.\n"
            "Extracts 1st, 2nd, and 3rd order cumulants from .asc files."
        )

        self.method_b_check = QCheckBox("Method B: Linear Fit (Simplest Method)")
        self.method_b_check.setToolTip(
            "Uses linear fit method to fit ln[sqrt(g2)] vs time.\n"
            "Best for narrow time ranges and well-behaved data."
        )

        self.method_c_check = QCheckBox("Method C: Iterative Non-Linear Fit")
        self.method_c_check.setToolTip(
            "Uses iterative non-linear least squares fitting.\n"
            "Most comprehensive method, can fit up to 4th cumulant."
        )

        method_layout.addWidget(self.method_a_check)
        method_layout.addWidget(self.method_b_check)
        method_layout.addWidget(self.method_c_check)
        method_group.setLayout(method_layout)
        layout.addWidget(method_group)

        # Parameter tabs
        self.tabs = QTabWidget()

        # Method A parameters (minimal - mostly automatic)
        self.create_method_a_tab()

        # Method B parameters
        self.create_method_b_tab()

        # Method C parameters
        self.create_method_c_tab()

        layout.addWidget(self.tabs)

        # Buttons
        button_layout = QHBoxLayout()

        self.ok_button = QPushButton("Run Analysis")
        self.ok_button.clicked.connect(self.accept)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)

        button_layout.addStretch()
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)

        layout.addLayout(button_layout)

        self.setLayout(layout)

    def create_method_a_tab(self):
        """Create parameter tab for Method A"""
        tab = QWidget()
        layout = QVBoxLayout()

        info_label = QLabel(
            "<b>Method A: ALV Software Cumulant Data</b><br><br>"
            "This method extracts cumulant fit results directly from the ALV correlator "
            "software output in the .asc files.<br><br>"
            "Results include:<br>"
            "• 1st order cumulant fit<br>"
            "• 2nd order cumulant fit (with polydispersity index)<br>"
            "• 3rd order cumulant fit (with polydispersity index)"
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # q-range selection (common to all methods)
        q_range_group = QGroupBox("Diffusion Analysis q² Range (optional)")
        q_range_layout = QFormLayout()

        # Enable checkbox
        self.q_range_enabled = QCheckBox("Restrict q² range for Γ vs q² fit")
        q_range_layout.addRow(self.q_range_enabled)

        # Min/Max inputs
        q_limit_layout = QHBoxLayout()
        self.q_min = QDoubleSpinBox()
        self.q_min.setRange(0, 1000)
        self.q_min.setValue(0.0)
        self.q_min.setDecimals(6)
        self.q_min.setSuffix(" nm⁻²")
        self.q_min.setEnabled(False)

        self.q_max = QDoubleSpinBox()
        self.q_max.setRange(0, 1000)
        self.q_max.setValue(100.0)
        self.q_max.setDecimals(6)
        self.q_max.setSuffix(" nm⁻²")
        self.q_max.setEnabled(False)

        q_limit_layout.addWidget(QLabel("Min:"))
        q_limit_layout.addWidget(self.q_min)
        q_limit_layout.addWidget(QLabel("Max:"))
        q_limit_layout.addWidget(self.q_max)

        q_range_layout.addRow("q² Range:", q_limit_layout)

        # Info
        q_info = QLabel(
            "<i>Use this to exclude outliers at very low or high q² values<br>"
            "from the Diffusion Coefficient analysis (Γ vs q² linear fit).</i>"
        )
        q_info.setWordWrap(True)
        q_range_layout.addRow("", q_info)

        q_range_group.setLayout(q_range_layout)
        layout.addWidget(q_range_group)

        # Connect enable checkbox
        self.q_range_enabled.toggled.connect(self.q_min.setEnabled)
        self.q_range_enabled.toggled.connect(self.q_max.setEnabled)

        layout.addStretch()

        tab.setLayout(layout)
        self.tabs.addTab(tab, "Method A")

    def create_method_b_tab(self):
        """Create parameter tab for Method B"""
        tab = QWidget()
        layout = QFormLayout()

        # Fit limits
        fit_limits_layout = QHBoxLayout()
        self.b_fit_min = QDoubleSpinBox()
        self.b_fit_min.setRange(0, 100)
        self.b_fit_min.setValue(0.0)  # Start from 0 (as in original Jupyter notebook)
        self.b_fit_min.setDecimals(9)
        self.b_fit_min.setSingleStep(0.000001)
        self.b_fit_min.setSuffix(" s")

        self.b_fit_max = QDoubleSpinBox()
        self.b_fit_max.setRange(0, 100)
        self.b_fit_max.setValue(0.0002)  # 200 microseconds (as in original Jupyter notebook)
        self.b_fit_max.setDecimals(6)
        self.b_fit_max.setSingleStep(0.0001)
        self.b_fit_max.setSuffix(" s")

        fit_limits_layout.addWidget(QLabel("Min:"))
        fit_limits_layout.addWidget(self.b_fit_min)
        fit_limits_layout.addWidget(QLabel("Max:"))
        fit_limits_layout.addWidget(self.b_fit_max)

        layout.addRow("Fit Time Range:", fit_limits_layout)

        # Info label
        info_label = QLabel(
            "<i>Note: Keep fit limits very narrow for Method B.<br>"
            "Only the initial decay needs to be fitted.</i>"
        )
        info_label.setWordWrap(True)
        layout.addRow("", info_label)

        tab.setLayout(layout)
        self.tabs.addTab(tab, "Method B")

    def create_method_c_tab(self):
        """Create parameter tab for Method C"""
        tab = QWidget()
        layout = QFormLayout()

        # Fit limits
        fit_limits_layout = QHBoxLayout()
        self.c_fit_min = QDoubleSpinBox()
        self.c_fit_min.setRange(1e-9, 100)
        self.c_fit_min.setValue(1e-9)
        self.c_fit_min.setDecimals(12)
        self.c_fit_min.setSuffix(" s")

        self.c_fit_max = QDoubleSpinBox()
        self.c_fit_max.setRange(1e-9, 100)
        self.c_fit_max.setValue(10.0)
        self.c_fit_max.setDecimals(6)
        self.c_fit_max.setSuffix(" s")

        fit_limits_layout.addWidget(QLabel("Min:"))
        fit_limits_layout.addWidget(self.c_fit_min)
        fit_limits_layout.addWidget(QLabel("Max:"))
        fit_limits_layout.addWidget(self.c_fit_max)

        layout.addRow("Fit Time Range:", fit_limits_layout)

        # Fit function selection
        self.c_fit_function = QComboBox()
        self.c_fit_function.addItems([
            "2nd Cumulant (up to c parameter)",
            "3rd Cumulant (up to d parameter)",
            "4th Cumulant (up to e parameter) - Recommended"
        ])
        self.c_fit_function.setCurrentIndex(2)
        layout.addRow("Fit Function:", self.c_fit_function)

        # Adaptive initial guesses
        self.c_adaptive = QCheckBox("Use adaptive initial parameter guesses")
        self.c_adaptive.setChecked(True)
        self.c_adaptive.setToolTip(
            "Automatically determine good initial parameters for each dataset"
        )
        layout.addRow("", self.c_adaptive)

        # Adaptation strategy
        self.c_strategy = QComboBox()
        self.c_strategy.addItems([
            "Individual - Adapt for each dataset separately",
            "Global - Use median values across all datasets",
            "Representative - Use best signal-to-noise dataset"
        ])
        self.c_strategy.setCurrentIndex(0)
        layout.addRow("Adaptation Strategy:", self.c_strategy)

        # Optimization method
        self.c_optimizer = QComboBox()
        self.c_optimizer.addItems([
            "Levenberg-Marquardt (lm) - Recommended",
            "Trust Region Reflective (trf)",
            "Dogbox"
        ])
        self.c_optimizer.setCurrentIndex(0)
        layout.addRow("Optimization Method:", self.c_optimizer)

        # Initial parameters (advanced)
        init_params_group = QGroupBox("Initial Parameters (Advanced)")
        init_layout = QFormLayout()

        self.c_init_a = QDoubleSpinBox()
        self.c_init_a.setRange(0, 1)
        self.c_init_a.setValue(0.8)
        self.c_init_a.setDecimals(2)
        self.c_init_a.setSingleStep(0.1)
        init_layout.addRow("a (baseline):", self.c_init_a)

        self.c_init_b = QDoubleSpinBox()
        self.c_init_b.setRange(0, 1000000)
        self.c_init_b.setValue(10000)
        self.c_init_b.setDecimals(0)
        init_layout.addRow("b (decay rate):", self.c_init_b)

        self.c_init_c = QDoubleSpinBox()
        self.c_init_c.setRange(-1000, 1000)
        self.c_init_c.setValue(0)
        self.c_init_c.setDecimals(2)
        init_layout.addRow("c (2nd cumulant):", self.c_init_c)

        init_params_group.setLayout(init_layout)
        layout.addRow(init_params_group)

        # Processing Options
        processing_group = QGroupBox("Processing Options")
        processing_layout = QVBoxLayout()

        self.c_multiprocessing_check = QCheckBox("Use multiprocessing (faster for large datasets)")
        self.c_multiprocessing_check.setChecked(False)
        self.c_multiprocessing_check.setToolTip("Enable parallel processing for faster analysis (uses joblib for cross-platform support)")
        processing_layout.addWidget(self.c_multiprocessing_check)

        processing_group.setLayout(processing_layout)
        layout.addRow(processing_group)

        # Info
        info_label = QLabel(
            "<i>Note: Initial parameters are only used if adaptive guesses are disabled.<br>"
            "For most cases, keep adaptive guesses enabled.</i>"
        )
        info_label.setWordWrap(True)
        layout.addRow("", info_label)

        tab.setLayout(layout)
        self.tabs.addTab(tab, "Method C")

    def accept(self):
        """Validate and accept the dialog"""
        # Check if at least one method is selected
        if not (self.method_a_check.isChecked() or
                self.method_b_check.isChecked() or
                self.method_c_check.isChecked()):
            QMessageBox.warning(
                self,
                "No Method Selected",
                "Please select at least one cumulant analysis method."
            )
            return

        # Collect selected methods
        self.selected_methods = []
        if self.method_a_check.isChecked():
            self.selected_methods.append('A')
        if self.method_b_check.isChecked():
            self.selected_methods.append('B')
        if self.method_c_check.isChecked():
            self.selected_methods.append('C')

        # Collect q-range parameter (applies to all methods)
        self.q_range = None
        if self.q_range_enabled.isChecked():
            q_min_val = self.q_min.value()
            q_max_val = self.q_max.value()
            if q_min_val >= q_max_val:
                QMessageBox.warning(
                    self,
                    "Invalid q² Range",
                    f"Minimum q² ({q_min_val}) must be less than maximum ({q_max_val})."
                )
                return
            self.q_range = (q_min_val, q_max_val)

        # Collect parameters for Method B
        if 'B' in self.selected_methods:
            self.method_b_params = {
                'fit_limits': (self.b_fit_min.value(), self.b_fit_max.value())
            }

            # Validate
            min_time = self.method_b_params['fit_limits'][0]
            max_time = self.method_b_params['fit_limits'][1]
            if min_time >= max_time:
                QMessageBox.warning(
                    self,
                    "Invalid Fit Range",
                    f"Method B: Minimum fit time ({min_time:.6f} s) must be less than maximum ({max_time:.6f} s)."
                )
                return

        # Collect parameters for Method C
        if 'C' in self.selected_methods:
            # Map fit function selection to number
            fit_func_map = {0: 'fit_function2', 1: 'fit_function3', 2: 'fit_function4'}

            # Map optimizer selection
            optimizer_map = {0: 'lm', 1: 'trf', 2: 'dogbox'}

            # Map strategy selection
            strategy_map = {0: 'individual', 1: 'global', 2: 'representative'}

            self.method_c_params = {
                'fit_limits': (self.c_fit_min.value(), self.c_fit_max.value()),
                'fit_function': fit_func_map[self.c_fit_function.currentIndex()],
                'adaptive_initial_guesses': self.c_adaptive.isChecked(),
                'adaptation_strategy': strategy_map[self.c_strategy.currentIndex()],
                'optimizer': optimizer_map[self.c_optimizer.currentIndex()],
                'use_multiprocessing': self.c_multiprocessing_check.isChecked(),
                'initial_parameters': [
                    self.c_init_a.value(),
                    self.c_init_b.value(),
                    self.c_init_c.value(),
                    0,  # d
                    0,  # e
                    0   # f (baseline offset)
                ]
            }

            # Validate
            if self.method_c_params['fit_limits'][0] >= self.method_c_params['fit_limits'][1]:
                QMessageBox.warning(
                    self,
                    "Invalid Fit Range",
                    "Method C: Minimum fit time must be less than maximum."
                )
                return

        super().accept()

    def get_configuration(self):
        """
        Get the configuration selected by the user

        Returns:
            dict: Configuration dictionary with keys:
                - 'methods': list of selected methods ['A', 'B', 'C']
                - 'q_range': tuple (min, max) or None for q² restriction
                - 'method_a_params': dict of Method A parameters
                - 'method_b_params': dict of Method B parameters
                - 'method_c_params': dict of Method C parameters
        """
        return {
            'methods': self.selected_methods,
            'q_range': self.q_range,
            'method_a_params': self.method_a_params,
            'method_b_params': self.method_b_params,
            'method_c_params': self.method_c_params
        }
