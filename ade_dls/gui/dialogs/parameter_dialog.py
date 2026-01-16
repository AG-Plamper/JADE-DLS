"""
Parameter Editor Dialog
Allows users to view and edit analysis parameters
"""

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
                             QLineEdit, QSpinBox, QDoubleSpinBox, QCheckBox,
                             QPushButton, QLabel, QTabWidget, QWidget,
                             QGroupBox, QComboBox)
from PyQt5.QtCore import Qt


class ParameterDialog(QDialog):
    """
    Dialog for editing analysis parameters

    Provides transparent control over all analysis settings
    """

    def __init__(self, analysis_type, current_params=None, parent=None):
        super().__init__(parent)
        self.analysis_type = analysis_type
        self.current_params = current_params or {}
        self.init_ui()

    def init_ui(self):
        """Initialize the UI"""
        self.setWindowTitle(f"Parameters: {self.analysis_type}")
        self.setMinimumWidth(500)

        layout = QVBoxLayout()

        # Title
        title = QLabel(f"Configure {self.analysis_type} Parameters")
        title.setStyleSheet("font-size: 14pt; font-weight: bold;")
        layout.addWidget(title)

        # Description
        desc = QLabel("Adjust parameters for your analysis. Hover over fields for help.")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Parameter form
        self.param_widgets = {}
        form = self.create_parameter_form()
        layout.addWidget(form)

        # Buttons
        button_layout = QHBoxLayout()

        reset_btn = QPushButton("Reset to Default")
        reset_btn.clicked.connect(self.reset_to_default)
        button_layout.addWidget(reset_btn)

        button_layout.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        ok_btn = QPushButton("OK")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(self.accept)
        button_layout.addWidget(ok_btn)

        layout.addLayout(button_layout)

        self.setLayout(layout)

    def create_parameter_form(self):
        """Create parameter form based on analysis type"""
        if self.analysis_type == "Cumulant C":
            return self.create_cumulant_c_form()
        elif self.analysis_type == "Regularized":
            return self.create_regularized_form()
        elif self.analysis_type == "NNLS":
            return self.create_nnls_form()
        else:
            return self.create_generic_form()

    def create_cumulant_c_form(self):
        """Create form for Cumulant C parameters"""
        group = QGroupBox("Fit Parameters")
        form_layout = QFormLayout()

        # Fit order
        order_combo = QComboBox()
        order_combo.addItems(["2nd Order", "3rd Order", "4th Order"])
        order_combo.setCurrentIndex(2)  # Default to 4th
        order_combo.setToolTip("Higher orders capture more polydispersity")
        self.param_widgets['fit_order'] = order_combo
        form_layout.addRow("Fit Order:", order_combo)

        # Fit range
        range_label = QLabel("Fit Range (s):")
        range_layout = QHBoxLayout()

        range_min = QDoubleSpinBox()
        range_min.setDecimals(9)
        range_min.setRange(1e-10, 1)
        range_min.setValue(1e-9)
        range_min.setToolTip("Start of fit range in seconds")
        self.param_widgets['fit_range_min'] = range_min
        range_layout.addWidget(QLabel("Min:"))
        range_layout.addWidget(range_min)

        range_max = QDoubleSpinBox()
        range_max.setRange(0.1, 100)
        range_max.setValue(10)
        range_max.setToolTip("End of fit range in seconds")
        self.param_widgets['fit_range_max'] = range_max
        range_layout.addWidget(QLabel("Max:"))
        range_layout.addWidget(range_max)

        form_layout.addRow(range_label, range_layout)

        # Optimization method
        opt_combo = QComboBox()
        opt_combo.addItems(["Levenberg-Marquardt (lm)", "Trust Region (trf)", "Dogbox"])
        opt_combo.setToolTip("Optimization algorithm to use")
        self.param_widgets['optimization'] = opt_combo
        form_layout.addRow("Optimization:", opt_combo)

        # Adaptive parameters
        adaptive_check = QCheckBox("Use Adaptive Initial Parameters")
        adaptive_check.setChecked(True)
        adaptive_check.setToolTip("Automatically estimate initial parameters from data")
        self.param_widgets['adaptive'] = adaptive_check
        form_layout.addRow(adaptive_check)

        # Adaptation strategy
        strategy_combo = QComboBox()
        strategy_combo.addItems(["Individual", "Global", "Representative"])
        strategy_combo.setToolTip("How to adapt parameters across datasets")
        self.param_widgets['adaptation_strategy'] = strategy_combo
        form_layout.addRow("Adaptation Strategy:", strategy_combo)

        group.setLayout(form_layout)
        return group

    def create_regularized_form(self):
        """Create form for Regularized fit parameters"""
        group = QGroupBox("Regularization Parameters")
        form_layout = QFormLayout()

        # Decay times
        decay_label = QLabel("Decay Time Range:")
        decay_layout = QHBoxLayout()

        decay_min = QDoubleSpinBox()
        decay_min.setDecimals(9)
        decay_min.setRange(1e-10, 1)
        decay_min.setValue(1e-8)
        self.param_widgets['decay_min'] = decay_min
        decay_layout.addWidget(QLabel("Min:"))
        decay_layout.addWidget(decay_min)

        decay_max = QDoubleSpinBox()
        decay_max.setRange(0.1, 100)
        decay_max.setValue(1)
        self.param_widgets['decay_max'] = decay_max
        decay_layout.addWidget(QLabel("Max:"))
        decay_layout.addWidget(decay_max)

        form_layout.addRow(decay_label, decay_layout)

        # Number of points
        n_points = QSpinBox()
        n_points.setRange(50, 500)
        n_points.setValue(200)
        n_points.setToolTip("Number of decay time points (more = higher resolution)")
        self.param_widgets['n_points'] = n_points
        form_layout.addRow("Number of Points:", n_points)

        # Alpha value
        alpha = QDoubleSpinBox()
        alpha.setDecimals(4)
        alpha.setRange(0.001, 10)
        alpha.setValue(0.5)
        alpha.setToolTip("Regularization strength (higher = smoother)")
        self.param_widgets['alpha'] = alpha
        form_layout.addRow("Alpha (α):", alpha)

        # Normalize
        normalize_check = QCheckBox("Normalize Distribution")
        normalize_check.setChecked(True)
        self.param_widgets['normalize'] = normalize_check
        form_layout.addRow(normalize_check)

        # Peak detection
        prominence = QDoubleSpinBox()
        prominence.setDecimals(3)
        prominence.setRange(0.001, 1)
        prominence.setValue(0.01)
        prominence.setToolTip("Minimum peak prominence (lower = more sensitive)")
        self.param_widgets['prominence'] = prominence
        form_layout.addRow("Peak Prominence:", prominence)

        group.setLayout(form_layout)
        return group

    def create_nnls_form(self):
        """Create form for NNLS parameters"""
        group = QGroupBox("NNLS Parameters")
        form_layout = QFormLayout()

        # Similar to regularized but without alpha
        n_points = QSpinBox()
        n_points.setRange(50, 500)
        n_points.setValue(200)
        self.param_widgets['n_points'] = n_points
        form_layout.addRow("Number of Points:", n_points)

        prominence = QDoubleSpinBox()
        prominence.setDecimals(3)
        prominence.setRange(0.001, 1)
        prominence.setValue(0.05)
        self.param_widgets['prominence'] = prominence
        form_layout.addRow("Peak Prominence:", prominence)

        group.setLayout(form_layout)
        return group

    def create_generic_form(self):
        """Create generic parameter form"""
        group = QGroupBox("Parameters")
        layout = QVBoxLayout()
        layout.addWidget(QLabel("No specific parameters for this analysis."))
        group.setLayout(layout)
        return group

    def reset_to_default(self):
        """Reset all parameters to default values"""
        # Re-create the form with default values
        # This is a simplified version - in practice, you'd restore defaults
        pass

    def get_parameters(self):
        """Get current parameter values as dictionary"""
        params = {}

        for key, widget in self.param_widgets.items():
            if isinstance(widget, QSpinBox):
                params[key] = widget.value()
            elif isinstance(widget, QDoubleSpinBox):
                params[key] = widget.value()
            elif isinstance(widget, QCheckBox):
                params[key] = widget.isChecked()
            elif isinstance(widget, QComboBox):
                params[key] = widget.currentText()
            elif isinstance(widget, QLineEdit):
                params[key] = widget.text()

        return params
