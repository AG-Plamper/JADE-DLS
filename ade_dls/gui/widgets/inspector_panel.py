"""
Inspector Panel Widget
Shows code, parameters, and documentation for transparency
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QTabWidget, QTextEdit,
                             QTableWidget, QTableWidgetItem, QLabel,
                             QPushButton, QHBoxLayout, QGroupBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QSyntaxHighlighter, QTextCharFormat, QColor
import re


class PythonHighlighter(QSyntaxHighlighter):
    """Syntax highlighter for Python code"""

    def __init__(self, parent=None):
        super().__init__(parent)

        # Define highlighting rules
        self.highlighting_rules = []

        # Keywords
        keyword_format = QTextCharFormat()
        keyword_format.setForeground(QColor("#0000FF"))
        keyword_format.setFontWeight(QFont.Bold)
        keywords = [
            'def', 'class', 'import', 'from', 'return', 'if', 'else', 'elif',
            'for', 'while', 'try', 'except', 'with', 'as', 'lambda', 'yield',
            'True', 'False', 'None', 'and', 'or', 'not', 'in', 'is'
        ]
        for word in keywords:
            pattern = f'\\b{word}\\b'
            self.highlighting_rules.append((re.compile(pattern), keyword_format))

        # Strings
        string_format = QTextCharFormat()
        string_format.setForeground(QColor("#008000"))
        self.highlighting_rules.append((re.compile(r'"[^"\\]*(\\.[^"\\]*)*"'), string_format))
        self.highlighting_rules.append((re.compile(r"'[^'\\]*(\\.[^'\\]*)*'"), string_format))

        # Comments
        comment_format = QTextCharFormat()
        comment_format.setForeground(QColor("#808080"))
        comment_format.setFontItalic(True)
        self.highlighting_rules.append((re.compile(r'#[^\n]*'), comment_format))

        # Numbers
        number_format = QTextCharFormat()
        number_format.setForeground(QColor("#FF8C00"))
        self.highlighting_rules.append((re.compile(r'\b[0-9]+\.?[0-9]*\b'), number_format))

    def highlightBlock(self, text):
        """Apply syntax highlighting to a block of text"""
        for pattern, format in self.highlighting_rules:
            for match in pattern.finditer(text):
                self.setFormat(match.start(), match.end() - match.start(), format)


class InspectorPanel(QWidget):
    """
    Right panel showing code, parameters, and documentation
    Provides transparency into what the GUI is doing
    """

    def __init__(self, pipeline, parent=None):
        super().__init__(parent)
        self.pipeline = pipeline
        self.init_ui()

        # Connect to pipeline signals
        self.pipeline.step_added.connect(self.update_code_view)

    def init_ui(self):
        """Initialize the UI"""
        layout = QVBoxLayout()

        # Title
        title = QLabel("Inspector")
        title.setStyleSheet("font-size: 14pt; font-weight: bold;")
        layout.addWidget(title)

        # Create tab widget
        self.tabs = QTabWidget()

        # Tab 1: Code View
        self.code_tab = self.create_code_tab()
        self.tabs.addTab(self.code_tab, "💻 Code")

        # Tab 2: Parameters
        self.params_tab = self.create_params_tab()
        self.tabs.addTab(self.params_tab, "⚙️ Parameters")

        # Tab 3: Documentation
        self.docs_tab = self.create_docs_tab()
        self.tabs.addTab(self.docs_tab, "📖 Docs")

        layout.addWidget(self.tabs)

        # Export buttons
        export_layout = QHBoxLayout()

        self.copy_code_btn = QPushButton("📋 Copy Code")
        self.copy_code_btn.clicked.connect(self.copy_code)
        export_layout.addWidget(self.copy_code_btn)

        self.export_btn = QPushButton("💾 Export")
        self.export_btn.clicked.connect(self.export_code)
        export_layout.addWidget(self.export_btn)

        layout.addLayout(export_layout)

        self.setLayout(layout)

    def create_code_tab(self):
        """Create code viewer tab"""
        widget = QWidget()
        layout = QVBoxLayout()

        # Description
        desc = QLabel("Generated Python code for reproducibility:")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Code editor
        self.code_view = QTextEdit()
        self.code_view.setReadOnly(True)
        self.code_view.setFont(QFont("Courier", 10))
        self.code_view.setPlainText("# No analysis steps yet\n# Load data to begin")

        # Apply syntax highlighting
        self.highlighter = PythonHighlighter(self.code_view.document())

        layout.addWidget(self.code_view)

        widget.setLayout(layout)
        return widget

    def create_params_tab(self):
        """Create parameters tab"""
        widget = QWidget()
        layout = QVBoxLayout()

        # Description
        desc = QLabel("Current analysis parameters:")
        layout.addWidget(desc)

        # Parameters table
        self.params_table = QTableWidget()
        self.params_table.setColumnCount(3)
        self.params_table.setHorizontalHeaderLabels(["Step", "Parameter", "Value"])
        layout.addWidget(self.params_table)

        # Add some default parameters
        self.update_params_table()

        widget.setLayout(layout)
        return widget

    def create_docs_tab(self):
        """Create documentation tab"""
        widget = QWidget()
        layout = QVBoxLayout()

        # Documentation text
        self.docs_text = QTextEdit()
        self.docs_text.setReadOnly(True)
        self.docs_text.setHtml(self.get_default_docs())
        layout.addWidget(self.docs_text)

        widget.setLayout(layout)
        return widget

    def get_default_docs(self):
        """Get default documentation text"""
        return """
<h3>JADE-DLS Analysis Methods</h3>

<h4>Cumulant Methods</h4>
<p>Cumulant analysis fits the autocorrelation function to extract decay rates and polydispersity:</p>
<ul>
<li><b>Method A:</b> Uses cumulant fit data from ALV software (1st, 2nd, 3rd order)</li>
<li><b>Method B:</b> Simple linear fit of ln[g(τ)^0.5] vs τ</li>
<li><b>Method C:</b> Iterative non-linear least squares fit</li>
</ul>

<h4>Inverse Laplace Methods</h4>
<p>These methods solve the inverse problem to obtain size distributions:</p>
<ul>
<li><b>NNLS:</b> Non-Negative Least Squares without constraints</li>
<li><b>Regularized:</b> NNLS with Tikhonov-Phillips regularization</li>
</ul>

<h4>Key Equations</h4>
<p>Diffusion coefficient: Γ = D × q²</p>
<p>Hydrodynamic radius: R<sub>h</sub> = k<sub>B</sub>T / (6πηD)</p>
<p>Polydispersity Index: PDI = μ₂/Γ²</p>

<h4>Tips</h4>
<ul>
<li>For monodisperse samples, cumulant methods are faster and reliable</li>
<li>For polydisperse samples, use regularized inverse Laplace</li>
<li>Always check R² and residuals for fit quality</li>
</ul>
"""

    def update_code_view(self):
        """Update code view with current pipeline code"""
        code = self.pipeline.get_current_code()
        self.code_view.setPlainText(code)

    def update_params_table(self):
        """Update parameters table with current parameters"""
        param_history = self.pipeline.get_parameter_history()

        if not param_history.empty:
            self.params_table.setRowCount(len(param_history))

            for i, row in param_history.iterrows():
                self.params_table.setItem(i, 0, QTableWidgetItem(str(row['step'])))
                self.params_table.setItem(i, 1, QTableWidgetItem(str(row['parameter'])))
                self.params_table.setItem(i, 2, QTableWidgetItem(str(row['value'])))

            self.params_table.resizeColumnsToContents()
        else:
            # Show default parameters
            self.params_table.setRowCount(3)
            defaults = [
                ("General", "Temperature", "297.94 K"),
                ("General", "Viscosity", "0.894 cP"),
                ("Cumulant", "Fit Range", "1e-9 to 10 s")
            ]
            for i, (step, param, value) in enumerate(defaults):
                self.params_table.setItem(i, 0, QTableWidgetItem(step))
                self.params_table.setItem(i, 1, QTableWidgetItem(param))
                self.params_table.setItem(i, 2, QTableWidgetItem(value))

            self.params_table.resizeColumnsToContents()

    def update_for_step(self, step_name):
        """Update documentation for specific step"""
        docs_map = {
            'load_data': self.get_load_data_docs(),
            'preprocess': self.get_preprocess_docs(),
            'cumulant_a': self.get_cumulant_a_docs(),
            'cumulant_b': self.get_cumulant_b_docs(),
            'cumulant_c': self.get_cumulant_c_docs(),
            'nnls': self.get_nnls_docs(),
            'regularized': self.get_regularized_docs(),
        }

        docs = docs_map.get(step_name, self.get_default_docs())
        self.docs_text.setHtml(docs)

    def get_load_data_docs(self):
        return """
<h3>Load Data</h3>
<p>Load .asc files from ALV DLS instrument.</p>
<p><b>Expected format:</b> ALV-5000/E correlator .asc files</p>
<p><b>Files are filtered to exclude:</b> *averaged.asc files</p>
"""

    def get_preprocess_docs(self):
        return """
<h3>Preprocessing</h3>
<p>Extract metadata and correlation data from files.</p>
<p><b>Extracted data includes:</b></p>
<ul>
<li>Angle, Temperature, Wavelength</li>
<li>Refractive index, Viscosity</li>
<li>Count rates, Correlation functions</li>
</ul>
"""

    def get_cumulant_a_docs(self):
        return """
<h3>Cumulant Method A</h3>
<p>Uses pre-calculated cumulant fit data from ALV software.</p>
<p><b>Provides:</b> 1st, 2nd, and 3rd order cumulant fits</p>
<p><b>Best for:</b> Quick analysis when ALV software fits are reliable</p>
"""

    def get_cumulant_b_docs(self):
        return """
<h3>Cumulant Method B</h3>
<p>Linear fit method: ln[g(τ)^0.5] vs τ</p>
<p><b>Fit function:</b> 0.5·ln(a) - b·τ + 0.5·c·τ²</p>
<p><b>Best for:</b> Simple, fast analysis of monodisperse samples</p>
<p><b>Note:</b> Keep fit limits narrow (0 to 0.2 ms typical)</p>
"""

    def get_cumulant_c_docs(self):
        return """
<h3>Cumulant Method C</h3>
<p>Iterative non-linear least squares fit up to 4th cumulant.</p>
<p><b>Optimization methods:</b> Levenberg-Marquardt, TRF, Dogbox</p>
<p><b>Best for:</b> Accurate analysis of moderately polydisperse samples</p>
<p><b>Features:</b> Adaptive initial parameter guessing</p>
"""

    def get_nnls_docs(self):
        return """
<h3>NNLS Analysis</h3>
<p>Non-Negative Least Squares inverse Laplace transform.</p>
<p><b>Provides:</b> Size distribution without regularization</p>
<p><b>Best for:</b> Initial exploration of polydisperse samples</p>
<p><b>Note:</b> May be noisy without regularization</p>
"""

    def get_regularized_docs(self):
        return """
<h3>Regularized Fit</h3>
<p>NNLS with Tikhonov-Phillips regularization.</p>
<p><b>Key parameter:</b> α (alpha) controls regularization strength</p>
<p><b>Best for:</b> Polydisperse samples, smooth distributions</p>
<p><b>Tip:</b> Use alpha analyzer to find optimal α value</p>
"""

    def copy_code(self):
        """Copy code to clipboard"""
        from PyQt5.QtWidgets import QApplication
        clipboard = QApplication.clipboard()
        clipboard.setText(self.code_view.toPlainText())

    def export_code(self):
        """Export code (triggers main window export)"""
        # This would trigger the main window's export dialog
        pass

    def set_code_view_visible(self, visible):
        """Set code view tab visibility"""
        if visible:
            if self.tabs.indexOf(self.code_tab) == -1:
                self.tabs.insertTab(0, self.code_tab, "💻 Code")
        else:
            index = self.tabs.indexOf(self.code_tab)
            if index != -1:
                self.tabs.removeTab(index)

    def set_params_view_visible(self, visible):
        """Set parameters view tab visibility"""
        if visible:
            if self.tabs.indexOf(self.params_tab) == -1:
                self.tabs.insertTab(1, self.params_tab, "⚙️ Parameters")
        else:
            index = self.tabs.indexOf(self.params_tab)
            if index != -1:
                self.tabs.removeTab(index)
