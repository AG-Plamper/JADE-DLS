"""
NNLS Results Dialog
Displays NNLS analysis results with multiple peaks
"""

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGroupBox,
                             QLabel, QPushButton, QTableWidget, QTableWidgetItem,
                             QTabWidget, QWidget, QHeaderView, QMessageBox,
                             QFileDialog, QTextEdit)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QColor
import pandas as pd
import numpy as np


class NNLSResultsDialog(QDialog):
    """
    Dialog to display NNLS analysis results

    Shows:
        - Summary table with Rh values for all peaks
        - Detailed statistics
        - Export options
    """

    def __init__(self, laplace_analyzer, parent=None):
        super().__init__(parent)
        self.laplace_analyzer = laplace_analyzer

        self.init_ui()
        self.setWindowTitle("NNLS Results")
        self.resize(900, 700)

    def init_ui(self):
        """Initialize the UI"""
        layout = QVBoxLayout()

        # Title
        title = QLabel("NNLS Analysis Results")
        title.setStyleSheet("font-size: 14pt; font-weight: bold;")
        layout.addWidget(title)

        # Check if results exist
        if self.laplace_analyzer.nnls_final_results is None:
            error_label = QLabel("No NNLS results available.\nPlease run NNLS analysis first.")
            error_label.setStyleSheet("color: red; font-size: 12pt; padding: 20px;")
            error_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(error_label)

            close_btn = QPushButton("Close")
            close_btn.clicked.connect(self.reject)
            layout.addWidget(close_btn)

            self.setLayout(layout)
            return

        # Tabs
        tabs = QTabWidget()

        # Tab 1: Summary
        summary_tab = self.create_summary_tab()
        tabs.addTab(summary_tab, "📊 Summary")

        # Tab 2: Detailed Data
        data_tab = self.create_data_tab()
        tabs.addTab(data_tab, "📋 Detailed Data")

        # Tab 3: Statistics
        stats_tab = self.create_statistics_tab()
        tabs.addTab(stats_tab, "📈 Statistics")

        layout.addWidget(tabs)

        # Bottom buttons
        button_layout = QHBoxLayout()

        export_btn = QPushButton("📁 Export Results")
        export_btn.clicked.connect(self.export_results)
        button_layout.addWidget(export_btn)

        refine_btn = QPushButton("🔧 Refine Results")
        refine_btn.setToolTip("Remove outliers and recalculate")
        refine_btn.clicked.connect(self.refine_results)
        button_layout.addWidget(refine_btn)

        button_layout.addStretch()

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        button_layout.addWidget(close_btn)

        layout.addLayout(button_layout)

        self.setLayout(layout)

    def create_summary_tab(self):
        """Create summary results tab"""
        widget = QWidget()
        layout = QVBoxLayout()

        # Info box
        info_group = QGroupBox("Analysis Summary")
        info_layout = QVBoxLayout()

        num_datasets = len(self.laplace_analyzer.nnls_data) if self.laplace_analyzer.nnls_data is not None else 0
        num_peaks = len(self.laplace_analyzer.nnls_final_results)

        info_text = f"""
        <b>Method:</b> NNLS (Non-Negative Least Squares)<br>
        <b>Datasets analyzed:</b> {num_datasets}<br>
        <b>Peaks found:</b> {num_peaks}<br>
        <b>Prominence:</b> {self.laplace_analyzer.nnls_params.get('prominence', 'N/A')}<br>
        <b>Distance:</b> {self.laplace_analyzer.nnls_params.get('distance', 'N/A')}
        """

        info_label = QLabel(info_text)
        info_label.setWordWrap(True)
        info_layout.addWidget(info_label)

        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        # Results table
        results_group = QGroupBox("Hydrodynamic Radii")
        results_layout = QVBoxLayout()

        self.results_table = QTableWidget()
        self.populate_results_table()

        results_layout.addWidget(self.results_table)
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)

        # Interpretation
        interp_group = QGroupBox("Interpretation")
        interp_layout = QVBoxLayout()

        interpretation = self.generate_interpretation()
        interp_label = QLabel(interpretation)
        interp_label.setWordWrap(True)
        interp_label.setStyleSheet("padding: 10px; background-color: #f0f0f0; border-radius: 5px;")

        interp_layout.addWidget(interp_label)
        interp_group.setLayout(interp_layout)
        layout.addWidget(interp_group)

        widget.setLayout(layout)
        return widget

    def populate_results_table(self):
        """Populate the results table"""
        df = self.laplace_analyzer.nnls_final_results

        # Set up table
        self.results_table.setRowCount(len(df))
        self.results_table.setColumnCount(5)
        self.results_table.setHorizontalHeaderLabels([
            'Peak', 'Rh [nm]', 'Error [nm]', 'R²', 'D [m²/s]'
        ])

        # Populate data
        for i, row in df.iterrows():
            # Peak number
            peak_item = QTableWidgetItem(row['Fit'])
            peak_item.setFont(QFont('Arial', 10, QFont.Bold))
            self.results_table.setItem(i, 0, peak_item)

            # Rh
            rh_item = QTableWidgetItem(f"{row['Rh [nm]']:.2f}")
            rh_item.setTextAlignment(Qt.AlignCenter)
            self.results_table.setItem(i, 1, rh_item)

            # Error
            err_item = QTableWidgetItem(f"± {row['Rh error [nm]']:.2f}")
            err_item.setTextAlignment(Qt.AlignCenter)
            self.results_table.setItem(i, 2, err_item)

            # R²
            r2_item = QTableWidgetItem(f"{row['R_squared']:.4f}")
            r2_item.setTextAlignment(Qt.AlignCenter)

            # Color code based on R² quality
            if row['R_squared'] > 0.99:
                r2_item.setBackground(QColor('#90EE90'))  # Light green
            elif row['R_squared'] > 0.95:
                r2_item.setBackground(QColor('#FFE4B5'))  # Light orange
            else:
                r2_item.setBackground(QColor('#FFB6C1'))  # Light red

            self.results_table.setItem(i, 3, r2_item)

            # D
            d_item = QTableWidgetItem(f"{row['D [m^2/s]']:.2e}")
            d_item.setTextAlignment(Qt.AlignCenter)
            self.results_table.setItem(i, 4, d_item)

        # Adjust column widths
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    def create_data_tab(self):
        """Create detailed data tab"""
        widget = QWidget()
        layout = QVBoxLayout()

        # Full data table
        data_group = QGroupBox("Full Analysis Data")
        data_layout = QVBoxLayout()

        self.data_table = QTableWidget()
        self.populate_data_table()

        data_layout.addWidget(self.data_table)
        data_group.setLayout(data_layout)
        layout.addWidget(data_group)

        widget.setLayout(layout)
        return widget

    def populate_data_table(self):
        """Populate the detailed data table"""
        if self.laplace_analyzer.nnls_data is None:
            return

        df = self.laplace_analyzer.nnls_data

        # Select relevant columns
        display_cols = ['filename', 'angle [°]', 'q^2']

        # Add tau and gamma columns
        tau_cols = [col for col in df.columns if col.startswith('tau_')]
        gamma_cols = [col for col in df.columns if col.startswith('gamma_')]
        intensity_cols = [col for col in df.columns if col.startswith('intensity_')]
        percent_cols = [col for col in df.columns if 'percent' in col]

        all_cols = display_cols + tau_cols + gamma_cols + intensity_cols + percent_cols
        available_cols = [col for col in all_cols if col in df.columns]

        # Set up table
        self.data_table.setRowCount(len(df))
        self.data_table.setColumnCount(len(available_cols))
        self.data_table.setHorizontalHeaderLabels(available_cols)

        # Populate data
        for i, row in df.iterrows():
            for j, col in enumerate(available_cols):
                value = row[col]

                if pd.isna(value):
                    item = QTableWidgetItem("")
                elif isinstance(value, (int, np.integer)):
                    item = QTableWidgetItem(str(value))
                elif isinstance(value, (float, np.floating)):
                    if abs(value) < 0.01 or abs(value) > 1000:
                        item = QTableWidgetItem(f"{value:.3e}")
                    else:
                        item = QTableWidgetItem(f"{value:.4f}")
                else:
                    item = QTableWidgetItem(str(value))

                item.setTextAlignment(Qt.AlignCenter)
                self.data_table.setItem(i, j, item)

        # Adjust column widths
        self.data_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

    def create_statistics_tab(self):
        """Create statistics tab"""
        widget = QWidget()
        layout = QVBoxLayout()

        # Statistics text
        stats_group = QGroupBox("Statistical Summary")
        stats_layout = QVBoxLayout()

        stats_text = QTextEdit()
        stats_text.setReadOnly(True)
        stats_text.setHtml(self.generate_statistics_html())

        stats_layout.addWidget(stats_text)
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)

        widget.setLayout(layout)
        return widget

    def generate_statistics_html(self):
        """Generate HTML formatted statistics"""
        df = self.laplace_analyzer.nnls_final_results

        html = "<html><body style='font-family: Arial;'>"
        html += "<h2>NNLS Analysis Statistics</h2>"

        # For each peak
        for i, row in df.iterrows():
            html += f"<h3>{row['Fit']}</h3>"
            html += "<table border='1' cellpadding='5' cellspacing='0' style='border-collapse: collapse;'>"

            html += f"<tr><td><b>Rh [nm]</b></td><td>{row['Rh [nm]']:.2f} ± {row['Rh error [nm]']:.2f}</td></tr>"
            html += f"<tr><td><b>D [m²/s]</b></td><td>{row['D [m^2/s]']:.3e} ± {row['D error [m^2/s]']:.3e}</td></tr>"
            html += f"<tr><td><b>R²</b></td><td>{row['R_squared']:.6f}</td></tr>"
            html += f"<tr><td><b>Residuals Normality</b></td><td>{row.get('Residuals', 'N/A')}</td></tr>"

            html += "</table><br>"

        # Overall statistics
        html += "<h3>Overall Statistics</h3>"
        html += "<table border='1' cellpadding='5' cellspacing='0' style='border-collapse: collapse;'>"

        html += f"<tr><td><b>Number of Peaks</b></td><td>{len(df)}</td></tr>"
        html += f"<tr><td><b>Mean R²</b></td><td>{df['R_squared'].mean():.4f}</td></tr>"
        html += f"<tr><td><b>Min R²</b></td><td>{df['R_squared'].min():.4f}</td></tr>"

        html += "</table>"

        html += "</body></html>"

        return html

    def generate_interpretation(self):
        """Generate interpretation text"""
        df = self.laplace_analyzer.nnls_final_results
        num_peaks = len(df)

        if num_peaks == 0:
            return "⚠️ No peaks found. Consider adjusting prominence and distance parameters."
        elif num_peaks == 1:
            rh = df.iloc[0]['Rh [nm]']
            r2 = df.iloc[0]['R_squared']

            interpretation = f"✅ <b>Monodisperse sample detected</b><br>"
            interpretation += f"Single population with Rh = {rh:.2f} nm<br>"

            if r2 > 0.99:
                interpretation += "Excellent fit quality (R² > 0.99)"
            elif r2 > 0.95:
                interpretation += "Good fit quality (R² > 0.95)"
            else:
                interpretation += "⚠️ Moderate fit quality - consider refining parameters"

        elif num_peaks == 2:
            rh1 = df.iloc[0]['Rh [nm]']
            rh2 = df.iloc[1]['Rh [nm]']

            interpretation = f"✅ <b>Bidisperse sample detected</b><br>"
            interpretation += f"Two populations:<br>"
            interpretation += f"  • Peak 1: Rh = {rh1:.2f} nm<br>"
            interpretation += f"  • Peak 2: Rh = {rh2:.2f} nm<br>"
            interpretation += f"  • Size ratio: {rh2/rh1:.2f}"

        else:
            interpretation = f"✅ <b>Polydisperse sample detected</b><br>"
            interpretation += f"{num_peaks} distinct populations found:<br>"

            for i, row in df.iterrows():
                interpretation += f"  • Peak {i+1}: Rh = {row['Rh [nm]']:.2f} nm<br>"

            interpretation += "<br>💡 Multiple peaks indicate a heterogeneous sample"

        return interpretation

    def export_results(self):
        """Export results to file"""
        filename, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export NNLS Results",
            "",
            "Excel Files (*.xlsx);;CSV Files (*.csv);;Text Files (*.txt)"
        )

        if filename:
            try:
                self.laplace_analyzer.export_results(filename, method='nnls')
                QMessageBox.information(self, "Export Successful",
                                       f"Results exported to:\n{filename}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error",
                                    f"Error exporting results:\n{str(e)}")

    def refine_results(self):
        """Open refinement dialog"""
        from gui.dialogs.postfilter_dialog import PostFilterDialog

        dialog = PostFilterDialog(
            self.laplace_analyzer.nnls_data,
            'NNLS',
            parent=self
        )

        if dialog.exec_():
            indices_to_remove = dialog.get_selected_indices()
            if indices_to_remove:
                # Remove outliers
                self.laplace_analyzer.remove_nnls_outliers(indices_to_remove)

                # Recalculate
                self.laplace_analyzer.calculate_nnls_diffusion_coefficients()

                # Refresh display
                self.populate_results_table()
                self.populate_data_table()

                QMessageBox.information(self, "Refinement Complete",
                                       f"Removed {len(indices_to_remove)} outliers and recalculated results.")
