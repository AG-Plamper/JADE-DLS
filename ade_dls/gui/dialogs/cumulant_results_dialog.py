"""
Cumulant Results Dialog
Displays cumulant analysis results with embedded plots and navigation
"""

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
                             QListWidget, QLabel, QSplitter, QTextEdit,
                             QGridLayout, QWidget, QScrollArea, QTableWidget,
                             QTableWidgetItem, QHeaderView)
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import pandas as pd
import numpy as np


class CumulantResultsDialog(QDialog):
    """
    Dialog for displaying cumulant analysis results

    Features:
    - Embedded matplotlib plots
    - Navigation through datasets
    - Grid view for all plots
    - Results table
    """

    def __init__(self, results_data: dict, parent=None):
        """
        Initialize the dialog

        Args:
            results_data: Dictionary with:
                - 'method_name': str (e.g., 'Method A')
                - 'individual_plots': dict {filename: (fig, data)}
                - 'summary_plot': matplotlib figure (Γ vs q²)
                - 'results_df': pandas DataFrame with final results
                - 'fit_quality': dict {filename: {'R2': float, 'residuals': float}}
        """
        super().__init__(parent)
        self.results_data = results_data
        self.method_name = results_data['method_name']
        self.individual_plots = results_data.get('individual_plots', {})
        self.summary_plot = results_data.get('summary_plot', None)
        self.results_df = results_data.get('results_df', pd.DataFrame())
        self.fit_quality = results_data.get('fit_quality', {})

        self.current_index = 0
        self.filenames = list(self.individual_plots.keys())

        self.setWindowTitle(f"Cumulant Analysis Results - {self.method_name}")
        self.setMinimumWidth(1000)
        self.setMinimumHeight(700)

        self.init_ui()

    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout()

        # Title
        title_label = QLabel(f"<h2>Cumulant Analysis: {self.method_name}</h2>")
        layout.addWidget(title_label)

        # Main splitter (horizontal)
        main_splitter = QSplitter(Qt.Horizontal)

        # Left panel: File list
        left_panel = self.create_file_list_panel()
        main_splitter.addWidget(left_panel)

        # Center panel: Plot area
        center_panel = self.create_plot_panel()
        main_splitter.addWidget(center_panel)

        # Set splitter sizes (20%, 80%)
        main_splitter.setSizes([200, 800])

        layout.addWidget(main_splitter)

        # Bottom: Results summary
        results_group = self.create_results_panel()
        layout.addWidget(results_group)

        # Buttons
        button_layout = QHBoxLayout()

        # Post-filter button (only for Methods B and C)
        if 'Method B' in self.method_name or 'Method C' in self.method_name:
            self.postfilter_btn = QPushButton("🔧 Post-Filter Results")
            self.postfilter_btn.setToolTip(
                "Remove bad fits and recalculate results with remaining data"
            )
            self.postfilter_btn.clicked.connect(self.open_postfilter_dialog)
            button_layout.addWidget(self.postfilter_btn)

        button_layout.addStretch()

        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.accept)
        button_layout.addWidget(self.close_btn)

        layout.addLayout(button_layout)

        self.setLayout(layout)

        # Show first plot
        if self.filenames:
            self.show_plot(0)

    def create_file_list_panel(self):
        """Create the left panel with file list"""
        widget = QWidget()
        layout = QVBoxLayout()

        label = QLabel(f"<b>Datasets ({len(self.filenames)})</b>")
        layout.addWidget(label)

        # File list
        self.file_list = QListWidget()
        for i, filename in enumerate(self.filenames):
            # Add quality indicator
            quality_str = ""
            if filename in self.fit_quality:
                r2 = self.fit_quality[filename].get('R2', 0)
                quality_str = f" (R²={r2:.3f})"

            self.file_list.addItem(f"{i+1}. {filename}{quality_str}")

        self.file_list.currentRowChanged.connect(self.on_file_selected)
        layout.addWidget(self.file_list)

        # Navigation buttons
        nav_layout = QHBoxLayout()

        self.prev_btn = QPushButton("◀ Previous")
        self.prev_btn.clicked.connect(self.show_previous)
        nav_layout.addWidget(self.prev_btn)

        self.next_btn = QPushButton("Next ▶")
        self.next_btn.clicked.connect(self.show_next)
        nav_layout.addWidget(self.next_btn)

        layout.addLayout(nav_layout)

        widget.setLayout(layout)
        return widget

    def create_plot_panel(self):
        """Create the center panel with plot"""
        widget = QWidget()
        layout = QVBoxLayout()

        # Info label
        self.info_label = QLabel()
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

        # Matplotlib canvas
        self.figure = plt.Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        # Fit info
        self.fit_info_label = QLabel()
        self.fit_info_label.setWordWrap(True)
        layout.addWidget(self.fit_info_label)

        widget.setLayout(layout)
        return widget

    def create_results_panel(self):
        """Create the bottom panel with results table"""
        widget = QWidget()
        layout = QVBoxLayout()

        label = QLabel("<b>Analysis Results Summary:</b>")
        layout.addWidget(label)

        # Create table
        self.results_table = QTableWidget()
        self.populate_results_table()
        layout.addWidget(self.results_table)

        widget.setLayout(layout)
        return widget

    def populate_results_table(self):
        """Populate the results table"""
        if self.results_df.empty:
            return

        self.results_table.setRowCount(len(self.results_df))
        self.results_table.setColumnCount(len(self.results_df.columns))
        self.results_table.setHorizontalHeaderLabels(self.results_df.columns)

        for i, row in self.results_df.iterrows():
            for j, value in enumerate(row):
                item = QTableWidgetItem(str(value))
                self.results_table.setItem(i, j, item)

        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    def on_file_selected(self, index):
        """Handle file selection from list"""
        if 0 <= index < len(self.filenames):
            self.show_plot(index)

    def show_plot(self, index):
        """Show plot for the given index"""
        if index < 0 or index >= len(self.filenames):
            return

        self.current_index = index
        filename = self.filenames[index]

        # Update list selection
        self.file_list.setCurrentRow(index)

        # Update info label
        self.info_label.setText(
            f"<b>Dataset {index + 1} of {len(self.filenames)}</b>: {filename}"
        )

        # Get plot data
        if filename in self.individual_plots:
            fig, data = self.individual_plots[filename]

            # Clear and redraw
            self.figure.clear()

            # Copy the plot
            if fig is not None:
                # Get all axes from the source figure
                source_axes = fig.get_axes()

                if source_axes:
                    # Create new axis in our figure
                    ax = self.figure.add_subplot(111)

                    # Copy plot content
                    source_ax = source_axes[0]

                    # Copy lines
                    for line in source_ax.get_lines():
                        ax.plot(line.get_xdata(), line.get_ydata(),
                               label=line.get_label(),
                               color=line.get_color(),
                               linestyle=line.get_linestyle(),
                               marker=line.get_marker(),
                               markersize=line.get_markersize(),
                               alpha=line.get_alpha())

                    # Copy labels and title
                    ax.set_xlabel(source_ax.get_xlabel())
                    ax.set_ylabel(source_ax.get_ylabel())
                    ax.set_title(source_ax.get_title())

                    # Copy legend if exists
                    if source_ax.get_legend():
                        ax.legend()

                    # Copy grid
                    ax.grid(source_ax.get_xgridlines() or source_ax.get_ygridlines())

            self.canvas.draw()

            # Update fit info
            if filename in self.fit_quality:
                quality = self.fit_quality[filename]
                info_text = f"<b>Fit Quality:</b> "
                info_text += f"R² = {quality.get('R2', 'N/A'):.4f}, "
                info_text += f"Residuals = {quality.get('residuals', 'N/A')}"
                self.fit_info_label.setText(info_text)
            else:
                self.fit_info_label.setText("")

        # Update button states
        self.prev_btn.setEnabled(index > 0)
        self.next_btn.setEnabled(index < len(self.filenames) - 1)

    def show_previous(self):
        """Show previous dataset"""
        if self.current_index > 0:
            self.show_plot(self.current_index - 1)

    def show_next(self):
        """Show next dataset"""
        if self.current_index < len(self.filenames) - 1:
            self.show_plot(self.current_index + 1)

    def open_postfilter_dialog(self):
        """Open post-filter dialog to remove bad fits"""
        from gui.dialogs.postfilter_dialog import show_postfilter_dialog
        from PyQt5.QtWidgets import QMessageBox

        # Get the fit data from results_data
        if 'fit_data' not in self.results_data:
            QMessageBox.warning(
                self,
                "No Data",
                "No fit data available for post-filtering.\n"
                "This feature requires the raw fit data."
            )
            return

        fit_data = self.results_data['fit_data']

        # Show post-filter dialog
        filtered_data = show_postfilter_dialog(fit_data, self.method_name, self)

        if filtered_data is not None and len(filtered_data) != len(fit_data):
            # Data was filtered - need to recalculate
            reply = QMessageBox.question(
                self,
                "Recalculate Results",
                f"Filter applied. {len(filtered_data)} datasets remain (removed {len(fit_data) - len(filtered_data)}).\n\n"
                "The parent window will recalculate results with the filtered data.\n"
                "This dialog will close.",
                QMessageBox.Yes | QMessageBox.Cancel
            )

            if reply == QMessageBox.Yes:
                # Store filtered data and signal that we need to recalculate
                self.results_data['filtered_data'] = filtered_data
                self.results_data['needs_recalculation'] = True
                self.accept()  # Close dialog so parent can recalculate


class SummaryPlotDialog(QDialog):
    """
    Dialog for showing summary plot (Γ vs q²)
    """

    def __init__(self, summary_fig, method_name: str, parent=None):
        super().__init__(parent)
        self.summary_fig = summary_fig
        self.method_name = method_name

        self.setWindowTitle(f"Summary Plot - {method_name}")
        self.setMinimumSize(800, 600)

        self.init_ui()

    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout()

        # Title
        title_label = QLabel(f"<h3>Diffusion Coefficient Analysis: {self.method_name}</h3>")
        layout.addWidget(title_label)

        # Canvas
        canvas = FigureCanvas(self.summary_fig)
        toolbar = NavigationToolbar(canvas, self)

        layout.addWidget(toolbar)
        layout.addWidget(canvas)

        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

        self.setLayout(layout)
