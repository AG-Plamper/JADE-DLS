"""
Method C Post-Fit Filtering Dialog
Allows visual inspection of non-linear fits and exclusion of bad fits
before recomputing the Diffusion Analysis
"""

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QListWidget, QListWidgetItem, QPushButton,
                             QSplitter, QWidget, QGroupBox, QMessageBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
import pandas as pd
import numpy as np


class MethodCPostFitDialog(QDialog):
    """
    Post-Fit Filtering Dialog for Method C

    Shows individual non-linear fits with quality metrics
    Allows exclusion of bad fits before recomputing diffusion analysis
    """

    def __init__(self, plots_dict, fit_quality, initially_excluded=None, parent=None):
        """
        Initialize the dialog

        Args:
            plots_dict: Dictionary {filename: (fig, data)} of Method C fits
            fit_quality: Dictionary {filename: {'R2': float, 'chi2': float, ...}}
            initially_excluded: List of filenames that are initially excluded (optional)
            parent: Parent widget
        """
        super().__init__(parent)

        self.plots_dict = plots_dict
        self.fit_quality = fit_quality

        # Track excluded files
        self.excluded_files = set(initially_excluded) if initially_excluded else set()

        # Current plot index
        self.current_index = 0
        self.filenames = list(plots_dict.keys())

        self.setWindowTitle("Method C - Post-Fit Filtering")
        self.setMinimumSize(1200, 800)

        self.init_ui()

        # Show first plot
        if self.filenames:
            self.file_list.setCurrentRow(0)

    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout()

        # Title
        title = QLabel("<h2>Post-Fit Filtering - Method C</h2>")
        layout.addWidget(title)

        info = QLabel(
            "Inspect individual non-linear fits and exclude bad fits.<br>"
            "The Diffusion Analysis will be recomputed using only the selected fits.<br>"
            "<b>Space:</b> Toggle selection | <b>Ctrl+X:</b> Exclude current | "
            "<b>Arrow Keys:</b> Navigate"
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        # Main splitter
        splitter = QSplitter(Qt.Horizontal)

        # Left panel: File list
        left_panel = self.create_file_list_panel()
        splitter.addWidget(left_panel)

        # Right panel: Plot viewer
        right_panel = self.create_plot_panel()
        splitter.addWidget(right_panel)

        # Set sizes (20%, 80%)
        splitter.setSizes([250, 950])

        layout.addWidget(splitter)

        # Bottom buttons
        button_layout = QHBoxLayout()

        self.select_all_btn = QPushButton("Select All")
        self.select_all_btn.clicked.connect(self.select_all)
        button_layout.addWidget(self.select_all_btn)

        self.deselect_all_btn = QPushButton("Deselect All")
        self.deselect_all_btn.clicked.connect(self.deselect_all)
        button_layout.addWidget(self.deselect_all_btn)

        button_layout.addStretch()

        self.stats_label = QLabel()
        self.update_stats_label()
        button_layout.addWidget(self.stats_label)

        button_layout.addStretch()

        self.recompute_btn = QPushButton("Recompute Diffusion Analysis")
        self.recompute_btn.setStyleSheet("font-weight: bold; padding: 8px;")
        self.recompute_btn.clicked.connect(self.recompute_diffusion)
        button_layout.addWidget(self.recompute_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)

        layout.addLayout(button_layout)

        self.setLayout(layout)

    def create_file_list_panel(self):
        """Create the file list panel"""
        widget = QWidget()
        layout = QVBoxLayout()

        label = QLabel(f"<b>Files ({len(self.filenames)} total)</b>")
        layout.addWidget(label)

        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QListWidget.SingleSelection)

        # Populate list
        for filename in self.filenames:
            quality = self.fit_quality.get(filename, {})
            chi2 = quality.get('chi2', quality.get('residuals', 'N/A'))
            r2 = quality.get('R2', quality.get('R_squared', 'N/A'))

            # Format display
            if isinstance(chi2, (int, float)):
                chi2_str = f"χ²={chi2:.4f}"
            else:
                chi2_str = f"χ²={chi2}"

            if isinstance(r2, (int, float)):
                r2_str = f"R²={r2:.3f}"
            else:
                r2_str = f"R²={r2}"

            item_text = f"{filename}\n   {chi2_str}, {r2_str}"
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, filename)
            # Check if file is initially excluded
            is_excluded = filename in self.excluded_files
            item.setCheckState(Qt.Unchecked if is_excluded else Qt.Checked)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            self.file_list.addItem(item)

        self.file_list.currentRowChanged.connect(self.on_file_selected)
        self.file_list.itemChanged.connect(self.on_item_checked)

        layout.addWidget(self.file_list)

        # Instructions
        instr = QLabel(
            "<i>Space: Toggle<br>"
            "Ctrl+X: Exclude</i>"
        )
        layout.addWidget(instr)

        widget.setLayout(layout)
        return widget

    def create_plot_panel(self):
        """Create the plot display panel"""
        widget = QWidget()
        layout = QVBoxLayout()

        # Current file label
        self.current_file_label = QLabel()
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        self.current_file_label.setFont(font)
        layout.addWidget(self.current_file_label)

        # Quality metrics label
        self.quality_label = QLabel()
        layout.addWidget(self.quality_label)

        # Plot canvas
        plot_group = QGroupBox("Non-Linear Fit")
        plot_layout = QVBoxLayout()

        # Matplotlib canvas
        try:
            from matplotlib.figure import Figure
            self.figure = Figure(figsize=(12, 5))
            self.canvas = FigureCanvasQTAgg(self.figure)
            self.toolbar = NavigationToolbar2QT(self.canvas, widget)

            plot_layout.addWidget(self.toolbar)
            plot_layout.addWidget(self.canvas)
        except ImportError:
            plot_layout.addWidget(QLabel("Matplotlib not available"))

        plot_group.setLayout(plot_layout)
        layout.addWidget(plot_group)

        widget.setLayout(layout)
        return widget

    def on_file_selected(self, index):
        """Handle file selection"""
        if index < 0 or index >= len(self.filenames):
            return

        self.current_index = index
        filename = self.filenames[index]

        # Update labels
        item = self.file_list.item(index)
        is_checked = item.checkState() == Qt.Checked
        status = "✓ Included" if is_checked else "✗ Excluded"

        self.current_file_label.setText(f"{filename} - {status}")

        # Update quality metrics
        quality = self.fit_quality.get(filename, {})
        quality_text = "<b>Fit Quality:</b> "
        if 'R2' in quality or 'R_squared' in quality:
            r2 = quality.get('R2', quality.get('R_squared'))
            quality_text += f"R² = {r2:.4f}  "
        if 'chi2' in quality:
            quality_text += f"χ² = {quality['chi2']:.4f}  "
        if 'residuals' in quality:
            quality_text += f"Residuals = {quality['residuals']:.4f}  "

        self.quality_label.setText(quality_text)

        # Show plot
        self.show_plot(filename)

    def show_plot(self, filename):
        """Display the plot for the given filename"""
        if filename not in self.plots_dict:
            return

        fig, data = self.plots_dict[filename]

        if fig is None:
            return

        # Clear and copy plot
        self.figure.clear()

        source_axes = fig.get_axes()

        for ax_idx, source_ax in enumerate(source_axes):
            # Create subplot in same position
            if len(source_axes) == 2:
                ax = self.figure.add_subplot(1, 2, ax_idx + 1)
            else:
                ax = self.figure.add_subplot(111)

            # Copy scatter plots (collections)
            for collection in source_ax.collections:
                offsets = collection.get_offsets()
                if len(offsets) > 0:
                    ax.scatter(offsets[:, 0], offsets[:, 1],
                             label=collection.get_label(),
                             c=collection.get_facecolors(),
                             s=collection.get_sizes(),
                             alpha=collection.get_alpha(),
                             edgecolors=collection.get_edgecolors(),
                             linewidths=collection.get_linewidths())

            # Copy lines
            for line in source_ax.get_lines():
                ax.plot(line.get_xdata(), line.get_ydata(),
                       label=line.get_label(),
                       color=line.get_color(),
                       linestyle=line.get_linestyle(),
                       marker=line.get_marker(),
                       markersize=line.get_markersize() if line.get_marker() else 1,
                       alpha=line.get_alpha())

            # Copy labels and title
            ax.set_xlabel(source_ax.get_xlabel())
            ax.set_ylabel(source_ax.get_ylabel())
            ax.set_title(source_ax.get_title())

            # Copy scale
            if source_ax.get_xscale() == 'log':
                ax.set_xscale('log')
            if source_ax.get_yscale() == 'log':
                ax.set_yscale('log')

            # Copy legend
            if source_ax.get_legend():
                ax.legend()

            # Copy grid
            x_gridlines = source_ax.xaxis.get_gridlines()
            grid_visible = any(line.get_visible() for line in x_gridlines) if x_gridlines else False
            ax.grid(grid_visible)

        self.figure.tight_layout()
        self.canvas.draw()

    def on_item_checked(self, item):
        """Handle checkbox state change"""
        filename = item.data(Qt.UserRole)
        is_checked = item.checkState() == Qt.Checked

        if is_checked:
            self.excluded_files.discard(filename)
        else:
            self.excluded_files.add(filename)

        self.update_stats_label()

        # Update current file label if this is the current file
        if self.filenames[self.current_index] == filename:
            self.on_file_selected(self.current_index)

    def select_all(self):
        """Select all files"""
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            item.setCheckState(Qt.Checked)
        self.excluded_files.clear()
        self.update_stats_label()

    def deselect_all(self):
        """Deselect all files"""
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            item.setCheckState(Qt.Unchecked)
        self.excluded_files = set(self.filenames)
        self.update_stats_label()

    def update_stats_label(self):
        """Update the statistics label"""
        included = len(self.filenames) - len(self.excluded_files)
        excluded = len(self.excluded_files)
        self.stats_label.setText(
            f"<b>Included:</b> {included} | <b>Excluded:</b> {excluded}"
        )

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts"""
        if event.key() == Qt.Key_Space:
            # Toggle current item
            current_item = self.file_list.currentItem()
            if current_item:
                new_state = Qt.Unchecked if current_item.checkState() == Qt.Checked else Qt.Checked
                current_item.setCheckState(new_state)

        elif event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_X:
            # Exclude current item
            current_item = self.file_list.currentItem()
            if current_item:
                current_item.setCheckState(Qt.Unchecked)

        else:
            super().keyPressEvent(event)

    def recompute_diffusion(self):
        """Recompute diffusion analysis with only selected files"""
        included_files = [f for f in self.filenames if f not in self.excluded_files]

        if len(included_files) < 3:
            QMessageBox.warning(
                self,
                "Insufficient Data",
                f"At least 3 data points are required for regression.\n"
                f"You have selected {len(included_files)} files.\n"
                f"Please include more files."
            )
            return

        # Ask for confirmation
        reply = QMessageBox.question(
            self,
            "Confirm Recompute",
            f"Recompute Diffusion Analysis with {len(included_files)} selected files?\n\n"
            f"The new results will be appended as 'Method C (filtered, N={len(included_files)})'.",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            # Store the filtered data
            self.filtered_results = {
                'included_files': included_files,
                'excluded_files': list(self.excluded_files)
            }
            self.accept()

    def get_filtered_results(self):
        """Get the filtering results"""
        if hasattr(self, 'filtered_results'):
            return self.filtered_results
        return None

    def get_excluded_files(self):
        """
        Get list of excluded filenames

        Returns:
            list: List of filenames that are excluded
        """
        return list(self.excluded_files)
