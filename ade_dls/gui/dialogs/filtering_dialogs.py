"""
Data Filtering Dialogs
Interactive dialogs for filtering countrates and correlations
"""

import numpy as np
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QListWidget, QListWidgetItem, QSplitter,
                             QMessageBox, QCheckBox, QGroupBox, QAbstractItemView)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure


class CountrateFilterDialog(QDialog):
    """
    Dialog for visualizing and filtering countrate data

    Shows all countrate plots and allows user to select bad measurements
    for exclusion
    """

    def __init__(self, countrate_data, parent=None):
        super().__init__(parent)
        self.countrate_data = countrate_data
        self.original_data = countrate_data.copy()
        self.excluded_files = set()
        self.current_file = None

        self.setWindowTitle("Filter Count Rates")
        self.setModal(True)
        self.resize(1200, 700)

        self.init_ui()
        self.update_plot()

    def init_ui(self):
        """Initialize the user interface"""
        layout = QHBoxLayout()

        # Left panel: File list
        left_panel = self.create_file_list_panel()

        # Right panel: Plot
        right_panel = self.create_plot_panel()

        # Create splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([300, 900])

        layout.addWidget(splitter)
        self.setLayout(layout)

    def create_file_list_panel(self):
        """Create left panel with file list"""
        widget = QGroupBox("Files")
        layout = QVBoxLayout()

        # Info label
        info = QLabel(
            "Select files with bad count rates to exclude.\n"
            "Click on a file to view its count rate plot."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        # Keyboard shortcuts banner
        shortcuts_label = QLabel(
            "<b>Keyboard Shortcuts:</b><br>"
            "Ctrl+X = Toggle exclude/include current file"
        )
        shortcuts_label.setStyleSheet("background-color: #e8f4f8; padding: 8px; border: 1px solid #b8d4e8; border-radius: 4px;")
        shortcuts_label.setWordWrap(True)
        layout.addWidget(shortcuts_label)

        # File list
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QAbstractItemView.ExtendedSelection)

        for filename in sorted(self.countrate_data.keys()):
            item = QListWidgetItem(filename)
            item.setData(Qt.UserRole, filename)
            self.file_list.addItem(item)

        self.file_list.currentItemChanged.connect(self.on_file_selected)
        layout.addWidget(self.file_list)

        # Statistics
        self.stats_label = QLabel(f"Total files: {len(self.countrate_data)}\nExcluded: 0")
        layout.addWidget(self.stats_label)

        # Mark as bad button
        mark_bad_btn = QPushButton("❌ Mark Selected as Bad")
        mark_bad_btn.clicked.connect(self.mark_selected_as_bad)
        layout.addWidget(mark_bad_btn)

        # Unmark button
        unmark_btn = QPushButton("✓ Unmark Selected")
        unmark_btn.clicked.connect(self.unmark_selected)
        layout.addWidget(unmark_btn)

        # Accept/Cancel buttons
        button_layout = QHBoxLayout()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        accept_btn = QPushButton("Apply Filter")
        accept_btn.setDefault(True)
        accept_btn.clicked.connect(self.accept)
        button_layout.addWidget(accept_btn)

        layout.addLayout(button_layout)

        widget.setLayout(layout)
        return widget

    def create_plot_panel(self):
        """Create right panel with plot"""
        widget = QGroupBox("Count Rate Plot")
        layout = QVBoxLayout()

        # Matplotlib figure
        self.figure = Figure(figsize=(10, 6))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        widget.setLayout(layout)
        return widget

    def on_file_selected(self, current, previous):
        """Handle file selection"""
        if current:
            filename = current.data(Qt.UserRole)
            self.current_file = filename
            self.update_plot()

    def update_plot(self):
        """Update the plot"""
        self.figure.clear()

        if self.current_file and self.current_file in self.countrate_data:
            # Plot single file
            ax = self.figure.add_subplot(111)
            df = self.countrate_data[self.current_file]

            # Plot each detector slot
            if 'time [s]' in df.columns:
                for col in ['detectorslot 1', 'detectorslot 2', 'detectorslot 3', 'detectorslot 4']:
                    if col in df.columns and not df[col].isna().all():
                        ax.plot(df['time [s]'], df[col], label=col, alpha=0.7)

            ax.set_xlabel('Time [s]')
            ax.set_ylabel('Count Rate [kHz]')
            ax.set_title(f'Count Rate: {self.current_file}')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Mark if excluded
            if self.current_file in self.excluded_files:
                ax.text(0.5, 0.95, '❌ EXCLUDED',
                       transform=ax.transAxes,
                       ha='center', va='top',
                       bbox=dict(boxstyle='round', facecolor='red', alpha=0.5),
                       fontsize=14, fontweight='bold')

        self.canvas.draw()

    def mark_selected_as_bad(self):
        """Mark selected files as bad"""
        selected_items = self.file_list.selectedItems()

        for item in selected_items:
            filename = item.data(Qt.UserRole)
            self.excluded_files.add(filename)
            item.setForeground(Qt.red)
            item.setText(f"❌ {filename}")

        self.update_stats()
        self.update_plot()

    def unmark_selected(self):
        """Unmark selected files"""
        selected_items = self.file_list.selectedItems()

        for item in selected_items:
            filename = item.data(Qt.UserRole)
            if filename in self.excluded_files:
                self.excluded_files.remove(filename)
            item.setForeground(Qt.black)
            item.setText(filename)

        self.update_stats()
        self.update_plot()

    def update_stats(self):
        """Update statistics label"""
        total = len(self.countrate_data)
        excluded = len(self.excluded_files)
        remaining = total - excluded

        self.stats_label.setText(
            f"Total files: {total}\n"
            f"Excluded: {excluded}\n"
            f"Remaining: {remaining}"
        )

    def toggle_exclude_current(self):
        """Toggle exclusion status of current file"""
        current_item = self.file_list.currentItem()
        if current_item:
            filename = current_item.data(Qt.UserRole)
            if filename in self.excluded_files:
                # Unmark
                self.excluded_files.remove(filename)
                current_item.setForeground(Qt.black)
                current_item.setText(filename)
            else:
                # Mark as bad
                self.excluded_files.add(filename)
                current_item.setForeground(Qt.red)
                current_item.setText(f"❌ {filename}")

            self.update_stats()
            self.update_plot()

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts"""
        # Ctrl+X: Toggle exclude/include current file
        if event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_X:
            self.toggle_exclude_current()
        else:
            super().keyPressEvent(event)

    def get_filtered_data(self):
        """Get filtered countrate data (excluding marked files)"""
        return {k: v for k, v in self.countrate_data.items()
                if k not in self.excluded_files}


class CorrelationFilterDialog(QDialog):
    """
    Dialog for visualizing and filtering correlation data

    Shows all correlation plots and allows user to select bad measurements
    for exclusion
    """

    def __init__(self, correlation_data, parent=None):
        super().__init__(parent)
        self.correlation_data = correlation_data
        self.original_data = correlation_data.copy()
        self.excluded_files = set()
        self.current_file = None

        self.setWindowTitle("Filter Correlations")
        self.setModal(True)
        self.resize(1200, 700)

        self.init_ui()
        self.update_plot()

    def init_ui(self):
        """Initialize the user interface"""
        layout = QHBoxLayout()

        # Left panel: File list
        left_panel = self.create_file_list_panel()

        # Right panel: Plot
        right_panel = self.create_plot_panel()

        # Create splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([300, 900])

        layout.addWidget(splitter)
        self.setLayout(layout)

    def create_file_list_panel(self):
        """Create left panel with file list"""
        widget = QGroupBox("Files")
        layout = QVBoxLayout()

        # Info label
        info = QLabel(
            "Select files with bad correlations to exclude.\n"
            "Click on a file to view its correlation plot."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        # Keyboard shortcuts banner
        shortcuts_label = QLabel(
            "<b>Keyboard Shortcuts:</b><br>"
            "Ctrl+X = Toggle exclude/include current file"
        )
        shortcuts_label.setStyleSheet("background-color: #e8f4f8; padding: 8px; border: 1px solid #b8d4e8; border-radius: 4px;")
        shortcuts_label.setWordWrap(True)
        layout.addWidget(shortcuts_label)

        # File list
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QAbstractItemView.ExtendedSelection)

        for filename in sorted(self.correlation_data.keys()):
            item = QListWidgetItem(filename)
            item.setData(Qt.UserRole, filename)
            self.file_list.addItem(item)

        self.file_list.currentItemChanged.connect(self.on_file_selected)
        layout.addWidget(self.file_list)

        # Statistics
        self.stats_label = QLabel(f"Total files: {len(self.correlation_data)}\nExcluded: 0")
        layout.addWidget(self.stats_label)

        # Mark as bad button
        mark_bad_btn = QPushButton("❌ Mark Selected as Bad")
        mark_bad_btn.clicked.connect(self.mark_selected_as_bad)
        layout.addWidget(mark_bad_btn)

        # Unmark button
        unmark_btn = QPushButton("✓ Unmark Selected")
        unmark_btn.clicked.connect(self.unmark_selected)
        layout.addWidget(unmark_btn)

        # Accept/Cancel buttons
        button_layout = QHBoxLayout()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        accept_btn = QPushButton("Apply Filter")
        accept_btn.setDefault(True)
        accept_btn.clicked.connect(self.accept)
        button_layout.addWidget(accept_btn)

        layout.addLayout(button_layout)

        widget.setLayout(layout)
        return widget

    def create_plot_panel(self):
        """Create right panel with plot"""
        widget = QGroupBox("Correlation Plot (normalized g²-1)")
        layout = QVBoxLayout()

        # Matplotlib figure
        self.figure = Figure(figsize=(10, 6))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        widget.setLayout(layout)
        return widget

    def on_file_selected(self, current, previous):
        """Handle file selection"""
        if current:
            filename = current.data(Qt.UserRole)
            self.current_file = filename
            self.update_plot()

    def update_plot(self):
        """Update the plot"""
        self.figure.clear()

        if self.current_file and self.current_file in self.correlation_data:
            # Plot single file
            ax = self.figure.add_subplot(111)
            df = self.correlation_data[self.current_file]

            # Plot normalized correlations
            if 'time [ms]' in df.columns:
                for col in ['correlation 1', 'correlation 2', 'correlation 3', 'correlation 4']:
                    if col in df.columns and not df[col].isna().all():
                        # Normalize: g(2) - 1
                        correlation = df[col].values
                        if len(correlation) > 0 and not np.all(np.isnan(correlation)):
                            ax.semilogx(df['time [ms]'], correlation, label=col, alpha=0.7)

            ax.set_xlabel('Time [ms]')
            ax.set_ylabel('g² - 1')
            ax.set_title(f'Correlation: {self.current_file}')
            ax.legend()
            ax.grid(True, alpha=0.3, which='both')
            ax.set_ylim(bottom=-0.1)

            # Mark if excluded
            if self.current_file in self.excluded_files:
                ax.text(0.5, 0.95, '❌ EXCLUDED',
                       transform=ax.transAxes,
                       ha='center', va='top',
                       bbox=dict(boxstyle='round', facecolor='red', alpha=0.5),
                       fontsize=14, fontweight='bold')

        self.canvas.draw()

    def mark_selected_as_bad(self):
        """Mark selected files as bad"""
        selected_items = self.file_list.selectedItems()

        for item in selected_items:
            filename = item.data(Qt.UserRole)
            self.excluded_files.add(filename)
            item.setForeground(Qt.red)
            item.setText(f"❌ {filename}")

        self.update_stats()
        self.update_plot()

    def unmark_selected(self):
        """Unmark selected files"""
        selected_items = self.file_list.selectedItems()

        for item in selected_items:
            filename = item.data(Qt.UserRole)
            if filename in self.excluded_files:
                self.excluded_files.remove(filename)
            item.setForeground(Qt.black)
            item.setText(filename)

        self.update_stats()
        self.update_plot()

    def update_stats(self):
        """Update statistics label"""
        total = len(self.correlation_data)
        excluded = len(self.excluded_files)
        remaining = total - excluded

        self.stats_label.setText(
            f"Total files: {total}\n"
            f"Excluded: {excluded}\n"
            f"Remaining: {remaining}"
        )

    def toggle_exclude_current(self):
        """Toggle exclusion status of current file"""
        current_item = self.file_list.currentItem()
        if current_item:
            filename = current_item.data(Qt.UserRole)
            if filename in self.excluded_files:
                # Unmark
                self.excluded_files.remove(filename)
                current_item.setForeground(Qt.black)
                current_item.setText(filename)
            else:
                # Mark as bad
                self.excluded_files.add(filename)
                current_item.setForeground(Qt.red)
                current_item.setText(f"❌ {filename}")

            self.update_stats()
            self.update_plot()

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts"""
        # Ctrl+X: Toggle exclude/include current file
        if event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_X:
            self.toggle_exclude_current()
        else:
            super().keyPressEvent(event)

    def get_filtered_data(self):
        """Get filtered correlation data (excluding marked files)"""
        return {k: v for k, v in self.correlation_data.items()
                if k not in self.excluded_files}
