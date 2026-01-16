"""
Post-filtering dialog for cumulant methods B and C
Allows removal of bad fits after initial analysis
"""

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QTableWidget, QTableWidgetItem, QLineEdit,
                             QGroupBox, QTextEdit, QMessageBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
import pandas as pd


class PostFilterDialog(QDialog):
    """
    Dialog for post-filtering analysis results

    Allows users to remove bad fits by specifying row indices
    """

    def __init__(self, data_df, method_name="Method", parent=None):
        super().__init__(parent)
        self.data_df = data_df.copy()
        self.original_data = data_df.copy()
        self.method_name = method_name
        self.filtered_data = None

        self.setWindowTitle(f"Post-Filter {method_name} Results")
        self.setModal(True)
        self.resize(900, 600)

        self.init_ui()

    def init_ui(self):
        """Initialize the UI"""
        layout = QVBoxLayout()

        # Title
        title = QLabel(f"<h2>Post-Filter {self.method_name} Results</h2>")
        layout.addWidget(title)

        # Instructions
        instructions = QLabel(
            "Review the fitted data below and enter the row <b>indices</b> of bad fits to remove.<br>"
            "After filtering, the diffusion coefficient will be recalculated with the remaining data."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        # Data table
        table_group = QGroupBox("Current Fit Data")
        table_layout = QVBoxLayout()

        self.data_table = QTableWidget()
        self.populate_table()
        table_layout.addWidget(self.data_table)

        table_group.setLayout(table_layout)
        layout.addWidget(table_group)

        # Statistics
        self.stats_label = QLabel()
        self.update_stats()
        layout.addWidget(self.stats_label)

        # Input for indices
        input_group = QGroupBox("Filter Settings")
        input_layout = QVBoxLayout()

        input_label = QLabel("Enter row indices to <b>REMOVE</b> (comma-separated, e.g., 1,5,8):")
        input_layout.addWidget(input_label)

        self.indices_input = QLineEdit()
        self.indices_input.setPlaceholderText("e.g., 1,5,8")
        input_layout.addWidget(self.indices_input)

        input_group.setLayout(input_layout)
        layout.addWidget(input_group)

        # Buttons
        button_layout = QHBoxLayout()

        self.preview_btn = QPushButton("Preview Filter")
        self.preview_btn.clicked.connect(self.preview_filter)
        button_layout.addWidget(self.preview_btn)

        self.apply_btn = QPushButton("Apply Filter && Close")
        self.apply_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.apply_btn.clicked.connect(self.apply_filter)
        button_layout.addWidget(self.apply_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)

        layout.addLayout(button_layout)

        self.setLayout(layout)

    def populate_table(self):
        """Populate the data table"""
        # Select relevant columns to display
        display_cols = []
        if 'filename' in self.data_df.columns:
            display_cols.append('filename')
        if 'angle [°]' in self.data_df.columns:
            display_cols.append('angle [°]')
        if 'q^2' in self.data_df.columns:
            display_cols.append('q^2')

        # Method-specific columns
        if 'b' in self.data_df.columns:
            display_cols.extend(['b', 'c'])
        elif 'best_b' in self.data_df.columns:
            display_cols.extend(['best_b', 'best_c'])

        if 'R-squared' in self.data_df.columns:
            display_cols.append('R-squared')

        # Set up table
        self.data_table.setRowCount(len(self.data_df))
        self.data_table.setColumnCount(len(display_cols) + 1)
        self.data_table.setHorizontalHeaderLabels(['Index'] + display_cols)

        # Populate data
        for row_idx, (df_idx, row) in enumerate(self.data_df.iterrows()):
            # Index column
            index_item = QTableWidgetItem(str(df_idx))
            index_item.setFlags(index_item.flags() & ~Qt.ItemIsEditable)
            self.data_table.setItem(row_idx, 0, index_item)

            # Data columns
            for col_idx, col in enumerate(display_cols):
                value = row[col]
                if isinstance(value, float):
                    text = f"{value:.6f}"
                else:
                    text = str(value)

                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.data_table.setItem(row_idx, col_idx + 1, item)

        self.data_table.resizeColumnsToContents()

    def update_stats(self):
        """Update statistics label"""
        total = len(self.original_data)
        current = len(self.data_df)
        removed = total - current

        self.stats_label.setText(
            f"<b>Total datasets:</b> {total} | "
            f"<b>Current:</b> {current} | "
            f"<b>Removed:</b> {removed}"
        )

    def preview_filter(self):
        """Preview what will be filtered"""
        indices_str = self.indices_input.text().strip()

        if not indices_str:
            QMessageBox.warning(self, "No Input", "Please enter indices to remove.")
            return

        try:
            # Parse indices
            indices = [int(idx.strip()) for idx in indices_str.split(',')]

            # Check valid indices
            valid_indices = [idx for idx in indices if idx in self.data_df.index]
            invalid_indices = [idx for idx in indices if idx not in self.data_df.index]

            if invalid_indices:
                QMessageBox.warning(
                    self,
                    "Invalid Indices",
                    f"The following indices are not valid: {invalid_indices}"
                )
                return

            # Highlight rows to be removed
            for row_idx in range(self.data_table.rowCount()):
                df_index = int(self.data_table.item(row_idx, 0).text())

                if df_index in valid_indices:
                    # Mark for removal (red)
                    for col_idx in range(self.data_table.columnCount()):
                        self.data_table.item(row_idx, col_idx).setBackground(QColor(255, 200, 200))
                else:
                    # Keep (green)
                    for col_idx in range(self.data_table.columnCount()):
                        self.data_table.item(row_idx, col_idx).setBackground(QColor(200, 255, 200))

            QMessageBox.information(
                self,
                "Preview",
                f"Will remove {len(valid_indices)} rows.\n"
                f"{len(self.data_df) - len(valid_indices)} rows will remain."
            )

        except ValueError:
            QMessageBox.warning(
                self,
                "Invalid Input",
                "Please enter valid comma-separated integers."
            )

    def apply_filter(self):
        """Apply the filter and close"""
        indices_str = self.indices_input.text().strip()

        if not indices_str:
            # No filtering requested, just return original data
            self.filtered_data = self.data_df
            self.accept()
            return

        try:
            # Parse indices
            indices = [int(idx.strip()) for idx in indices_str.split(',')]

            # Filter data
            self.filtered_data = self.data_df.drop(indices, errors='ignore')
            self.filtered_data = self.filtered_data.reset_index(drop=True)
            self.filtered_data.index = self.filtered_data.index + 1

            if len(self.filtered_data) == len(self.data_df):
                QMessageBox.warning(
                    self,
                    "No Changes",
                    "No valid indices were removed. Check your input."
                )
                return

            if len(self.filtered_data) < 2:
                QMessageBox.critical(
                    self,
                    "Too Few Points",
                    "Cannot filter: need at least 2 data points for analysis."
                )
                return

            self.accept()

        except ValueError:
            QMessageBox.warning(
                self,
                "Invalid Input",
                "Please enter valid comma-separated integers."
            )

    def get_filtered_data(self):
        """Get the filtered data"""
        return self.filtered_data


def show_postfilter_dialog(data_df, method_name, parent=None):
    """
    Convenience function to show post-filter dialog

    Args:
        data_df: DataFrame with analysis data
        method_name: Name of the method (e.g., "Method B")
        parent: Parent widget

    Returns:
        Filtered DataFrame or None if cancelled
    """
    dialog = PostFilterDialog(data_df, method_name, parent)
    result = dialog.exec_()

    if result == QDialog.Accepted:
        return dialog.get_filtered_data()
    else:
        return None
