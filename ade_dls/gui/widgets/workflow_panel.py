"""
Workflow Panel Widget
Shows analysis steps and their status
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QListWidget, QListWidgetItem, QGroupBox)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QIcon, QColor


class WorkflowPanel(QWidget):
    """
    Left panel showing workflow steps

    Signals:
        step_selected: Emitted when a step is selected
        run_analysis: Emitted when user wants to run analysis
    """

    step_selected = pyqtSignal(str)
    run_analysis = pyqtSignal(str)

    def __init__(self, pipeline, parent=None):
        super().__init__(parent)
        self.pipeline = pipeline
        self.init_ui()

    def init_ui(self):
        """Initialize the UI"""
        layout = QVBoxLayout()

        # Title
        title = QLabel("Analysis Workflow")
        title.setStyleSheet("font-size: 14pt; font-weight: bold;")
        layout.addWidget(title)

        # Workflow steps list
        self.setup_workflow_list()
        layout.addWidget(self.workflow_group)

        # Control buttons
        self.setup_control_buttons()
        layout.addLayout(self.button_layout)

        # Stretch to push everything to top
        layout.addStretch()

        self.setLayout(layout)

    def setup_workflow_list(self):
        """Setup the workflow steps list"""
        self.workflow_group = QGroupBox("Steps:")
        workflow_layout = QVBoxLayout()

        self.step_list = QListWidget()
        self.step_list.itemClicked.connect(self.on_step_clicked)

        # Define workflow steps
        self.steps = [
            ("📂 Load Data", "load_data", "Load .asc data files"),
            ("🔍 Preprocess", "preprocess", "Extract and filter data"),
            ("📊 Cumulant A", "cumulant_a", "ALV software cumulant data"),
            ("📈 Cumulant B", "cumulant_b", "Linear cumulant fit"),
            ("📉 Cumulant C", "cumulant_c", "Iterative non-linear fit"),
            ("🔬 NNLS", "nnls", "Inverse Laplace NNLS"),
            ("⚙️ Regularized", "regularized", "Tikhonov-Phillips regularization"),
            ("📋 Compare", "compare", "Compare all methods")
        ]

        for icon_name, step_id, description in self.steps:
            item = QListWidgetItem(f"{icon_name}\n{description}")
            item.setData(Qt.UserRole, step_id)
            item.setToolTip(description)
            self.step_list.addItem(item)

        workflow_layout.addWidget(self.step_list)
        self.workflow_group.setLayout(workflow_layout)

    def setup_control_buttons(self):
        """Setup control buttons"""
        self.button_layout = QVBoxLayout()

        # Run All button
        self.run_all_btn = QPushButton("▶ Run All")
        self.run_all_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.run_all_btn.clicked.connect(self.on_run_all)
        self.button_layout.addWidget(self.run_all_btn)

        # Run Selected button
        self.run_selected_btn = QPushButton("▶ Run Selected")
        self.run_selected_btn.clicked.connect(self.on_run_selected)
        self.button_layout.addWidget(self.run_selected_btn)

        # Reset button
        self.reset_btn = QPushButton("⟲ Reset")
        self.reset_btn.clicked.connect(self.on_reset)
        self.button_layout.addWidget(self.reset_btn)

    def on_step_clicked(self, item):
        """Handle step click"""
        step_id = item.data(Qt.UserRole)
        self.step_selected.emit(step_id)

    def on_run_all(self):
        """Run all analysis steps"""
        self.run_analysis.emit("all")

    def on_run_selected(self):
        """Run selected step"""
        current_item = self.step_list.currentItem()
        if current_item:
            step_id = current_item.data(Qt.UserRole)
            self.run_analysis.emit(step_id)

    def on_reset(self):
        """Reset the workflow"""
        # Additional reset logic

    def activate_step(self, step_id):
        """Activate a specific step"""
        for i in range(self.step_list.count()):
            item = self.step_list.item(i)
            if item.data(Qt.UserRole) == step_id:
                self.step_list.setCurrentItem(item)
                self.on_step_clicked(item)
                break

    def mark_step_complete(self, step_id):
        """Mark a step as complete with visual indicator"""
        for i in range(self.step_list.count()):
            item = self.step_list.item(i)
            if item.data(Qt.UserRole) == step_id:
                # Add checkmark to text
                current_text = item.text()
                if "✓" not in current_text:
                    item.setText(f"✓ {current_text}")
                # Change background color - adapt to system theme
                from PyQt5.QtWidgets import QApplication
                palette = QApplication.palette()
                base_color = palette.base().color()

                # Check if dark mode (base color is dark)
                is_dark = base_color.lightness() < 128
                if is_dark:
                    # Dark mode: use dark green
                    item.setBackground(QColor(0, 80, 0))  # Dark green
                else:
                    # Light mode: use light green
                    item.setBackground(QColor("#E8F5E9"))  # Light green
                break
