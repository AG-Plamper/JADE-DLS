"""
Status Manager for detailed progress tracking
Provides real-time feedback to prevent "is it frozen?" confusion
"""

from PyQt5.QtCore import QObject, pyqtSignal, QTimer
from PyQt5.QtWidgets import QProgressDialog, QApplication
from datetime import datetime


class StatusManager(QObject):
    """
    Manages status messages and progress tracking

    Signals:
        status_changed: Emitted when status message changes
        progress_changed: Emitted when progress value changes
    """

    status_changed = pyqtSignal(str)
    progress_changed = pyqtSignal(int, int)  # current, maximum

    def __init__(self, status_bar=None):
        super().__init__()
        self.status_bar = status_bar
        self.current_operation = None
        self.start_time = None

        # Connect signals
        self.status_changed.connect(self._update_status_bar)

    def _update_status_bar(self, message):
        """Update the status bar with message"""
        if self.status_bar:
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.status_bar.showMessage(f"[{timestamp}] {message}")
            # Force GUI update
            QApplication.processEvents()

    def start_operation(self, operation_name):
        """Start a new operation"""
        self.current_operation = operation_name
        self.start_time = datetime.now()
        self.status_changed.emit(f"Starting: {operation_name}...")

    def update(self, message, progress=None, maximum=None):
        """
        Update status with detailed message

        Args:
            message: Status message
            progress: Current progress value (optional)
            maximum: Maximum progress value (optional)
        """
        if self.current_operation:
            full_message = f"{self.current_operation}: {message}"
        else:
            full_message = message

        self.status_changed.emit(full_message)

        if progress is not None and maximum is not None:
            self.progress_changed.emit(progress, maximum)

    def complete_operation(self, message=None):
        """Complete current operation"""
        if self.start_time:
            elapsed = (datetime.now() - self.start_time).total_seconds()

            if message:
                final_message = f"{message} (completed in {elapsed:.1f}s)"
            else:
                final_message = f"{self.current_operation} completed in {elapsed:.1f}s"

            self.status_changed.emit(final_message)
        else:
            self.status_changed.emit(message or "Operation completed")

        self.current_operation = None
        self.start_time = None

    def error(self, message):
        """Report an error"""
        self.status_changed.emit(f"ERROR: {message}")
        self.current_operation = None
        self.start_time = None

    def ready(self):
        """Set status to ready"""
        self.status_changed.emit("Ready")
        self.current_operation = None


class ProgressDialog(QProgressDialog):
    """
    Enhanced progress dialog with detailed status
    """

    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setMinimumDuration(500)  # Show after 500ms
        self.setAutoClose(True)
        self.setAutoReset(True)

        # Prevent cancellation by default
        self.setCancelButton(None)

        self.current_step = ""

    def update_status(self, message, current, maximum):
        """
        Update progress dialog

        Args:
            message: Current operation message
            current: Current progress value
            maximum: Maximum progress value
        """
        self.setLabelText(f"{message}\n\nProcessing {current} of {maximum}...")
        self.setMaximum(maximum)
        self.setValue(current)

        # Force GUI update
        QApplication.processEvents()

    def set_step(self, step_name):
        """Set current processing step"""
        self.current_step = step_name
        self.setLabelText(f"Step: {step_name}\n\nInitializing...")
        QApplication.processEvents()
