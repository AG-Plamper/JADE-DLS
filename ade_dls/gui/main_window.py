"""
JADE-DLS Main GUI Window
A transparent, user-friendly GUI for Dynamic Light Scattering analysis
"""

import numpy as np
import pandas as pd

from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QSplitter, QMenuBar, QAction, QStatusBar,
                             QMessageBox, QFileDialog, QTabWidget, QProgressBar)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QIcon

from .widgets.workflow_panel import WorkflowPanel
from .widgets.analysis_view import AnalysisView
from .widgets.inspector_panel import InspectorPanel
from .core.pipeline import TransparentPipeline
from .core.status_manager import StatusManager, ProgressDialog
from .core.data_loader import DataLoader
from .dialogs.filtering_dialogs import CountrateFilterDialog, CorrelationFilterDialog
from .dialogs.cumulant_dialog import CumulantAnalysisDialog
from .dialogs.nnls_dialog import NNLSDialog
from .dialogs.nnls_results_dialog import NNLSResultsDialog
from .dialogs.regularized_dialog import RegularizedDialog
from .analysis.cumulant_analyzer import CumulantAnalyzer
from .analysis.laplace_analyzer import LaplaceAnalyzer


class JADEDLSMainWindow(QMainWindow):
    """
    Main application window for JADE-DLS analysis

    Features:
    - Transparent analysis pipeline
    - Interactive visualization
    - Code export capabilities
    - Parameter tracking
    """

    def __init__(self):
        super().__init__()

        # Initialize the transparent pipeline
        self.pipeline = TransparentPipeline()

        # Initialize status manager
        self.status_manager = None  # Will be set after status bar is created

        # Initialize data loader
        self.data_loader = DataLoader()

        # Progress dialog for long operations
        self.progress_dialog = None

        # Data storage
        self.loaded_data = None

        # Setup UI
        self.init_ui()

        # Connect signals
        self.connect_signals()

    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("JADE-DLS - Dynamic Light Scattering Analysis")
        self.setGeometry(100, 100, 1400, 900)

        # Create central widget with splitter layout
        self.setup_central_widget()

        # Create menu bar
        self.create_menu_bar()

        # Create status bar
        self.statusBar().showMessage("Ready")

        # Initialize status manager (after status bar exists)
        self.status_manager = StatusManager(self.statusBar())

    def setup_central_widget(self):
        """Setup the main layout with three panels"""
        central_widget = QWidget()
        main_layout = QHBoxLayout()

        # Create main splitter
        self.splitter = QSplitter(Qt.Horizontal)

        # Left panel: Workflow steps
        self.workflow_panel = WorkflowPanel(self.pipeline)
        self.workflow_panel.setMinimumWidth(200)

        # Center panel: Analysis view with tabs
        self.analysis_view = AnalysisView(self.pipeline)
        self.analysis_view.setMinimumWidth(500)

        # Right panel: Inspector (code viewer, parameters)
        self.inspector_panel = InspectorPanel(self.pipeline)
        self.inspector_panel.setMinimumWidth(300)

        # Add panels to splitter
        self.splitter.addWidget(self.workflow_panel)
        self.splitter.addWidget(self.analysis_view)
        self.splitter.addWidget(self.inspector_panel)

        # Set initial sizes (20%, 50%, 30%)
        self.splitter.setSizes([280, 700, 420])

        main_layout.addWidget(self.splitter)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def create_menu_bar(self):
        """Create the application menu bar"""
        menubar = self.menuBar()

        # File Menu
        file_menu = menubar.addMenu('&File')

        load_action = QAction('&Load Data...', self)
        load_action.setShortcut('Ctrl+O')
        load_action.setStatusTip('Load DLS data files')
        load_action.triggered.connect(self.load_data)
        file_menu.addAction(load_action)

        file_menu.addSeparator()

        export_notebook_action = QAction('Export as &Jupyter Notebook...', self)
        export_notebook_action.setShortcut('Ctrl+J')
        export_notebook_action.setStatusTip('Export analysis as Jupyter notebook')
        export_notebook_action.triggered.connect(self.export_notebook)
        file_menu.addAction(export_notebook_action)

        export_script_action = QAction('Export as Python &Script...', self)
        export_script_action.setShortcut('Ctrl+P')
        export_script_action.setStatusTip('Export analysis as Python script')
        export_script_action.triggered.connect(self.export_script)
        file_menu.addAction(export_script_action)

        export_report_action = QAction('Export &Report (PDF)...', self)
        export_report_action.setStatusTip('Export complete analysis report')
        export_report_action.triggered.connect(self.export_report)
        file_menu.addAction(export_report_action)

        file_menu.addSeparator()

        exit_action = QAction('E&xit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.setStatusTip('Exit application')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Analysis Menu
        analysis_menu = menubar.addMenu('&Analysis')

        # Preprocessing submenu
        preprocess_action = QAction('&Preprocessing...', self)
        preprocess_action.setShortcut('Ctrl+P')
        preprocess_action.setStatusTip('Run preprocessing and filtering')
        preprocess_action.triggered.connect(self.run_preprocessing)
        analysis_menu.addAction(preprocess_action)

        filter_countrates_action = QAction('Filter &Count Rates...', self)
        filter_countrates_action.setStatusTip('Manually filter count rate data')
        filter_countrates_action.triggered.connect(self.filter_countrates_manual)
        analysis_menu.addAction(filter_countrates_action)

        filter_correlations_action = QAction('Filter C&orrelations...', self)
        filter_correlations_action.setStatusTip('Manually filter correlation data')
        filter_correlations_action.triggered.connect(self.filter_correlations_manual)
        analysis_menu.addAction(filter_correlations_action)

        analysis_menu.addSeparator()

        cumulant_action = QAction('&Cumulant Analysis...', self)
        cumulant_action.setStatusTip('Run cumulant fitting methods')
        cumulant_action.triggered.connect(self.run_cumulant_analysis)
        analysis_menu.addAction(cumulant_action)

        nnls_action = QAction('&NNLS Analysis...', self)
        nnls_action.setStatusTip('Run NNLS inverse Laplace transform')
        nnls_action.triggered.connect(self.run_nnls_analysis)
        analysis_menu.addAction(nnls_action)

        regularized_action = QAction('&Regularized Fit...', self)
        regularized_action.setStatusTip('Run regularized inverse Laplace transform')
        regularized_action.triggered.connect(self.run_regularized_analysis)
        analysis_menu.addAction(regularized_action)

        analysis_menu.addSeparator()

        compare_action = QAction('C&ompare Results...', self)
        compare_action.setStatusTip('Compare all analysis methods')
        compare_action.triggered.connect(self.compare_results)
        analysis_menu.addAction(compare_action)

        # View Menu
        view_menu = menubar.addMenu('&View')

        show_code_action = QAction('Show &Code', self, checkable=True)
        show_code_action.setChecked(True)
        show_code_action.setStatusTip('Show generated code')
        show_code_action.triggered.connect(self.toggle_code_view)
        view_menu.addAction(show_code_action)

        show_params_action = QAction('Show &Parameters', self, checkable=True)
        show_params_action.setChecked(True)
        show_params_action.setStatusTip('Show analysis parameters')
        show_params_action.triggered.connect(self.toggle_params_view)
        view_menu.addAction(show_params_action)

        view_menu.addSeparator()

        reset_layout_action = QAction('&Reset Layout', self)
        reset_layout_action.setStatusTip('Reset window layout to default')
        reset_layout_action.triggered.connect(self.reset_layout)
        view_menu.addAction(reset_layout_action)

        # Help Menu
        help_menu = menubar.addMenu('&Help')

        tutorial_action = QAction('&Tutorial', self)
        tutorial_action.setShortcut('F1')
        tutorial_action.setStatusTip('Show interactive tutorial')
        tutorial_action.triggered.connect(self.show_tutorial)
        help_menu.addAction(tutorial_action)

        docs_action = QAction('&Documentation', self)
        docs_action.setStatusTip('Open documentation')
        docs_action.triggered.connect(self.show_documentation)
        help_menu.addAction(docs_action)

        help_menu.addSeparator()

        about_action = QAction('&About JADE-DLS', self)
        about_action.setStatusTip('About this application')
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def connect_signals(self):
        """Connect signals between components"""
        # Connect workflow panel signals
        self.workflow_panel.step_selected.connect(self.on_step_selected)
        self.workflow_panel.run_analysis.connect(self.on_run_analysis)

        # Connect pipeline signals
        self.pipeline.step_added.connect(self.on_pipeline_step_added)
        self.pipeline.analysis_completed.connect(self.on_analysis_completed)

        # Connect data loader signals
        self.data_loader.progress.connect(self.on_load_progress)
        self.data_loader.data_loaded.connect(self.on_data_loaded)
        self.data_loader.error.connect(self.on_load_error)

    # ========== Slot Methods ==========

    def load_data(self):
        """Load DLS data files"""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Data Folder",
            "",
            QFileDialog.ShowDirsOnly
        )

        if folder:
            # Create and show progress dialog
            self.progress_dialog = ProgressDialog("Loading Data", self)
            self.progress_dialog.setRange(0, 5)
            self.progress_dialog.setValue(0)
            self.progress_dialog.show()

            # Update status
            self.status_manager.start_operation(f"Loading data from {folder}")

            # Start loading in background thread
            self.data_loader.load_data(
                folder,
                load_countrates=True,
                load_correlations=True
            )

    def export_notebook(self):
        """Export analysis as Jupyter notebook"""
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Jupyter Notebook",
            "analysis.ipynb",
            "Jupyter Notebooks (*.ipynb)"
        )

        if filename:
            try:
                self.pipeline.export_to_notebook(filename)
                QMessageBox.information(
                    self,
                    "Export Successful",
                    f"Analysis exported to:\n{filename}"
                )
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Export failed: {str(e)}")

    def export_script(self):
        """Export analysis as Python script"""
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Python Script",
            "analysis.py",
            "Python Scripts (*.py)"
        )

        if filename:
            try:
                self.pipeline.export_to_script(filename)
                QMessageBox.information(
                    self,
                    "Export Successful",
                    f"Analysis exported to:\n{filename}"
                )
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Export failed: {str(e)}")

    def export_report(self):
        """Export complete analysis report as PDF"""
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export PDF Report",
            "analysis_report.pdf",
            "PDF Files (*.pdf)"
        )

        if filename:
            try:
                # Pass cumulant analyzer if available
                analyzer = getattr(self, 'cumulant_analyzer', None)
                self.pipeline.export_to_pdf(filename, cumulant_analyzer=analyzer)
                QMessageBox.information(
                    self,
                    "Export Successful",
                    f"Report exported to:\n{filename}"
                )
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Export failed: {str(e)}")

    def run_cumulant_analysis(self):
        """Run cumulant analysis"""
        # Use filtered data from pipeline if available, otherwise use loaded_data
        # This ensures that filtering is respected!
        if hasattr(self.pipeline, 'data') and self.pipeline.data and 'correlations' in self.pipeline.data:
            working_data = self.pipeline.data
            data_source = "filtered"
            print(f"[Cumulant] Using filtered data from pipeline ({len(working_data.get('correlations', {}))} datasets)")
        elif self.loaded_data and 'correlations' in self.loaded_data:
            working_data = self.loaded_data
            data_source = "original"
            print(f"[Cumulant] Using original loaded data ({len(working_data.get('correlations', {}))} datasets)")
        else:
            QMessageBox.warning(
                self,
                "No Data",
                "Please load and preprocess data first (File > Load Data)"
            )
            return

        # Check if data folder is available
        if 'data_folder' not in working_data:
            QMessageBox.warning(
                self,
                "Missing Information",
                "Data folder information is missing. Please reload the data."
            )
            return

        # Inform user about data source
        if data_source == "filtered":
            num_datasets = len(working_data.get('correlations', {}))
            # Don't show dialog here - will be distracting before parameter dialog
            print(f"[Cumulant] Will analyze {num_datasets} filtered datasets")

        # Show configuration dialog
        dialog = CumulantAnalysisDialog(self)
        if dialog.exec_() == dialog.Accepted:
            config = dialog.get_configuration()

            # Start analysis
            self.status_manager.start_operation("Running cumulant analysis...")

            try:
                # Create analyzer
                analyzer = CumulantAnalyzer(
                    working_data,
                    working_data['data_folder']
                )

                # Prepare basedata
                self.status_manager.update("Preparing basedata...")
                analyzer.prepare_basedata()

                results = []

                # Get q-range if specified
                q_range = config.get('q_range', None)

                # Run selected methods
                if 'A' in config['methods']:
                    self.status_manager.update("Running Cumulant Method A...")
                    try:
                        result_a = analyzer.run_method_a(q_range=q_range)
                        results.append(('Method A', result_a))

                        # Add to pipeline
                        self._add_cumulant_step_to_pipeline('A', config)

                    except Exception as e:
                        QMessageBox.warning(
                            self,
                            "Method A Failed",
                            f"Cumulant Method A failed:\n{str(e)}\n\nContinuing with other methods..."
                        )

                if 'B' in config['methods']:
                    self.status_manager.update("Running Cumulant Method B...")
                    try:
                        result_b = analyzer.run_method_b(
                            config['method_b_params']['fit_limits'],
                            q_range=q_range
                        )
                        results.append(('Method B', result_b))

                        # Add to pipeline
                        self._add_cumulant_step_to_pipeline('B', config)

                    except Exception as e:
                        QMessageBox.warning(
                            self,
                            "Method B Failed",
                            f"Cumulant Method B failed:\n{str(e)}\n\nContinuing with other methods..."
                        )

                if 'C' in config['methods']:
                    self.status_manager.update("Running Cumulant Method C...")
                    try:
                        result_c = analyzer.run_method_c(config['method_c_params'], q_range=q_range)
                        results.append(('Method C', result_c))

                        # Add to pipeline
                        self._add_cumulant_step_to_pipeline('C', config)

                    except Exception as e:
                        QMessageBox.warning(
                            self,
                            "Method C Failed",
                            f"Cumulant Method C failed:\n{str(e)}"
                        )

                # Show results
                if results:
                    self.status_manager.complete_operation(
                        f"Cumulant analysis completed ({len(results)} methods)"
                    )

                    # Display results
                    self._display_cumulant_results(results, analyzer)

                    # Mark workflow step complete
                    self.workflow_panel.mark_step_complete('cumulant')

                else:
                    QMessageBox.warning(
                        self,
                        "Analysis Failed",
                        "All cumulant methods failed. Please check your data and parameters."
                    )

            except Exception as e:
                self.status_manager.error(f"Cumulant analysis failed: {str(e)}")
                QMessageBox.critical(
                    self,
                    "Analysis Error",
                    f"Cumulant analysis failed:\n\n{str(e)}"
                )

    def run_nnls_analysis(self):
        """Run NNLS analysis"""
        self.workflow_panel.activate_step('nnls')

    def compare_results(self):
        """Compare all analysis results"""
        self.analysis_view.show_comparison()

    def toggle_code_view(self, checked):
        """Toggle code viewer visibility"""
        self.inspector_panel.set_code_view_visible(checked)

    def toggle_params_view(self, checked):
        """Toggle parameters view visibility"""
        self.inspector_panel.set_params_view_visible(checked)

    def reset_layout(self):
        """Reset window layout to default"""
        self.splitter.setSizes([280, 700, 420])

    def show_tutorial(self):
        """Show interactive tutorial"""
        QMessageBox.information(
            self,
            "Tutorial",
            "Interactive tutorial coming soon!\n\n"
            "For now, use File > Load Data to get started."
        )

    def show_documentation(self):
        """Open documentation"""
        QMessageBox.information(
            self,
            "Documentation",
            "Documentation available at:\n"
            "https://github.com/your-repo/jade-dls/docs"
        )

    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self,
            "About JADE-DLS",
            "<h3>JADE-DLS</h3>"
            "<p><b>J</b>upyter-based <b>A</b>ngular <b>D</b>ependent "
            "<b>E</b>valuator for <b>D</b>ynamic <b>L</b>ight <b>S</b>cattering</p>"
            "<p>Version 2.0 (GUI Edition)</p>"
            "<p>A transparent, user-friendly tool for DLS data analysis</p>"
            "<p>© 2025 JADE-DLS Development Team</p>"
        )

    def on_step_selected(self, step_name):
        """Handle workflow step selection"""
        self.analysis_view.show_step(step_name)
        self.inspector_panel.update_for_step(step_name)

    def on_run_analysis(self, analysis_type):
        """Handle run analysis request from workflow panel"""
        if analysis_type == 'load_data':
            self.load_data()
        elif analysis_type == 'preprocess':
            self.run_preprocessing()
        elif analysis_type in ['cumulant_a', 'cumulant_b', 'cumulant_c']:
            # All cumulant methods go through the same dialog
            self.run_cumulant_analysis()
        elif analysis_type == 'nnls':
            self.run_nnls_analysis()
        elif analysis_type == 'regularized':
            self.run_regularized_analysis()
        elif analysis_type == 'compare':
            self.compare_results()
        elif analysis_type == 'all':
            # Run all steps sequentially
            self.run_preprocessing()
            # More steps will be added as they're implemented
        else:
            self.statusBar().showMessage(f"Running {analysis_type} analysis...")
            # Other analysis types will be implemented

    def on_pipeline_step_added(self, step_info):
        """Handle new pipeline step"""
        self.inspector_panel.update_code_view()

    def on_analysis_completed(self, results):
        """Handle analysis completion"""
        self.statusBar().showMessage("Analysis completed successfully")
        self.analysis_view.display_results(results)

    def on_load_progress(self, current, total, message):
        """Handle data loading progress updates"""
        # Update status manager
        self.status_manager.update(message, current, total)

        # Update progress dialog if it exists
        if self.progress_dialog:
            self.progress_dialog.update_status(message, current, total)

    def on_data_loaded(self, data):
        """Handle successful data loading"""
        # Close progress dialog
        if self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None

        # Store raw data
        self.loaded_data = data
        num_files = data.get('num_files', 0)
        data_folder = data.get('data_folder', '')

        # Check for errors during loading
        errors = data.get('errors', [])
        if errors:
            # Group errors by type
            basedata_errors = [e for e in errors if e['step'] == 'basedata']
            countrate_errors = [e for e in errors if e['step'] == 'countrates']
            correlation_errors = [e for e in errors if e['step'] == 'correlations']

            # Build error message
            error_parts = []
            if basedata_errors:
                error_parts.append(f"Base data extraction failed for {len(basedata_errors)} file(s):")
                for e in basedata_errors[:5]:  # Show first 5
                    error_parts.append(f"  - {e['file']}: {e['error']}")
                if len(basedata_errors) > 5:
                    error_parts.append(f"  ... and {len(basedata_errors) - 5} more")

            if countrate_errors:
                error_parts.append(f"\nCountrate extraction issues for {len(countrate_errors)} file(s)")

            if correlation_errors:
                error_parts.append(f"Correlation extraction issues for {len(correlation_errors)} file(s)")

            error_message = "\n".join(error_parts)

            # Show warning with error details
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("Data Loading Warnings")
            msg_box.setText(f"Data loaded with warnings.\n\n{data.get('successful_files', 0)}/{data.get('total_files', 0)} files loaded successfully.")
            msg_box.setDetailedText(error_message)
            msg_box.setStandardButtons(QMessageBox.Ok)
            msg_box.exec_()

        # Add data loading step to pipeline with reproducible code
        load_code = f"""
# Load data from folder (case-insensitive for Linux compatibility)
data_folder = r"{data_folder}"
datafiles = []
datafiles.extend(glob.glob(os.path.join(data_folder, "*.asc")))
datafiles.extend(glob.glob(os.path.join(data_folder, "*.ASC")))

# Filter out averaged files
filtered_files = [f for f in datafiles
                  if "averaged" not in os.path.basename(f).lower()]

print(f"Found {{len(filtered_files)}} .asc/.ASC files in {{data_folder}}")

# Initialize data dictionaries
countrates_data = {{}}
correlations_data = {{}}
df_basedata = pd.DataFrame()
"""

        from ade_dls.gui.core.pipeline import AnalysisStep
        load_step = AnalysisStep(
            name="Load Data",
            step_type='custom',
            custom_code=load_code,
            params={
                'data_folder': data_folder,
                'num_files': num_files
            }
        )
        self.pipeline.steps.append(load_step)

        # Add extraction step for countrates and correlations
        extract_code = """
# Extract countrates from all files
for file in filtered_files:
    try:
        df = extract_data(file)
        countrates_data[os.path.basename(file)] = df
    except Exception as e:
        print(f"Error extracting countrates from {file}: {e}")

print(f"Extracted countrates from {len(countrates_data)} files")

# Extract correlations from all files
for file in filtered_files:
    try:
        df = extract_correlation(file)
        correlations_data[os.path.basename(file)] = df
    except Exception as e:
        print(f"Error extracting correlations from {file}: {e}")

print(f"Extracted correlations from {len(correlations_data)} files")
"""

        extract_step = AnalysisStep(
            name="Extract Data",
            step_type='custom',
            custom_code=extract_code,
            params={
                'num_files': num_files
            }
        )
        self.pipeline.steps.append(extract_step)

        # Update status
        self.status_manager.complete_operation(
            f"Successfully loaded {num_files} files"
        )

        # Mark load data step as complete
        self.workflow_panel.mark_step_complete('load_data')

        # Now run filtering
        self.perform_filtering(auto_after_load=True)

    def on_load_error(self, error_message):
        """Handle data loading errors"""
        # Close progress dialog
        if self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None

        # Update status
        self.status_manager.error(error_message)

        # Show error message
        QMessageBox.critical(
            self,
            "Error Loading Data",
            f"Failed to load data:\n\n{error_message}"
        )

    # ========== Preprocessing and Filtering Methods ==========

    def perform_filtering(self, auto_after_load=False):
        """
        Perform complete filtering workflow (count rates and correlations)

        Args:
            auto_after_load: If True, this is automatic after loading
        """
        if not self.loaded_data or 'correlations' not in self.loaded_data:
            QMessageBox.warning(
                self,
                "No Data",
                "Please load data first (File > Load Data)"
            )
            return

        data = self.loaded_data if auto_after_load else self.pipeline.data
        original_count = data.get('num_files', 0)
        filtered_data = data.copy()

        # Step 1: Filter count rates
        if 'countrates' in data and len(data['countrates']) > 0:
            self.status_manager.start_operation("Filtering count rates")

            countrate_dialog = CountrateFilterDialog(data['countrates'], self)
            result = countrate_dialog.exec_()

            if result == countrate_dialog.Accepted:
                filtered_countrates = countrate_dialog.get_filtered_data()
                filtered_data['countrates'] = filtered_countrates
                excluded_count = len(data['countrates']) - len(filtered_countrates)

                # Track which files were excluded for reproducibility
                excluded_files = [f for f in data['countrates'].keys()
                                 if f not in filtered_countrates]

                # Add filter step to pipeline
                self.pipeline.add_filter_step(
                    filter_type='countrates',
                    excluded_files=excluded_files,
                    original_count=len(data['countrates']),
                    remaining_count=len(filtered_countrates)
                )

                if excluded_count > 0:
                    self.status_manager.update(
                        f"Excluded {excluded_count} files based on count rates"
                    )
                else:
                    self.status_manager.update("No count rates excluded")
            else:
                # User cancelled - use all data
                self.status_manager.update("Count rate filtering cancelled - using all data")

        # Step 2: Filter correlations
        if 'correlations' in data and len(data['correlations']) > 0:
            self.status_manager.start_operation("Filtering correlations")

            # Only show correlations that weren't excluded in countrate step
            correlations_to_filter = {k: v for k, v in data['correlations'].items()
                                     if k in filtered_data.get('countrates', data['correlations']).keys()}

            correlation_dialog = CorrelationFilterDialog(correlations_to_filter, self)
            result = correlation_dialog.exec_()

            if result == correlation_dialog.Accepted:
                filtered_correlations = correlation_dialog.get_filtered_data()
                filtered_data['correlations'] = filtered_correlations
                excluded_count = len(correlations_to_filter) - len(filtered_correlations)

                # Track which files were excluded for reproducibility
                excluded_files = [f for f in correlations_to_filter.keys()
                                 if f not in filtered_correlations]

                # Add filter step to pipeline
                self.pipeline.add_filter_step(
                    filter_type='correlations',
                    excluded_files=excluded_files,
                    original_count=len(correlations_to_filter),
                    remaining_count=len(filtered_correlations)
                )

                if excluded_count > 0:
                    self.status_manager.update(
                        f"Excluded {excluded_count} files based on correlations"
                    )
                else:
                    self.status_manager.update("No correlations excluded")
            else:
                # User cancelled - use all data
                self.status_manager.update("Correlation filtering cancelled - using all data")
                filtered_data['correlations'] = correlations_to_filter

        # Step 3: Update base data to match filtered correlations
        if 'basedata' in filtered_data and 'correlations' in filtered_data:
            import pandas as pd
            basedata = filtered_data['basedata']

            # Keep only base data for files that have correlations
            remaining_files = list(filtered_data['correlations'].keys())
            filtered_basedata = basedata[basedata['filename'].isin(remaining_files)]

            filtered_data['basedata'] = filtered_basedata
            filtered_data['num_files'] = len(remaining_files)

        # Store filtered data in pipeline
        self.pipeline.data = filtered_data
        self.pipeline.data_loaded.emit(filtered_data.get('data_folder', 'Unknown'))

        # Update analysis view
        self.analysis_view.update_data_overview()

        # Mark workflow step as complete
        self.workflow_panel.mark_step_complete('preprocess')

        # Show summary
        final_count = filtered_data.get('num_files', 0)
        excluded_total = original_count - final_count

        summary_msg = f"Filtering complete!\n\n"
        summary_msg += f"Original files: {original_count}\n"
        summary_msg += f"Excluded: {excluded_total}\n"
        summary_msg += f"Final dataset: {final_count} files\n\n"
        summary_msg += f"Base data: {len(filtered_data.get('basedata', []))} entries\n"
        summary_msg += f"Correlations: {len(filtered_data.get('correlations', {}))} datasets\n"
        summary_msg += f"Count rates: {len(filtered_data.get('countrates', {}))} datasets"

        self.status_manager.ready()

        QMessageBox.information(
            self,
            "Data Ready",
            summary_msg
        )

    def filter_countrates_manual(self):
        """Manually filter count rate data"""
        if not self.pipeline.data or 'countrates' not in self.pipeline.data:
            QMessageBox.warning(
                self,
                "No Data",
                "No count rate data available.\nPlease load data first."
            )
            return

        self.status_manager.start_operation("Manual count rate filtering")

        data = self.pipeline.data
        countrate_dialog = CountrateFilterDialog(data['countrates'], self)
        result = countrate_dialog.exec_()

        if result == countrate_dialog.Accepted:
            filtered_countrates = countrate_dialog.get_filtered_data()
            original_count = len(data['countrates'])
            excluded_count = original_count - len(filtered_countrates)

            # Track excluded files for reproducibility
            excluded_files = [f for f in data['countrates'].keys()
                             if f not in filtered_countrates]

            # Update data
            self.pipeline.data['countrates'] = filtered_countrates

            # Also filter correlations and basedata to match
            if 'correlations' in self.pipeline.data:
                filtered_correlations = {k: v for k, v in self.pipeline.data['correlations'].items()
                                        if k in filtered_countrates}
                self.pipeline.data['correlations'] = filtered_correlations

            if 'basedata' in self.pipeline.data:
                import pandas as pd
                remaining_files = list(filtered_countrates.keys())
                filtered_basedata = self.pipeline.data['basedata']
                filtered_basedata = filtered_basedata[filtered_basedata['filename'].isin(remaining_files)]
                self.pipeline.data['basedata'] = filtered_basedata
                self.pipeline.data['num_files'] = len(remaining_files)

            # Add to pipeline for code export
            self.pipeline.add_filter_step(
                filter_type='countrates',
                excluded_files=excluded_files,
                original_count=original_count,
                remaining_count=len(filtered_countrates)
            )

            # Update view
            self.analysis_view.update_data_overview()

            self.status_manager.complete_operation(
                f"Count rate filtering complete - excluded {excluded_count} files"
            )

            QMessageBox.information(
                self,
                "Filtering Complete",
                f"Excluded {excluded_count} of {original_count} files based on count rates.\n\n"
                f"Remaining: {len(filtered_countrates)} files"
            )
        else:
            self.status_manager.update("Count rate filtering cancelled")

    def filter_correlations_manual(self):
        """Manually filter correlation data"""
        if not self.pipeline.data or 'correlations' not in self.pipeline.data:
            QMessageBox.warning(
                self,
                "No Data",
                "No correlation data available.\nPlease load data first."
            )
            return

        self.status_manager.start_operation("Manual correlation filtering")

        data = self.pipeline.data
        correlation_dialog = CorrelationFilterDialog(data['correlations'], self)
        result = correlation_dialog.exec_()

        if result == correlation_dialog.Accepted:
            filtered_correlations = correlation_dialog.get_filtered_data()
            original_count = len(data['correlations'])
            excluded_count = original_count - len(filtered_correlations)

            # Track excluded files for reproducibility
            excluded_files = [f for f in data['correlations'].keys()
                             if f not in filtered_correlations]

            # Update data
            self.pipeline.data['correlations'] = filtered_correlations

            # Also filter basedata to match
            if 'basedata' in self.pipeline.data:
                import pandas as pd
                remaining_files = list(filtered_correlations.keys())
                filtered_basedata = self.pipeline.data['basedata']
                filtered_basedata = filtered_basedata[filtered_basedata['filename'].isin(remaining_files)]
                self.pipeline.data['basedata'] = filtered_basedata
                self.pipeline.data['num_files'] = len(remaining_files)

            # Add to pipeline for code export
            self.pipeline.add_filter_step(
                filter_type='correlations',
                excluded_files=excluded_files,
                original_count=original_count,
                remaining_count=len(filtered_correlations)
            )

            # Update view
            self.analysis_view.update_data_overview()

            self.status_manager.complete_operation(
                f"Correlation filtering complete - excluded {excluded_count} files"
            )

            QMessageBox.information(
                self,
                "Filtering Complete",
                f"Excluded {excluded_count} of {original_count} files based on correlations.\n\n"
                f"Remaining: {len(filtered_correlations)} files"
            )
        else:
            self.status_manager.update("Correlation filtering cancelled")

    def run_preprocessing(self):
        """Run preprocessing workflow (filtering)"""
        if not self.loaded_data:
            QMessageBox.warning(
                self,
                "No Data",
                "Please load data first (File > Load Data)"
            )
            return

        self.status_manager.start_operation("Preprocessing")
        self.perform_filtering(auto_after_load=False)

    # ========== Cumulant Analysis Helper Methods ==========

    def _calculate_c_and_delta_c(self):
        """
        Calculate c and delta_c constants needed for Rh calculation
        Stores results in self.loaded_data
        """
        self._calculate_c_and_delta_c_for_data(self.loaded_data)

    def _calculate_c_and_delta_c_for_data(self, data_dict):
        """
        Calculate c and delta_c constants for a given data dictionary
        Stores results in the provided data dictionary

        Args:
            data_dict: Dictionary containing 'df_basedata' key
        """
        from scipy.constants import k  # Boltzmann constant

        df_basedata = data_dict['df_basedata']

        # Calculate mean and error for temperature and viscosity
        mean_temperature = df_basedata['temperature [K]'].mean()
        std_temperature = df_basedata['temperature [K]'].std()

        mean_viscosity = df_basedata['viscosity [cp]'].mean()
        std_viscosity = df_basedata['viscosity [cp]'].std()

        # Calculate c [ c = kb*T/(6*pi*eta) ]
        c = (k * mean_temperature) / (6 * np.pi * mean_viscosity * 1e-3)

        # Calculate error in c
        fractional_error_c = np.sqrt(
            (std_temperature / mean_temperature)**2 +
            (std_viscosity / mean_viscosity)**2
        )
        delta_c = fractional_error_c * c

        # Store in data_dict as Series to maintain compatibility
        data_dict['c'] = pd.Series([c])
        data_dict['delta_c'] = pd.Series([delta_c])

        print(f"[C calculation] c = {c:.4e} +/- {delta_c:.4e} (rel. error: {(delta_c/c):.4%})")

    def _add_basedata_calculation_to_pipeline(self):
        """
        Add basedata statistics and c/delta_c calculation to pipeline
        This should be called once before any cumulant methods
        """
        from ade_dls.gui.core.pipeline import AnalysisStep

        # Check if already added
        if hasattr(self, '_basedata_calc_added') and self._basedata_calc_added:
            return

        code = """
# Calculate basedata statistics and c/delta_c constants
# These are needed for all cumulant methods to calculate Rh

# Calculate mean and error for temperature and viscosity
mean_temperature = df_basedata['temperature [K]'].mean()
std_temperature = df_basedata['temperature [K]'].std()
sem_temperature = df_basedata['temperature [K]'].sem()

mean_viscosity = df_basedata['viscosity [cp]'].mean()
std_viscosity = df_basedata['viscosity [cp]'].std()
sem_viscosity = df_basedata['viscosity [cp]'].sem()

df_basedata_stats = pd.DataFrame({
    'mean temperature [K]': [mean_temperature],
    'std temperature [K]': [std_temperature],
    'sem temperature [K]': [sem_temperature],
    'mean viscosity [cp]': [mean_viscosity],
    'std viscosity [cp]': [std_viscosity],
    'sem viscosity [cp]': [sem_viscosity]
})

# Calculate c and error for determination of Rh [ c = kb*T/(6*pi*eta) ]
from scipy.constants import k  # Boltzmann constant
c = (k * df_basedata_stats['mean temperature [K]']) / (6 * np.pi * df_basedata_stats['mean viscosity [cp]'] * 10**(-3))
c = c.values[0]  # Extract scalar value

fractional_error_c = np.sqrt(
    (df_basedata_stats['std temperature [K]'] / df_basedata_stats['mean temperature [K]'])**2 +
    (df_basedata_stats['std viscosity [cp]'] / df_basedata_stats['mean viscosity [cp]'])**2
)
delta_c = (fractional_error_c * c).values[0]  # Extract scalar value

print(f"\\nc = {c:.4e} +/- {delta_c:.4e}")
print(f"Relative error in c: {(delta_c/c):.4%}\\n")
"""

        step = AnalysisStep(
            name="Calculate c and delta_c",
            step_type='custom',
            custom_code=code,
            params={}
        )
        self.pipeline.steps.append(step)
        self.pipeline.step_added.emit(step.to_dict())
        self._basedata_calc_added = True

    def _add_cumulant_step_to_pipeline(self, method: str, config: dict):
        """
        Add cumulant analysis step to pipeline for code export

        Args:
            method: 'A', 'B', or 'C'
            config: Configuration dictionary from dialog
        """
        from ade_dls.gui.core.pipeline import AnalysisStep

        # Ensure c/delta_c calculation is added first
        self._add_basedata_calculation_to_pipeline()

        if method == 'A':
            code = """
# Cumulant Method A: Extract cumulant data from ALV software
from ade_dls.analysis.cumulants import extract_cumulants, analyze_diffusion_coefficient, calculate_cumulant_results_A
import glob
import os

# Get file paths (case-insensitive for Linux compatibility)
datafiles = []
datafiles.extend(glob.glob(os.path.join(data_folder, "*.asc")))
datafiles.extend(glob.glob(os.path.join(data_folder, "*.ASC")))
file_to_path = {os.path.basename(f): f for f in datafiles}

# Extract cumulant data
all_cumulant_data = []
for filename in correlations_data.keys():
    if filename in file_to_path:
        file_path = file_to_path[filename]
        extracted_cumulants = extract_cumulants(file_path)
        if extracted_cumulants is not None:
            extracted_cumulants['filename'] = filename
            all_cumulant_data.append(extracted_cumulants)

df_extracted_cumulants = pd.concat(all_cumulant_data, ignore_index=True)
df_extracted_cumulants.index = df_extracted_cumulants.index + 1

# Merge with basedata
cumulant_method_A_data = pd.merge(df_basedata, df_extracted_cumulants, on='filename', how='outer')
cumulant_method_A_data = cumulant_method_A_data.reset_index(drop=True)
cumulant_method_A_data.index = cumulant_method_A_data.index + 1

# Analyze diffusion coefficient
cumulant_method_A_diff = analyze_diffusion_coefficient(
    data_df=cumulant_method_A_data,
    q_squared_col='q^2',
    gamma_cols=['1st order frequency [1/ms]', '2nd order frequency [1/ms]', '3rd order frequency [1/ms]'],
    gamma_unit='1/ms'
)

# Calculate diffusion coefficients
A_diff = pd.DataFrame()
A_diff['D [m^2/s]'] = cumulant_method_A_diff['q^2_coef'] * 10**(-15)
A_diff['std err D [m^2/s]'] = cumulant_method_A_diff['q^2_se'] * 10**(-15)

# Calculate polydispersity indices
cumulant_method_A_data['polydispersity_2nd_order'] = (
    cumulant_method_A_data['2nd order frequency exp param [ms^2]'] /
    (cumulant_method_A_data['2nd order frequency [1/ms]'])**2
)
polydispersity_method_A_2 = cumulant_method_A_data['polydispersity_2nd_order'].mean()

cumulant_method_A_data['polydispersity_3rd_order'] = (
    cumulant_method_A_data['3rd order frequency exp param [ms^2]'] /
    (cumulant_method_A_data['3rd order frequency [1/ms]'])**2
)
polydispersity_method_A_3 = cumulant_method_A_data['polydispersity_3rd_order'].mean()

# Calculate results
method_a_results = calculate_cumulant_results_A(A_diff, cumulant_method_A_diff, 
                                                 polydispersity_method_A_2, polydispersity_method_A_3, 
                                                 c, delta_c)

print("\\nCumulant Method A Results:")
print(method_a_results)
"""

        elif method == 'B':
            fit_limits = config['method_b_params']['fit_limits']
            code = f"""
# Cumulant Method B: Linear fit method
from ade_dls.analysis.cumulants import calculate_g2_B, plot_processed_correlations, analyze_diffusion_coefficient
from ade_dls.core.preprocessing import process_correlation_data

# Process correlations
columns_to_drop = ['time [ms]', 'correlation 1', 'correlation 2', 'correlation 3', 'correlation 4']
processed_correlations_1 = process_correlation_data(correlations_data, columns_to_drop)

# Calculate sqrt(g2)
processed_correlations = calculate_g2_B(processed_correlations_1)

# Define fit function
def fit_function(x, a, b, c):
    return 0.5 * np.log(a) - b * x + 0.5 * c * x**2

# Fit data
fit_limits = {fit_limits}
cumulant_method_B_fit = plot_processed_correlations(processed_correlations, fit_function, fit_limits)

# Merge with basedata
cumulant_method_B_data = pd.merge(df_basedata, cumulant_method_B_fit, on='filename', how='outer')
cumulant_method_B_data = cumulant_method_B_data.reset_index(drop=True)
cumulant_method_B_data.index = cumulant_method_B_data.index + 1

# Analyze diffusion coefficient
cumulant_method_B_diff = analyze_diffusion_coefficient(
    data_df=cumulant_method_B_data,
    q_squared_col='q^2',
    gamma_cols=['b'],
    method_names=['Method B']
)

# Calculate diffusion coefficients
B_diff = pd.DataFrame()
B_diff['D [m^2/s]'] = cumulant_method_B_diff['q^2_coef'] * 10**(-18)
B_diff['std err D [m^2/s]'] = cumulant_method_B_diff['q^2_se'] * 10**(-18)

# Calculate polydispersity
cumulant_method_B_data['polydispersity'] = cumulant_method_B_data['c'] / (cumulant_method_B_data['b'])**2
polydispersity_method_B = cumulant_method_B_data['polydispersity'].mean()

# Calculate results
method_b_results = pd.DataFrame()
method_b_results['Rh [nm]'] = c * (1 / B_diff['D [m^2/s]'][0]) * 10**9
fractional_error_Rh_B = np.sqrt((delta_c / c)**2 + (B_diff['std err D [m^2/s]'][0] / B_diff['D [m^2/s]'][0])**2)
method_b_results['Rh error [nm]'] = fractional_error_Rh_B * method_b_results['Rh [nm]']
method_b_results['R_squared'] = cumulant_method_B_diff['R_squared']
method_b_results['Fit'] = 'Rh from linear cumulant fit'
method_b_results['Residuals'] = cumulant_method_B_diff['Normality']
method_b_results['PDI'] = polydispersity_method_B

print("\\nCumulant Method B Results:")
print(method_b_results)
"""

        elif method == 'C':
            params = config['method_c_params']
            code = f"""
# Cumulant Method C: Iterative non-linear fit
from ade_dls.analysis.cumulants_C import plot_processed_correlations_iterative, get_adaptive_initial_parameters, get_meaningful_parameters
from ade_dls.analysis.cumulants import analyze_diffusion_coefficient
from ade_dls.core.preprocessing import process_correlation_data

# Process correlations
columns_to_drop = ['time [ms]', 'correlation 1', 'correlation 2', 'correlation 3', 'correlation 4']
processed_correlations_1 = process_correlation_data(correlations_data, columns_to_drop)

# Define fit function
def {params['fit_function']}(x, a, b, c, *args):
    # Function defined based on selection: {params['fit_function']}
    pass  # See cumulants_C.py for full implementation

# Parameters
fit_limits = {params['fit_limits']}
adaptive_initial_guesses = {params['adaptive_initial_guesses']}
adaptation_strategy = '{params['adaptation_strategy']}'
optimizer = '{params['optimizer']}'
base_initial_parameters = {params['initial_parameters']}

# Get initial parameters
if adaptive_initial_guesses:
    initial_parameters = get_adaptive_initial_parameters(
        processed_correlations_1, {params['fit_function']}, 
        base_initial_parameters, strategy=adaptation_strategy, verbose=False
    )
else:
    initial_parameters = get_meaningful_parameters({params['fit_function']}, base_initial_parameters)

# Run fitting
cumulant_method_C_fit = plot_processed_correlations_iterative(
    processed_correlations_1, {params['fit_function']}, fit_limits, initial_parameters, method=optimizer
)

# Merge with basedata
cumulant_method_C_data = pd.merge(df_basedata, cumulant_method_C_fit, on='filename', how='outer')
cumulant_method_C_data = cumulant_method_C_data.reset_index(drop=True)
cumulant_method_C_data.index = cumulant_method_C_data.index + 1

# Analyze diffusion coefficient
cumulant_method_C_diff = analyze_diffusion_coefficient(
    data_df=cumulant_method_C_data,
    q_squared_col='q^2',
    gamma_cols=['best_b'],
    method_names=['Method C']
)

# Calculate diffusion coefficients
C_diff = pd.DataFrame()
C_diff['D [m^2/s]'] = cumulant_method_C_diff['q^2_coef'] * 10**(-18)
C_diff['std err D [m^2/s]'] = cumulant_method_C_diff['q^2_se'] * 10**(-18)

# Calculate polydispersity
cumulant_method_C_data['polydispersity'] = cumulant_method_C_data['best_c'] / (cumulant_method_C_data['best_b'])**2
polydispersity_method_C = cumulant_method_C_data['polydispersity'].mean()

# Calculate results
method_c_results = pd.DataFrame()
method_c_results['Rh [nm]'] = c * (1 / C_diff['D [m^2/s]'][0]) * 10**9
fractional_error_Rh_C = np.sqrt((delta_c / c)**2 + (C_diff['std err D [m^2/s]'][0] / C_diff['D [m^2/s]'][0])**2)
method_c_results['Rh error [nm]'] = fractional_error_Rh_C * method_c_results['Rh [nm]']
method_c_results['R_squared'] = cumulant_method_C_diff['R_squared']
method_c_results['Fit'] = 'Rh from iterative non-linear cumulant fit'
method_c_results['Residuals'] = cumulant_method_C_diff['Normality']
method_c_results['PDI'] = polydispersity_method_C

print("\\nCumulant Method C Results:")
print(method_c_results)
"""

        # Add step to pipeline
        step = AnalysisStep(
            name=f"Cumulant Analysis Method {method}",
            step_type='custom',
            custom_code=code,
            params=config
        )
        self.pipeline.steps.append(step)
        self.pipeline.step_added.emit(step.to_dict())

    def _display_cumulant_results(self, results: list, analyzer: 'CumulantAnalyzer'):
        """
        Display cumulant analysis results

        Args:
            results: List of tuples (method_name, result_dataframe)
            analyzer: CumulantAnalyzer instance with results
        """
        # Store analyzer for export
        self.cumulant_analyzer = analyzer

        print(f"[MAIN WINDOW] _display_cumulant_results called with {len(results)} methods")
        for method_name, result_df in results:
            print(f"[MAIN WINDOW]   - {method_name}: {result_df.shape[0]} rows")

        # Clear previous results
        self.analysis_view.clear_results()

        # Combine all results
        combined_results = analyzer.get_combined_results()

        # Print to console
        print("\n" + "=" * 60)
        print("CUMULANT ANALYSIS RESULTS")
        print("=" * 60)
        print(combined_results.to_string())
        print("=" * 60 + "\n")

        # Display results for each method in the AnalysisView
        for method_name, result_df in results:
            # Get plots and fit quality based on method
            plots_dict = None
            fit_quality = None

            if 'A' in method_name:
                # Method A has a summary plot (Gamma vs q²)
                plots_dict = None
                if hasattr(analyzer, 'method_a_summary_plot'):
                    # Wrap the summary plot as a single-item dict
                    plots_dict = {'Method A Summary': (analyzer.method_a_summary_plot, {})}

                # Get regression statistics
                regression_stats = None
                if hasattr(analyzer, 'method_a_regression_stats'):
                    regression_stats = analyzer.method_a_regression_stats

                # Don't switch tabs yet - accumulate all results first
                self.analysis_view.display_cumulant_results(
                    method_name, result_df, plots_dict, None, switch_tab=False,
                    regression_stats=regression_stats
                )

            elif 'B' in method_name:
                plots_dict = {}
                if hasattr(analyzer, 'method_b_plots'):
                    plots_dict.update(analyzer.method_b_plots)

                # Add summary plot (Gamma vs q²) if available
                if hasattr(analyzer, 'method_b_summary_plot'):
                    plots_dict['Method B Summary'] = (analyzer.method_b_summary_plot, {})

                if hasattr(analyzer, 'method_b_fit_quality'):
                    fit_quality = analyzer.method_b_fit_quality

                # Get regression statistics (as dict, not model object)
                regression_stats = None
                if hasattr(analyzer, 'method_b_regression_stats'):
                    regression_stats = analyzer.method_b_regression_stats

                # Don't switch tabs yet
                self.analysis_view.display_cumulant_results(
                    method_name, result_df, plots_dict, fit_quality, switch_tab=False,
                    regression_stats=regression_stats
                )

            elif 'C' in method_name:
                plots_dict = {}
                if hasattr(analyzer, 'method_c_plots'):
                    plots_dict.update(analyzer.method_c_plots)

                # Add summary plot (Gamma vs q²) if available
                if hasattr(analyzer, 'method_c_summary_plot'):
                    plots_dict['Method C Summary'] = (analyzer.method_c_summary_plot, {})

                if hasattr(analyzer, 'method_c_fit_quality'):
                    fit_quality = analyzer.method_c_fit_quality

                # Get regression statistics (as dict, not model object)
                regression_stats = None
                if hasattr(analyzer, 'method_c_regression_stats'):
                    regression_stats = analyzer.method_c_regression_stats

                # Don't switch tabs yet
                self.analysis_view.display_cumulant_results(
                    method_name, result_df, plots_dict, fit_quality, switch_tab=False,
                    regression_stats=regression_stats, analyzer=analyzer
                )

        # After all methods are loaded, switch to Results tab to show the summary
        self.analysis_view.show_results_tab()

        # Update status
        self.status_manager.complete_operation(
            f"Cumulant analysis complete - Results displayed in Analysis tabs"
        )

    def run_nnls_analysis(self):
        """Run NNLS (Non-Negative Least Squares) analysis"""
        # Use filtered data from pipeline if available, otherwise use loaded_data
        # This ensures that filtering is respected!
        if hasattr(self.pipeline, 'data') and self.pipeline.data and 'correlations' in self.pipeline.data:
            working_data = self.pipeline.data
            data_source = "filtered"
            print(f"[NNLS] Using filtered data from pipeline ({len(working_data.get('correlations', {}))} datasets)")
        elif self.loaded_data and 'correlations' in self.loaded_data:
            working_data = self.loaded_data
            data_source = "original"
            print(f"[NNLS] Using original loaded data ({len(working_data.get('correlations', {}))} datasets)")
        else:
            QMessageBox.warning(
                self,
                "No Data",
                "Please load data first (File > Load Data)"
            )
            return

        # Check for basedata
        if 'basedata' not in working_data:
            QMessageBox.warning(
                self,
                "Missing Basedata",
                "Base data is missing.\n"
                "Please ensure data loading was completed successfully."
            )
            return

        # Prepare df_basedata and calculate c/delta_c if not already done
        if 'df_basedata' not in working_data:
            # Create df_basedata from basedata if needed
            working_data['df_basedata'] = working_data['basedata'].copy()

        if 'c' not in working_data or 'delta_c' not in working_data:
            self.status_manager.update("Calculating c and delta_c constants...")
            self._calculate_c_and_delta_c_for_data(working_data)

        try:
            # Always recreate laplace analyzer to ensure we use current (possibly filtered) data
            # Check if processed correlations exist, otherwise use raw correlations
            processed_corr = working_data.get('processed_correlations', None)
            raw_corr = working_data.get('correlations', None) if processed_corr is None else None

            self.laplace_analyzer = LaplaceAnalyzer(
                processed_correlations=processed_corr,
                df_basedata=working_data['df_basedata'],
                c=working_data['c'],
                delta_c=working_data['delta_c'],
                raw_correlations=raw_corr
            )

            # Store processed correlations back for future use
            if processed_corr is None and self.laplace_analyzer.processed_correlations is not None:
                working_data['processed_correlations'] = self.laplace_analyzer.processed_correlations
                print("[NNLS] Stored processed correlations")

            # Inform user about data source
            if data_source == "filtered":
                num_datasets = len(working_data.get('correlations', {}))
                QMessageBox.information(
                    self,
                    "Using Filtered Data",
                    f"NNLS analysis will use the filtered dataset.\n\n"
                    f"Number of datasets: {num_datasets}\n\n"
                    f"💡 Tip: This respects your filtering choices from preprocessing."
                )

            # Show NNLS dialog
            dialog = NNLSDialog(self.laplace_analyzer, self)
            if dialog.exec_() == dialog.Accepted:
                params = dialog.get_parameters()

                # Start analysis
                self.status_manager.start_operation("Running NNLS analysis...")

                try:
                    # Run NNLS (show_plots=False to avoid extra windows, plots shown in GUI)
                    self.status_manager.update("Performing NNLS fits...")
                    self.laplace_analyzer.run_nnls(
                        params,
                        use_multiprocessing=params.get('use_multiprocessing', False),
                        show_plots=False  # Plots are displayed in Analysis View, not as separate windows
                    )

                    # Calculate diffusion coefficients
                    self.status_manager.update("Calculating diffusion coefficients...")
                    self.laplace_analyzer.calculate_nnls_diffusion_coefficients()

                    # Add to pipeline
                    self._add_nnls_step_to_pipeline(params)

                    # Mark workflow step as complete
                    self.workflow_panel.mark_step_complete('nnls')

                    # Display results in analysis view
                    self.status_manager.complete_operation("NNLS analysis completed")
                    self._display_nnls_results()

                except Exception as e:
                    self.status_manager.error(f"NNLS analysis failed: {str(e)}")
                    QMessageBox.critical(
                        self,
                        "NNLS Analysis Failed",
                        f"An error occurred during NNLS analysis:\n\n{str(e)}"
                    )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Setup Error",
                f"Error setting up NNLS analyzer:\n\n{str(e)}"
            )

    def _display_nnls_results(self, replace_existing=False):
        """
        Display NNLS analysis results in Analysis View (analog to cumulant results)

        Args:
            replace_existing: If True, replace existing NNLS plots (for post-refinement)
        """
        if not hasattr(self, 'laplace_analyzer') or self.laplace_analyzer.nnls_final_results is None:
            QMessageBox.warning(
                self,
                "No Results",
                "No NNLS results available."
            )
            return

        print("\n" + "="*60)
        print("NNLS ANALYSIS RESULTS")
        print("="*60)
        print(self.laplace_analyzer.nnls_final_results.to_string())
        print("="*60 + "\n")

        # Don't clear previous results - append instead
        # This allows combining Cumulant and Laplace results
        # self.analysis_view.clear_results()  # Removed to allow appending results

        # Prepare plots dictionary
        plots_dict = {}

        # Add individual fit plots
        if hasattr(self.laplace_analyzer, 'nnls_plots'):
            plots_dict.update(self.laplace_analyzer.nnls_plots)

        # Add summary plot (Gamma vs q²)
        if hasattr(self.laplace_analyzer, 'nnls_summary_plot'):
            plots_dict['NNLS Diffusion Analysis'] = (self.laplace_analyzer.nnls_summary_plot, {})

        # Add clustering visualization plot
        if hasattr(self.laplace_analyzer, 'nnls_clustering_plot'):
            plots_dict['NNLS Peak Clustering'] = (self.laplace_analyzer.nnls_clustering_plot, {})

        # Create fit quality dict
        fit_quality = {}
        if hasattr(self.laplace_analyzer, 'nnls_plots'):
            for filename, (fig, data) in self.laplace_analyzer.nnls_plots.items():
                fit_quality[filename] = {
                    'num_peaks': data.get('num_peaks', 0),
                    'peaks': data.get('peaks', [])
                }

        # Get regression statistics
        regression_stats = {}
        if hasattr(self.laplace_analyzer, 'nnls_diff_results'):
            # Store regression stats for each peak
            regression_stats['regression_results'] = []
            for i, row in self.laplace_analyzer.nnls_diff_results.iterrows():
                regression_stats['regression_results'].append({
                    'gamma_col': f'NNLS Peak {i+1}',
                    'q^2_coef': row.get('q^2_coef', 0),
                    'q^2_se': row.get('q^2_se', 0),
                    'R_squared': row.get('R_squared', 0),
                    'Normality': row.get('Normality', 'N/A')
                })

        # Display in analysis view
        self.analysis_view.display_cumulant_results(
            "NNLS Analysis",
            self.laplace_analyzer.nnls_final_results,
            plots_dict,
            fit_quality,
            switch_tab=False,
            regression_stats=regression_stats if regression_stats.get('regression_results') else None,
            analyzer=self.laplace_analyzer,  # Pass analyzer for post-fit refinement
            replace_existing=replace_existing  # Replace plots if this is post-refinement update
        )

        # Switch to Results tab to show the summary
        self.analysis_view.show_results_tab()

    def show_nnls_results(self):
        """Show NNLS results dialog (legacy support)"""
        if not hasattr(self, 'laplace_analyzer') or self.laplace_analyzer.nnls_final_results is None:
            QMessageBox.warning(
                self,
                "No Results",
                "No NNLS results available. Please run NNLS analysis first."
            )
            return

        dialog = NNLSResultsDialog(self.laplace_analyzer, self)
        dialog.exec_()

    def run_regularized_analysis(self):
        """Run Regularized NNLS analysis with Tikhonov-Phillips regularization"""
        # Use filtered data from pipeline if available
        if hasattr(self.pipeline, 'data') and self.pipeline.data and 'correlations' in self.pipeline.data:
            working_data = self.pipeline.data
            data_source = "filtered"
            print(f"[Regularized] Using filtered data from pipeline ({len(working_data.get('correlations', {}))} datasets)")
        elif self.loaded_data and 'correlations' in self.loaded_data:
            working_data = self.loaded_data
            data_source = "original"
            print(f"[Regularized] Using original loaded data ({len(working_data.get('correlations', {}))} datasets)")
        else:
            QMessageBox.warning(
                self,
                "No Data",
                "Please load data first (File > Load Data)"
            )
            return

        # Check for basedata
        if 'basedata' not in working_data:
            QMessageBox.warning(
                self,
                "Missing Basedata",
                "Base data is missing.\n"
                "Please ensure data loading was completed successfully."
            )
            return

        # Prepare df_basedata and calculate c/delta_c if needed
        if 'df_basedata' not in working_data:
            working_data['df_basedata'] = working_data['basedata'].copy()

        if 'c' not in working_data or 'delta_c' not in working_data:
            self.status_manager.update("Calculating c and delta_c constants...")
            self._calculate_c_and_delta_c_for_data(working_data)

        try:
            # Always recreate laplace analyzer to use current data
            processed_corr = working_data.get('processed_correlations', None)
            raw_corr = working_data.get('correlations', None) if processed_corr is None else None

            self.laplace_analyzer = LaplaceAnalyzer(
                processed_correlations=processed_corr,
                df_basedata=working_data['df_basedata'],
                c=working_data['c'],
                delta_c=working_data['delta_c'],
                raw_correlations=raw_corr
            )

            # Store processed correlations back for future use
            if processed_corr is None and self.laplace_analyzer.processed_correlations is not None:
                working_data['processed_correlations'] = self.laplace_analyzer.processed_correlations
                print("[Regularized] Stored processed correlations")

            # Inform user about data source
            if data_source == "filtered":
                num_datasets = len(working_data.get('correlations', {}))
                QMessageBox.information(
                    self,
                    "Using Filtered Data",
                    f"Regularized analysis will use the filtered dataset.\n\n"
                    f"Number of datasets: {num_datasets}\n\n"
                    f"💡 Tip: This respects your filtering choices from preprocessing."
                )

            # Show Regularized dialog
            dialog = RegularizedDialog(self.laplace_analyzer, self)
            if dialog.exec_() == dialog.Accepted:
                params = dialog.get_parameters()

                # Start analysis
                self.status_manager.start_operation("Running Regularized NNLS analysis...")

                try:
                    # Run Regularized fit (show_plots=False to avoid extra windows, plots shown in GUI)
                    self.status_manager.update("Performing regularized fits...")
                    self.laplace_analyzer.run_regularized(
                        params,
                        use_multiprocessing=params.get('use_multiprocessing', False),
                        show_plots=False  # Plots are displayed in Analysis View, not as separate windows
                    )

                    # Calculate diffusion coefficients
                    self.status_manager.update("Calculating diffusion coefficients...")
                    self.laplace_analyzer.calculate_regularized_diffusion_coefficients()

                    # Add to pipeline
                    self._add_regularized_step_to_pipeline(params)

                    # Mark workflow step as complete
                    self.workflow_panel.mark_step_complete('regularized')

                    # Display results in analysis view
                    self.status_manager.complete_operation("Regularized analysis completed")
                    self._display_regularized_results()

                except Exception as e:
                    self.status_manager.error(f"Regularized analysis failed: {str(e)}")
                    QMessageBox.critical(
                        self,
                        "Regularized Analysis Failed",
                        f"An error occurred during regularized analysis:\n\n{str(e)}"
                    )
                    import traceback
                    traceback.print_exc()

        except Exception as e:
            QMessageBox.critical(
                self,
                "Setup Error",
                f"Error setting up Laplace analyzer:\n\n{str(e)}"
            )
            import traceback
            traceback.print_exc()

    def _display_regularized_results(self, replace_existing=False):
        """
        Display Regularized NNLS analysis results in Analysis View

        Args:
            replace_existing: If True, replace existing Regularized plots (for post-refinement)
        """
        if not hasattr(self, 'laplace_analyzer') or self.laplace_analyzer.regularized_final_results is None:
            QMessageBox.warning(
                self,
                "No Results",
                "No Regularized NNLS results available."
            )
            return

        print("\n" + "="*60)
        print("REGULARIZED NNLS ANALYSIS RESULTS")
        print("="*60)
        print(self.laplace_analyzer.regularized_final_results.to_string())
        print("="*60 + "\n")

        # Prepare plots dictionary
        plots_dict = {}

        # Add individual fit plots
        if hasattr(self.laplace_analyzer, 'regularized_plots'):
            plots_dict.update(self.laplace_analyzer.regularized_plots)

        # Add summary plot (Gamma vs q²)
        if hasattr(self.laplace_analyzer, 'regularized_summary_plot'):
            plots_dict['Regularized Diffusion Analysis'] = (self.laplace_analyzer.regularized_summary_plot, {})

        # Add clustering visualization plot
        if hasattr(self.laplace_analyzer, 'regularized_clustering_plot'):
            plots_dict['Regularized Peak Clustering'] = (self.laplace_analyzer.regularized_clustering_plot, {})

        # Create fit quality dict
        fit_quality = {}
        if hasattr(self.laplace_analyzer, 'regularized_plots'):
            for filename, (fig, data) in self.laplace_analyzer.regularized_plots.items():
                fit_quality[filename] = {
                    'num_peaks': data.get('num_peaks', 0),
                    'peaks': data.get('peaks', [])
                }

        # Get regression statistics
        regression_stats = {}
        if hasattr(self.laplace_analyzer, 'regularized_diff_results'):
            # Store regression stats for each peak
            regression_stats['regression_results'] = []
            for i, row in self.laplace_analyzer.regularized_diff_results.iterrows():
                regression_stats['regression_results'].append({
                    'gamma_col': f'Regularized Peak {i+1}',
                    'q^2_coef': row.get('q^2_coef', 0),
                    'q^2_se': row.get('q^2_se', 0),
                    'R_squared': row.get('R_squared', 0),
                    'Normality': row.get('Normality', 'N/A')
                })

        # Display in analysis view
        self.analysis_view.display_cumulant_results(
            "Regularized NNLS",
            self.laplace_analyzer.regularized_final_results,
            plots_dict,
            fit_quality,
            switch_tab=False,
            regression_stats=regression_stats if regression_stats.get('regression_results') else None,
            analyzer=self.laplace_analyzer,  # Pass analyzer for post-fit refinement
            replace_existing=replace_existing  # Replace plots if this is post-refinement update
        )

        # Switch to Results tab to show the summary
        self.analysis_view.show_results_tab()

    def compare_results(self):
        """Compare results from different methods"""
        # TODO: Implement comparison view
        QMessageBox.information(
            self,
            "Coming Soon",
            "Results comparison feature will allow you to:\n\n"
            "• Compare Cumulant A/B/C methods\n"
            "• Compare NNLS vs Regularized fits\n"
            "• View all Rh values side-by-side\n"
            "• Export comprehensive comparison reports"
        )

    def _add_nnls_step_to_pipeline(self, params):
        """Add NNLS analysis step to pipeline"""
        # Extract parameters
        use_clustering = params.get('use_clustering', True)
        use_centroid = params.get('use_centroid', False)
        eps_factor = params.get('eps_factor', 0.3)

        # Create code for this step
        code = f"""
# NNLS Analysis
# Parameters: prominence={params['prominence']}, distance={params['distance']}
# Peak position: {'Centroid (center of mass)' if use_centroid else 'Maximum'}
# Clustering: {'Enabled (automatic mode detection)' if use_clustering else 'Disabled'}

from ade_dls.analysis.regularized_optimized import nnls_all_optimized, calculate_decay_rates
from ade_dls.analysis.peak_clustering import cluster_peaks_across_datasets, analyze_diffusion_coefficient_robust
import numpy as np

# Run NNLS
nnls_params = {{
    'decay_times': np.logspace({np.log10(params['decay_times'][0]):.2f}, {np.log10(params['decay_times'][-1]):.2f}, {len(params['decay_times'])}),
    'prominence': {params['prominence']},
    'distance': {params['distance']},
    'use_centroid': {use_centroid}  # Use centroid instead of maximum for peak position
}}

nnls_results = nnls_all_optimized(
    processed_correlations,
    nnls_params,
    use_multiprocessing={params.get('use_multiprocessing', False)},
    show_plots=False  # Don't show individual plots in batch mode
)

# Merge with basedata
nnls_data = pd.merge(df_basedata, nnls_results, on='filename', how='outer')
"""

        # Add clustering code if enabled
        if use_clustering:
            code += f"""
# Perform automatic peak clustering to identify common modes across angles
print("\\n[NNLS] Performing automatic peak clustering...")
nnls_data, cluster_info = cluster_peaks_across_datasets(
    nnls_data,
    tau_prefix='tau_',
    method='dbscan',
    eps_factor={eps_factor},
    min_samples=max(2, int(0.2 * len(nnls_data)))
)

print(f"[NNLS] Detected {{len(cluster_info)}} mode(s) via clustering")
for mode_key, info in cluster_info.items():
    print(f"  {{mode_key}}: n={{info['n_points']}}, D={{info['mean_D']:.2e}} nm²/s, CV={{info['cv_D']:.2%}}")
"""

        code += """
# Calculate decay rates
tau_columns = [col for col in nnls_data.columns if col.startswith('tau_')]
nnls_data = calculate_decay_rates(nnls_data, tau_columns)

# Calculate diffusion coefficients using robust regression
gamma_columns = [col.replace('tau', 'gamma') for col in tau_columns]
nnls_diff_results = analyze_diffusion_coefficient_robust(
    data_df=nnls_data,
    q_squared_col='q^2',
    gamma_cols=gamma_columns,
    robust_method='ransac',
    show_plots=False
)

# Calculate final Rh results
nnls_final_results = []
for i in range(len(nnls_diff_results)):
    if pd.notna(nnls_diff_results['q^2_coef'][i]):
        # Convert D from nm²/s to m²/s
        D_m2s = nnls_diff_results['q^2_coef'][i] * 1e-18
        D_err_m2s = nnls_diff_results['q^2_se'][i] * 1e-18

        # Calculate Rh
        Rh_nm = c * (1 / D_m2s) * 1e9

        # Error propagation
        fractional_error = np.sqrt((delta_c / c)**2 + (D_err_m2s / D_m2s)**2)
        Rh_error_nm = fractional_error * Rh_nm

        result = {
            'Rh [nm]': Rh_nm,
            'Rh error [nm]': Rh_error_nm,
            'D [m^2/s]': D_m2s,
            'D error [m^2/s]': D_err_m2s,
            'R_squared': nnls_diff_results['R_squared'][i],
            'Fit': f'NNLS Peak {i+1}',
            'Residuals': nnls_diff_results.get('Normality', [0])[i]
        }
        nnls_final_results.append(result)

nnls_final_results_df = pd.DataFrame(nnls_final_results)
print("\\nNNLS Analysis Results:")
print(nnls_final_results_df)
"""

        # Add step to pipeline
        from ade_dls.gui.core.pipeline import AnalysisStep
        step = AnalysisStep(
            name="NNLS Analysis",
            step_type='custom',
            custom_code=code,
            params=params
        )
        self.pipeline.steps.append(step)
        self.pipeline.step_added.emit(step.to_dict())

    def _add_regularized_step_to_pipeline(self, params):
        """Add Regularized NNLS analysis step to pipeline"""
        # Extract parameters
        use_centroid = params.get('use_centroid', False)

        # Create code for this step
        code = f"""
# Regularized NNLS Analysis with Tikhonov-Phillips Regularization
# Parameters: alpha={params['alpha']}, prominence={params['prominence']}, distance={params['distance']}
# Peak position: {'Centroid (center of mass)' if use_centroid else 'Maximum'}

from ade_dls.analysis.regularized import nnls_reg_all, calculate_decay_rates
from ade_dls.analysis.peak_clustering import analyze_diffusion_coefficient_robust
import numpy as np

# Run Regularized NNLS
reg_params = {{
    'decay_times': np.logspace({np.log10(params['decay_times'][0]):.2f}, {np.log10(params['decay_times'][-1]):.2f}, {len(params['decay_times'])}),
    'alpha': {params['alpha']},
    'prominence': {params['prominence']},
    'distance': {params['distance']},
    'normalize': {params.get('normalize', True)},
    'sparsity_penalty': {params.get('sparsity_penalty', 0.0)},
    'enforce_unimodality': {params.get('enforce_unimodality', False)},
    'use_centroid': {use_centroid}
}}

regularized_results, full_results = nnls_reg_all(
    processed_correlations,
    reg_params
)

# Merge with basedata
regularized_data = pd.merge(df_basedata, regularized_results, on='filename', how='outer')

# Calculate decay rates
tau_columns = [col for col in regularized_data.columns if col.startswith('tau_')]
regularized_data = calculate_decay_rates(regularized_data, tau_columns)

# Calculate diffusion coefficients using robust regression
gamma_columns = [col.replace('tau', 'gamma') for col in tau_columns]
regularized_diff_results = analyze_diffusion_coefficient_robust(
    data_df=regularized_data,
    q_squared_col='q^2',
    gamma_cols=gamma_columns,
    robust_method='ransac',
    show_plots=False
)

# Calculate final Rh results
regularized_final_results = []
for i in range(len(regularized_diff_results)):
    if pd.notna(regularized_diff_results['q^2_coef'][i]):
        D_m2s = regularized_diff_results['q^2_coef'][i] * 1e-18
        D_err_m2s = regularized_diff_results['q^2_se'][i] * 1e-18
        Rh_nm = c * (1 / D_m2s) * 1e9
        fractional_error = np.sqrt((delta_c / c)**2 + (D_err_m2s / D_m2s)**2)
        Rh_error_nm = fractional_error * Rh_nm

        result = {{
            'Rh [nm]': Rh_nm,
            'Rh error [nm]': Rh_error_nm,
            'D [m^2/s]': D_m2s,
            'D error [m^2/s]': D_err_m2s,
            'R_squared': regularized_diff_results['R_squared'][i],
            'Fit': f'Regularized Peak {{i+1}}',
            'Residuals': regularized_diff_results.get('Normality', [0])[i]
        }}
        regularized_final_results.append(result)

regularized_final_results_df = pd.DataFrame(regularized_final_results)
print("\\nRegularized NNLS Results:")
print(regularized_final_results_df)
"""

        # Add step to pipeline
        from ade_dls.gui.core.pipeline import AnalysisStep
        step = AnalysisStep(
            name="Regularized NNLS Analysis",
            step_type='custom',
            custom_code=code,
            params=params
        )
        self.pipeline.steps.append(step)
        self.pipeline.step_added.emit(step.to_dict())

def main():
    """Main entry point for ADE-DLS GUI application."""
    import sys
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    app.setApplicationName("ADE-DLS")
    app.setOrganizationName("ADE-DLS")
    
    window = JADEDLSMainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
