"""
Transparent Analysis Pipeline
Tracks all analysis steps, parameters, and generates reproducible code
"""

import inspect
import json
from datetime import datetime
from typing import List, Dict, Any, Callable
from PyQt5.QtCore import QObject, pyqtSignal
import pandas as pd
import numpy as np


class AnalysisStep:
    """Represents a single step in the analysis pipeline"""

    def __init__(self, name: str, function: Callable = None, params: Dict[str, Any] = None,
                 step_type: str = "function", custom_code: str = None):
        self.name = name
        self.function = function
        self.function_name = function.__name__ if function else None
        self.params = params or {}
        self.timestamp = datetime.now()
        self.result = None
        self.code = None
        self.metadata = {}
        self.step_type = step_type  # 'function' or 'filter' or 'custom'
        self.custom_code = custom_code

        # Extract function source code
        if function:
            try:
                self.code = inspect.getsource(function)
            except (OSError, TypeError):
                self.code = f"# Built-in function: {self.function_name}"

    def execute(self, *args, **kwargs):
        """Execute this analysis step"""
        if self.function:
            self.result = self.function(*args, **self.params, **kwargs)
            return self.result
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert step to dictionary for serialization"""
        return {
            'name': self.name,
            'function': self.function_name,
            'params': self.params,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata,
            'step_type': self.step_type
        }

    def generate_code(self) -> str:
        """Generate Python code for this step"""
        # If we have custom code (for filters or manual steps), use that
        if self.custom_code:
            return f"""
# Step: {self.name}
# Timestamp: {self.timestamp}
{self.custom_code}
"""
        elif self.function:
            params_str = ', '.join([f"{k}={repr(v)}" for k, v in self.params.items()])
            return f"""
# Step: {self.name}
# Timestamp: {self.timestamp}
result_{self.name.replace(' ', '_')} = {self.function_name}({params_str})
"""
        else:
            return f"""
# Step: {self.name}
# Timestamp: {self.timestamp}
# Manual step - no automatic code generation
"""


class TransparentPipeline(QObject):
    """
    Transparent analysis pipeline that tracks all steps and can export to
    Jupyter notebooks or Python scripts

    Signals:
        step_added: Emitted when a new step is added
        analysis_completed: Emitted when analysis is complete
        data_loaded: Emitted when data is loaded
    """

    step_added = pyqtSignal(dict)
    analysis_completed = pyqtSignal(dict)
    data_loaded = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.steps: List[AnalysisStep] = []
        self.data = {}
        self.results = {}
        self.metadata = {
            'created': datetime.now(),
            'version': '2.0',
            'description': 'JADE-DLS Analysis Pipeline'
        }

    def add_step(self, name: str, function: Callable, params: Dict[str, Any] = None) -> AnalysisStep:
        """
        Add a step to the pipeline

        Args:
            name: Human-readable step name
            function: Function to execute
            params: Parameters for the function

        Returns:
            The created AnalysisStep
        """
        if params is None:
            params = {}

        step = AnalysisStep(name, function, params)
        self.steps.append(step)

        # Emit signal
        self.step_added.emit(step.to_dict())

        return step

    def add_post_refinement_step(self, method_name: str, q_range: tuple = None,
                                 excluded_files: List[str] = None,
                                 is_laplace: bool = False) -> AnalysisStep:
        """
        Add a post-fit refinement step to the pipeline

        Args:
            method_name: Name of method being refined ('Method C', 'NNLS', 'Regularized')
            q_range: Optional (min, max) q² range for refinement
            excluded_files: Optional list of files/datasets to exclude
            is_laplace: True for NNLS/Regularized, False for Cumulant methods

        Returns:
            The created AnalysisStep
        """
        # Generate code based on method type
        if is_laplace:
            # NNLS or Regularized refinement
            if method_name == "NNLS":
                code = self._generate_nnls_refinement_code(q_range, excluded_files)
            else:  # Regularized
                code = self._generate_regularized_refinement_code(q_range, excluded_files)
        else:
            # Cumulant Method C refinement
            code = self._generate_method_c_refinement_code(q_range, excluded_files)

        # Create step
        step = AnalysisStep(
            name=f"{method_name} Post-Refinement",
            step_type='refinement',
            custom_code=code,
            params={
                'method': method_name,
                'q_range': q_range,
                'excluded_files': excluded_files or []
            }
        )

        self.steps.append(step)
        self.step_added.emit(step.to_dict())

        return step

    def add_filter_step(self, filter_type: str, excluded_files: List[str],
                       original_count: int, remaining_count: int) -> AnalysisStep:
        """
        Add a data filtering step to the pipeline

        This generates reproducible code that excludes specific files

        Args:
            filter_type: Type of filter ('countrate' or 'correlation')
            excluded_files: List of filenames that were excluded
            original_count: Number of files before filtering
            remaining_count: Number of files after filtering

        Returns:
            The created AnalysisStep
        """
        # Generate code for this filtering step
        if len(excluded_files) > 0:
            files_list_str = ',\n    '.join([f"'{f}'" for f in excluded_files])
            code = f"""
# Filtering step: Exclude {len(excluded_files)} files based on {filter_type}
# Original files: {original_count}, Remaining: {remaining_count}
files_to_exclude_{filter_type} = [
    {files_list_str}
]

# Filter {filter_type} data
{filter_type}_data = {{k: v for k, v in {filter_type}_data.items()
                if k not in files_to_exclude_{filter_type}}}

print(f"Filtered {filter_type}: {{len(files_to_exclude_{filter_type})}} files excluded, {{len({filter_type}_data)}} files remaining")

# Update basedata to match (if it exists)
if 'df_basedata' in locals() and len(df_basedata) > 0:
    df_basedata = df_basedata[~df_basedata['filename'].isin(files_to_exclude_{filter_type})]
    df_basedata = df_basedata.reset_index(drop=True)
    df_basedata.index = df_basedata.index + 1
"""
        else:
            code = f"""
# Filtering step: No files excluded from {filter_type}
# All {original_count} files kept
print(f"No files excluded from {filter_type} filtering - {{len({filter_type}_data)}} files kept")
"""

        # Create step with custom code
        step = AnalysisStep(
            name=f"Filter {filter_type.capitalize()}",
            step_type='filter',
            custom_code=code,
            params={
                'filter_type': filter_type,
                'excluded_files': excluded_files,
                'original_count': original_count,
                'remaining_count': remaining_count
            }
        )

        self.steps.append(step)

        # Emit signal
        self.step_added.emit(step.to_dict())

        return step

    def execute_step(self, step_index: int, *args, **kwargs):
        """Execute a specific step by index"""
        if 0 <= step_index < len(self.steps):
            step = self.steps[step_index]
            result = step.execute(*args, **kwargs)
            self.results[step.name] = result
            return result
        else:
            raise IndexError(f"Step index {step_index} out of range")

    def execute_all(self):
        """Execute all steps in the pipeline"""
        for i, step in enumerate(self.steps):
            print(f"Executing step {i+1}/{len(self.steps)}: {step.name}")
            self.execute_step(i)

        self.analysis_completed.emit(self.results)

    def load_data(self, data_folder: str):
        """
        Load data from folder

        Args:
            data_folder: Path to folder containing .asc files
        """
        import glob
        import os
        from preprocessing import extract_data, extract_correlation

        # Find all .asc files
        datafiles = glob.glob(os.path.join(data_folder, "*.asc"))
        filtered_files = [f for f in datafiles
                          if "averaged" not in os.path.basename(f).lower()]

        # Record this as a step
        self.add_step(
            "Load Data",
            self._load_data_internal,
            {'data_folder': data_folder, 'files': filtered_files}
        )

        # Store data
        self.data['data_folder'] = data_folder
        self.data['files'] = filtered_files
        self.data['num_files'] = len(filtered_files)

        # Emit signal
        self.data_loaded.emit(data_folder)

    def _load_data_internal(self, data_folder: str, files: List[str]):
        """Internal data loading function"""
        # This would call the actual preprocessing functions
        return {'folder': data_folder, 'count': len(files)}

    def get_current_code(self) -> str:
        """
        Generate Python code for all steps executed so far

        Returns:
            Python code as string
        """
        code_parts = []

        # Header
        code_parts.append(f"""
# JADE-DLS Analysis Script
# Auto-generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# Generated from GUI session
# FAIR-compliant data processing pipeline

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ade_dls.core.preprocessing import extract_data, extract_correlation
from ade_dls.analysis.cumulants import analyze_diffusion_coefficient
from ade_dls.analysis.regularized import nnls_reg_all
import glob
import os
""")

        # Add each step
        for i, step in enumerate(self.steps):
            code_parts.append(f"\n\n# ========== Step {i+1}: {step.name} ==========")
            code_parts.append(step.generate_code())

        # Add final export code
        code_parts.append(self._generate_export_code())

        return '\n'.join(code_parts)

    def _generate_export_code(self) -> str:
        """Generate code for final results export"""
        return """

# ========== Final Step: Export Results ==========

# Combine all results into a summary
all_results_list = []

# Add cumulant results if they exist
if 'method_a_results' in locals():
    all_results_list.append(method_a_results)
if 'method_b_results' in locals():
    all_results_list.append(method_b_results)
if 'method_c_results' in locals():
    all_results_list.append(method_c_results)
if 'method_c_results_refined' in locals():
    all_results_list.append(method_c_results_refined)

# Add NNLS results if they exist
if 'nnls_final_results_df' in locals():
    all_results_list.append(nnls_final_results_df)
if 'nnls_final_results_refined_df' in locals():
    all_results_list.append(nnls_final_results_refined_df)

# Add Regularized results if they exist
if 'regularized_final_results_df' in locals():
    all_results_list.append(regularized_final_results_df)
if 'regularized_final_results_refined_df' in locals():
    all_results_list.append(regularized_final_results_refined_df)

# Combine all results
if all_results_list:
    all_results = pd.concat(all_results_list, ignore_index=True)

    print("\\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    print(all_results.to_string())
    print("="*60)

    # Export to Excel
    output_file = "DLS_Analysis_Results.xlsx"
    with pd.ExcelWriter(output_file) as writer:
        all_results.to_excel(writer, sheet_name='Summary', index=False)

        # Export individual method results to separate sheets
        if 'method_a_results' in locals():
            method_a_results.to_excel(writer, sheet_name='Method_A', index=False)
        if 'method_b_results' in locals():
            method_b_results.to_excel(writer, sheet_name='Method_B', index=False)
        if 'method_c_results' in locals():
            method_c_results.to_excel(writer, sheet_name='Method_C', index=False)
        if 'method_c_results_refined' in locals():
            method_c_results_refined.to_excel(writer, sheet_name='Method_C_Refined', index=False)
        if 'nnls_final_results_df' in locals():
            nnls_final_results_df.to_excel(writer, sheet_name='NNLS', index=False)
        if 'nnls_final_results_refined_df' in locals():
            nnls_final_results_refined_df.to_excel(writer, sheet_name='NNLS_Refined', index=False)
        if 'regularized_final_results_df' in locals():
            regularized_final_results_df.to_excel(writer, sheet_name='Regularized', index=False)
        if 'regularized_final_results_refined_df' in locals():
            regularized_final_results_refined_df.to_excel(writer, sheet_name='Regularized_Refined', index=False)

    print(f"\\nResults exported to: {output_file}")

    # Also export to CSV for FAIR compliance
    csv_file = "DLS_Analysis_Results.csv"
    all_results.to_csv(csv_file, index=False)
    print(f"Results also exported to: {csv_file}")
else:
    print("\\nNo results to export.")
"""

    def export_to_script(self, filename: str):
        """
        Export pipeline to Python script

        Args:
            filename: Output filename
        """
        code = self.get_current_code()

        with open(filename, 'w') as f:
            f.write(code)

    def export_to_notebook(self, filename: str):
        """
        Export pipeline to Jupyter notebook

        Args:
            filename: Output filename (.ipynb)
        """
        import nbformat
        from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

        nb = new_notebook()

        # Title cell
        nb.cells.append(new_markdown_cell(f"""# JADE-DLS Analysis

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Data Folder:** {self.data.get('data_folder', 'N/A')}
**Number of Files:** {self.data.get('num_files', 0)}

This notebook was auto-generated from the JADE-DLS GUI and contains all analysis steps.
"""))

        # Imports cell
        nb.cells.append(new_code_cell("""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ade_dls.core.preprocessing import extract_data, extract_correlation
from ade_dls.analysis.cumulants import analyze_diffusion_coefficient
from ade_dls.analysis.regularized import nnls_reg_all
import glob
import os

%matplotlib inline"""))

        # Add each step as a separate cell
        for i, step in enumerate(self.steps):
            # Markdown description
            nb.cells.append(new_markdown_cell(f"""## Step {i+1}: {step.name}

**Timestamp:** {step.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
**Function:** `{step.function_name}`
**Parameters:**
```python
{json.dumps(step.params, indent=2)}
```
"""))

            # Code cell
            nb.cells.append(new_code_cell(step.generate_code()))

        # Write notebook
        with open(filename, 'w') as f:
            nbformat.write(nb, f)

    def export_to_pdf(self, filename: str, cumulant_analyzer=None):
        """
        Export complete analysis report to PDF

        Args:
            filename: Output PDF filename
            cumulant_analyzer: Optional CumulantAnalyzer instance with results
        """
        # This would use reportlab or similar to create a PDF
        # For now, we'll create a simple text-based report
        report = f"""
JADE-DLS Analysis Report
========================

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Data Folder: {self.data.get('data_folder', 'N/A')}
Number of Files: {self.data.get('num_files', 0)}

Analysis Steps:
---------------
"""
        for i, step in enumerate(self.steps):
            report += f"\n{i+1}. {step.name}\n"
            report += f"   Function: {step.function_name}\n"
            report += f"   Parameters: {step.params}\n"

        # Add Results Summary if cumulant analysis was performed
        if cumulant_analyzer is not None:
            report += self._format_cumulant_results(cumulant_analyzer)

        # For now, save as text (PDF generation would need reportlab)
        text_filename = filename.replace('.pdf', '.txt')
        with open(text_filename, 'w') as f:
            f.write(report)

    def _format_cumulant_results(self, analyzer) -> str:
        """
        Format cumulant analysis results for export

        Args:
            analyzer: CumulantAnalyzer instance

        Returns:
            Formatted string with results
        """
        report = "\n\nResults Summary:\n"
        report += "================\n\n"

        # Method A
        if hasattr(analyzer, 'method_a_results') and not analyzer.method_a_results.empty:
            report += "Method A - Linear Cumulant Analysis:\n"
            report += "-" * 60 + "\n\n"

            # Get regression stats
            regression_stats = getattr(analyzer, 'method_a_regression_stats', {})
            models_list = regression_stats.get('models', [])

            for i, row_idx in enumerate(analyzer.method_a_results.index):
                row = analyzer.method_a_results.loc[row_idx]
                fit_name = row['Fit']

                # Determine order
                if '1st' in fit_name:
                    order_label = "1st Order"
                elif '2nd' in fit_name:
                    order_label = "2nd Order"
                elif '3rd' in fit_name:
                    order_label = "3rd Order"
                else:
                    order_label = "Unknown Order"

                report += f"  {order_label}:\n"
                report += f"    Rh [nm]:              {row['Rh [nm]']:.2f} ± {row['Rh error [nm]']:.2f}\n"
                report += f"    PDI:                  {row['PDI']:.4f}\n"
                report += f"    R²:                   {row['R_squared']:.6f}\n"

                # Add detailed statistics if available
                if i < len(models_list):
                    model_stats = models_list[i]
                    report += f"    Adj. R²:              {model_stats.get('rsquared_adj', 0):.6f}\n"
                    report += f"    F-statistic:          {model_stats.get('fvalue', 0):.2f}\n"
                    report += f"    p-value (F):          {model_stats.get('f_pvalue', 0):.4e}\n"

                    if model_stats.get('condition_number') is not None:
                        report += f"    Condition Number:     {model_stats.get('condition_number', 0):.2f}\n"

                    report += f"    AIC:                  {model_stats.get('aic', 0):.2f}\n"
                    report += f"    BIC:                  {model_stats.get('bic', 0):.2f}\n"
                    report += f"    Observations:         {model_stats.get('nobs', 0)}\n"

                    # Regression coefficients
                    params = model_stats.get('params', {})
                    stderr_intercept = model_stats.get('stderr_intercept', 0)
                    stderr_slope = model_stats.get('stderr_slope', 0)

                    report += f"    Intercept:            {params.get('const', 0):.4e} ± {stderr_intercept:.4e}\n"

                    # Find slope key
                    slope_key = None
                    for key in params.keys():
                        if key != 'const':
                            slope_key = key
                            break
                    if slope_key:
                        report += f"    Slope (q²):           {params[slope_key]:.4e} ± {stderr_slope:.4e}\n"

                report += "\n"

        # Method B
        if hasattr(analyzer, 'method_b_results') and not analyzer.method_b_results.empty:
            report += "Method B - Linear Fit of ln[sqrt(g2)]:\n"
            report += "-" * 60 + "\n\n"

            row = analyzer.method_b_results.iloc[0]
            report += f"  Rh [nm]:              {row['Rh [nm]']:.2f} ± {row['Rh error [nm]']:.2f}\n"
            report += f"  PDI:                  {row['PDI']:.4f}\n"
            report += f"  R²:                   {row['R_squared']:.6f}\n"

            # Add detailed statistics if available
            regression_stats = getattr(analyzer, 'method_b_regression_stats', {})
            if regression_stats:
                report += f"  Adj. R²:              {regression_stats.get('rsquared_adj', 0):.6f}\n"
                report += f"  F-statistic:          {regression_stats.get('fvalue', 0):.2f}\n"
                report += f"  p-value (F):          {regression_stats.get('f_pvalue', 0):.4e}\n"

                if regression_stats.get('condition_number') is not None:
                    report += f"  Condition Number:     {regression_stats.get('condition_number', 0):.2f}\n"

                report += f"  AIC:                  {regression_stats.get('aic', 0):.2f}\n"
                report += f"  BIC:                  {regression_stats.get('bic', 0):.2f}\n"
                report += f"  Observations:         {regression_stats.get('nobs', 0)}\n"

                # Regression coefficients
                params = regression_stats.get('params', {})
                stderr_intercept = regression_stats.get('stderr_intercept', 0)
                stderr_slope = regression_stats.get('stderr_slope', 0)

                report += f"  Intercept:            {params.get('const', 0):.4e} ± {stderr_intercept:.4e}\n"

                # Find slope key
                slope_key = None
                for key in params.keys():
                    if key != 'const':
                        slope_key = key
                        break
                if slope_key:
                    report += f"  Slope (q²):           {params[slope_key]:.4e} ± {stderr_slope:.4e}\n"

            report += "\n"

        # Method C
        if hasattr(analyzer, 'method_c_results') and not analyzer.method_c_results.empty:
            report += "Method C - Iterative Non-Linear Fit:\n"
            report += "-" * 60 + "\n\n"

            row = analyzer.method_c_results.iloc[0]
            report += f"  Rh [nm]:              {row['Rh [nm]']:.2f} ± {row['Rh error [nm]']:.2f}\n"
            report += f"  PDI:                  {row['PDI']:.4f}\n"
            report += f"  R²:                   {row['R_squared']:.6f}\n"

            # Add detailed statistics if available
            regression_stats = getattr(analyzer, 'method_c_regression_stats', {})
            if regression_stats:
                report += f"  Adj. R²:              {regression_stats.get('rsquared_adj', 0):.6f}\n"
                report += f"  F-statistic:          {regression_stats.get('fvalue', 0):.2f}\n"
                report += f"  p-value (F):          {regression_stats.get('f_pvalue', 0):.4e}\n"

                if regression_stats.get('condition_number') is not None:
                    report += f"  Condition Number:     {regression_stats.get('condition_number', 0):.2f}\n"

                report += f"  AIC:                  {regression_stats.get('aic', 0):.2f}\n"
                report += f"  BIC:                  {regression_stats.get('bic', 0):.2f}\n"
                report += f"  Observations:         {regression_stats.get('nobs', 0)}\n"

                # Regression coefficients
                params = regression_stats.get('params', {})
                stderr_intercept = regression_stats.get('stderr_intercept', 0)
                stderr_slope = regression_stats.get('stderr_slope', 0)

                report += f"  Intercept:            {params.get('const', 0):.4e} ± {stderr_intercept:.4e}\n"

                # Find slope key
                slope_key = None
                for key in params.keys():
                    if key != 'const':
                        slope_key = key
                        break
                if slope_key:
                    report += f"  Slope (q²):           {params[slope_key]:.4e} ± {stderr_slope:.4e}\n"

            report += "\n"

        return report

    def get_parameter_history(self) -> pd.DataFrame:
        """
        Get history of all parameters used in the pipeline

        Returns:
            DataFrame with parameter history
        """
        history = []
        for step in self.steps:
            for param_name, param_value in step.params.items():
                history.append({
                    'step': step.name,
                    'timestamp': step.timestamp,
                    'parameter': param_name,
                    'value': param_value
                })

        return pd.DataFrame(history)

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of the pipeline

        Returns:
            Dictionary with pipeline summary
        """
        return {
            'total_steps': len(self.steps),
            'created': self.metadata['created'],
            'data_folder': self.data.get('data_folder'),
            'num_files': self.data.get('num_files', 0),
            'steps': [step.name for step in self.steps]
        }

    def clear(self):
        """Clear all steps and results"""
        self.steps = []
        self.results = {}

    def save_session(self, filename: str):
        """
        Save pipeline session to JSON

        Args:
            filename: Output JSON filename
        """
        session_data = {
            'metadata': {
                **self.metadata,
                'created': self.metadata['created'].isoformat()
            },
            'data': self.data,
            'steps': [step.to_dict() for step in self.steps]
        }

        with open(filename, 'w') as f:
            json.dump(session_data, f, indent=2)

    def load_session(self, filename: str):
        """
        Load pipeline session from JSON

        Args:
            filename: Input JSON filename
        """
        with open(filename, 'r') as f:
            session_data = json.load(f)

        self.metadata = session_data['metadata']
        self.data = session_data['data']

        # Note: Can't fully restore functions from JSON,
        # would need to re-create steps manually
        print("Session metadata loaded. Note: Step functions need manual restoration.")

    def _generate_method_c_refinement_code(self, q_range: tuple, excluded_files: List[str]) -> str:
        """Generate code for Method C post-refinement"""
        excluded_str = ""
        if excluded_files and len(excluded_files) > 0:
            files_list = ',\n    '.join([f"'{f}'" for f in excluded_files])
            excluded_str = f"""
# Files to exclude
excluded_files_c = [
    {files_list}
]

# Filter data
cumulant_method_C_data_refined = cumulant_method_C_data[
    ~cumulant_method_C_data['filename'].isin(excluded_files_c)
].reset_index(drop=True)
cumulant_method_C_data_refined.index = cumulant_method_C_data_refined.index + 1
print(f"Excluded {{len(excluded_files_c)}} files, {{len(cumulant_method_C_data_refined)}} remaining")
"""
        else:
            excluded_str = "\ncumulant_method_C_data_refined = cumulant_method_C_data.copy()\n"

        q_range_str = ""
        if q_range:
            q_range_str = f"""
# Apply q² range filter
q_range_c = {q_range}
mask = ((cumulant_method_C_data_refined['q^2'] >= q_range_c[0]) &
        (cumulant_method_C_data_refined['q^2'] <= q_range_c[1]))
cumulant_method_C_data_refined = cumulant_method_C_data_refined[mask].reset_index(drop=True)
cumulant_method_C_data_refined.index = cumulant_method_C_data_refined.index + 1
print(f"Applied q² range filter: {{len(cumulant_method_C_data_refined)}} points remaining")
"""

        return f"""
# Post-refinement for Method C
# Exclude outliers and recalculate with q² range filter

from ade_dls.analysis.cumulants import analyze_diffusion_coefficient
import numpy as np
{excluded_str}{q_range_str}
# Recalculate diffusion coefficient
cumulant_method_C_diff_refined = analyze_diffusion_coefficient(
    data_df=cumulant_method_C_data_refined,
    q_squared_col='q^2',
    gamma_cols=['best_b'],
    method_names=['Method C (Post-Refined)'],
    x_range={q_range if q_range else 'None'}
)

# Calculate refined results
C_diff_refined = pd.DataFrame()
C_diff_refined['D [m^2/s]'] = cumulant_method_C_diff_refined['q^2_coef'] * 10**(-18)
C_diff_refined['std err D [m^2/s]'] = cumulant_method_C_diff_refined['q^2_se'] * 10**(-18)

# Calculate polydispersity
cumulant_method_C_data_refined['polydispersity'] = (
    cumulant_method_C_data_refined['best_c'] / (cumulant_method_C_data_refined['best_b'])**2
)
polydispersity_method_C_refined = cumulant_method_C_data_refined['polydispersity'].mean()

# Calculate Rh
method_c_results_refined = pd.DataFrame()
method_c_results_refined['Rh [nm]'] = c * (1 / C_diff_refined['D [m^2/s]'][0]) * 10**9
fractional_error_Rh_C_refined = np.sqrt(
    (delta_c / c)**2 +
    (C_diff_refined['std err D [m^2/s]'][0] / C_diff_refined['D [m^2/s]'][0])**2
)
method_c_results_refined['Rh error [nm]'] = fractional_error_Rh_C_refined * method_c_results_refined['Rh [nm]']
method_c_results_refined['R_squared'] = cumulant_method_C_diff_refined['R_squared']
method_c_results_refined['Fit'] = 'Method C (Post-Refined)'
method_c_results_refined['Residuals'] = cumulant_method_C_diff_refined['Normality']
method_c_results_refined['PDI'] = polydispersity_method_C_refined

print("\\nMethod C Post-Refined Results:")
print(method_c_results_refined)
"""

    def _generate_nnls_refinement_code(self, q_range: tuple, excluded_files: List[str]) -> str:
        """Generate code for NNLS post-refinement"""
        excluded_str = ""
        if excluded_files and len(excluded_files) > 0:
            files_list = ',\n    '.join([f"'{f}'" for f in excluded_files])
            excluded_str = f"""
# Files to exclude
excluded_files_nnls = [
    {files_list}
]

# Filter data
nnls_data_refined = nnls_data[
    ~nnls_data['filename'].isin(excluded_files_nnls)
].reset_index(drop=True)
nnls_data_refined.index = nnls_data_refined.index + 1
print(f"[NNLS Post-Refinement] Removed {{len(excluded_files_nnls)}} datasets, {{len(nnls_data_refined)}} remaining")
"""
        else:
            excluded_str = "\nnnls_data_refined = nnls_data.copy()\n"

        return f"""
# Post-refinement for NNLS
# Apply q² range and exclude distributions

from ade_dls.analysis.peak_clustering import analyze_diffusion_coefficient_robust
import numpy as np
{excluded_str}
# Recalculate diffusion coefficients with new q² range (using robust regression)
q_range_nnls = {q_range if q_range else 'None'}

# Get gamma columns (should exist from initial analysis)
gamma_columns_refined = [col for col in nnls_data_refined.columns if col.startswith('gamma_')]

nnls_diff_results_refined = analyze_diffusion_coefficient_robust(
    data_df=nnls_data_refined,
    q_squared_col='q^2',
    gamma_cols=gamma_columns_refined,
    robust_method='ransac',
    x_range=q_range_nnls,
    show_plots=False
)

# Calculate refined Rh results
nnls_final_results_refined = []
for i in range(len(nnls_diff_results_refined)):
    if pd.notna(nnls_diff_results_refined['q^2_coef'][i]):
        # Convert D from nm²/s to m²/s
        D_m2s = nnls_diff_results_refined['q^2_coef'][i] * 1e-18
        D_err_m2s = nnls_diff_results_refined['q^2_se'][i] * 1e-18

        # Calculate Rh
        Rh_nm = c * (1 / D_m2s) * 1e9

        # Error propagation
        fractional_error = np.sqrt((delta_c / c)**2 + (D_err_m2s / D_m2s)**2)
        Rh_error_nm = fractional_error * Rh_nm

        result = {{
            'Rh [nm]': Rh_nm,
            'Rh error [nm]': Rh_error_nm,
            'D [m^2/s]': D_m2s,
            'D error [m^2/s]': D_err_m2s,
            'R_squared': nnls_diff_results_refined['R_squared'][i],
            'Fit': f'NNLS Peak {{i+1}} (Post-Refined)',
            'Residuals': nnls_diff_results_refined.get('Normality', [0])[i]
        }}
        nnls_final_results_refined.append(result)

nnls_final_results_refined_df = pd.DataFrame(nnls_final_results_refined)
print("\\nNNLS Post-Refined Results:")
print(nnls_final_results_refined_df)
"""

    def _generate_regularized_refinement_code(self, q_range: tuple, excluded_files: List[str]) -> str:
        """Generate code for Regularized NNLS post-refinement"""
        excluded_str = ""
        if excluded_files and len(excluded_files) > 0:
            files_list = ',\n    '.join([f"'{f}'" for f in excluded_files])
            excluded_str = f"""
# Files to exclude
excluded_files_reg = [
    {files_list}
]

# Filter data
regularized_data_refined = regularized_data[
    ~regularized_data['filename'].isin(excluded_files_reg)
].reset_index(drop=True)
regularized_data_refined.index = regularized_data_refined.index + 1
print(f"[Regularized Post-Refinement] Removed {{len(excluded_files_reg)}} datasets, {{len(regularized_data_refined)}} remaining")
"""
        else:
            excluded_str = "\nregularized_data_refined = regularized_data.copy()\n"

        return f"""
# Post-refinement for Regularized NNLS
# Apply q² range and exclude distributions

from ade_dls.analysis.peak_clustering import analyze_diffusion_coefficient_robust
import numpy as np
{excluded_str}
# Recalculate diffusion coefficients with new q² range
q_range_reg = {q_range if q_range else 'None'}

# Get gamma columns (should exist from initial analysis)
gamma_columns_refined = [col for col in regularized_data_refined.columns if col.startswith('gamma_')]

regularized_diff_results_refined = analyze_diffusion_coefficient_robust(
    data_df=regularized_data_refined,
    q_squared_col='q^2',
    gamma_cols=gamma_columns_refined,
    robust_method='ransac',
    x_range=q_range_reg,
    show_plots=False
)

# Calculate refined Rh results
regularized_final_results_refined = []
for i in range(len(regularized_diff_results_refined)):
    if pd.notna(regularized_diff_results_refined['q^2_coef'][i]):
        # Convert D from nm²/s to m²/s
        D_m2s = regularized_diff_results_refined['q^2_coef'][i] * 1e-18
        D_err_m2s = regularized_diff_results_refined['q^2_se'][i] * 1e-18

        # Calculate Rh
        Rh_nm = c * (1 / D_m2s) * 1e9

        # Error propagation
        fractional_error = np.sqrt((delta_c / c)**2 + (D_err_m2s / D_m2s)**2)
        Rh_error_nm = fractional_error * Rh_nm

        result = {{
            'Rh [nm]': Rh_nm,
            'Rh error [nm]': Rh_error_nm,
            'D [m^2/s]': D_m2s,
            'D error [m^2/s]': D_err_m2s,
            'R_squared': regularized_diff_results_refined['R_squared'][i],
            'Fit': f'Regularized Peak {{i+1}} (Post-Refined)',
            'Residuals': regularized_diff_results_refined.get('Normality', [0])[i]
        }}
        regularized_final_results_refined.append(result)

regularized_final_results_refined_df = pd.DataFrame(regularized_final_results_refined)
print("\\nRegularized NNLS Post-Refined Results:")
print(regularized_final_results_refined_df)
"""
