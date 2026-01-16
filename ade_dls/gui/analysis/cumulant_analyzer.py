"""
Cumulant Analyzer
Executes the three cumulant analysis methods and manages results
"""

import pandas as pd
import numpy as np
from scipy.constants import k as boltzmann_constant
from typing import Dict, List, Tuple, Any
import os


class CumulantAnalyzer:
    """
    Performs cumulant analysis on DLS data using multiple methods

    Methods:
    - Method A: Extract cumulants from ALV software
    - Method B: Linear fit method
    - Method C: Iterative non-linear fit
    """

    def __init__(self, loaded_data: Dict[str, Any], data_folder: str):
        """
        Initialize the analyzer

        Args:
            loaded_data: Dictionary containing filtered data
                - 'correlations': dict of correlation DataFrames
                - 'countrates': dict of countrate DataFrames
                - 'num_files': number of files
            data_folder: Path to folder with .asc files
        """
        self.loaded_data = loaded_data
        self.data_folder = data_folder

        # Results storage
        self.method_a_results = None
        self.method_b_results = None
        self.method_c_results = None

        # Intermediate data
        self.df_basedata = None
        self.processed_correlations = None
        self.c_value = None
        self.delta_c = None

    def prepare_basedata(self):
        """
        Prepare base data (temperature, viscosity, q, q^2)

        This is required for all cumulant methods
        """
        from preprocessing import extract_data

        # Get list of files from correlations (these are the filtered ones)
        filtered_files = list(self.loaded_data['correlations'].keys())

        # Create mapping from filename to full path (case-insensitive for Linux)
        import glob
        datafiles = []
        datafiles.extend(glob.glob(os.path.join(self.data_folder, "*.asc")))
        datafiles.extend(glob.glob(os.path.join(self.data_folder, "*.ASC")))
        file_to_path = {os.path.basename(f): f for f in datafiles}

        # Extract base data for filtered files only
        all_data = []
        for filename in filtered_files:
            if filename in file_to_path:
                file_path = file_to_path[filename]
                extracted_data = extract_data(file_path)
                if extracted_data is not None:
                    extracted_data['filename'] = filename
                    all_data.append(extracted_data)

        if all_data:
            self.df_basedata = pd.concat(all_data, ignore_index=True)
            self.df_basedata.index = self.df_basedata.index + 1
        else:
            raise ValueError("No basedata extracted!")

        # Calculate q and q^2
        self.df_basedata['q'] = abs(
            ((4 * np.pi * self.df_basedata['refractive_index']) /
             (self.df_basedata['wavelength [nm]'])) *
            np.sin(np.radians(self.df_basedata['angle [°]']) / 2)
        )
        self.df_basedata['q^2'] = self.df_basedata['q'] ** 2

        # Calculate mean temperature and viscosity
        mean_temperature = self.df_basedata['temperature [K]'].mean()
        std_temperature = self.df_basedata['temperature [K]'].std()

        mean_viscosity = self.df_basedata['viscosity [cp]'].mean()
        std_viscosity = self.df_basedata['viscosity [cp]'].std()

        # Calculate c = kb*T / (6*pi*eta) for Rh determination
        self.c_value = (
            (boltzmann_constant * mean_temperature) /
            (6 * np.pi * mean_viscosity * 10**(-3))
        )

        # Calculate error in c
        fractional_error_c = np.sqrt(
            (std_temperature / mean_temperature)**2 +
            (std_viscosity / mean_viscosity)**2
        )
        self.delta_c = fractional_error_c * self.c_value

        return self.df_basedata

    def prepare_processed_correlations(self):
        """
        Prepare processed correlation data

        Creates a dictionary with:
        - 't (s)': time in seconds
        - 'g(2)': mean of correlation detectors
        """
        from preprocessing import process_correlation_data

        columns_to_drop = ['time [ms]', 'correlation 1', 'correlation 2',
                          'correlation 3', 'correlation 4']

        self.processed_correlations = process_correlation_data(
            self.loaded_data['correlations'],
            columns_to_drop
        )

        return self.processed_correlations

    def run_method_a(self, q_range=None) -> pd.DataFrame:
        """
        Run Cumulant Method A

        Extracts cumulant fit data from ALV software output

        Args:
            q_range: Optional tuple (min_q, max_q) to restrict Diffusion Analysis

        Returns:
            DataFrame with results (Rh, errors, R^2, PDI)
        """
        from cumulants import extract_cumulants, calculate_cumulant_results_A
        from gui.analysis.cumulant_plotting import create_summary_plot
        import statsmodels.api as sm

        print("\n" + "="*60)
        print("CUMULANT METHOD A - ALV SOFTWARE CUMULANT DATA")
        print("="*60)
        if q_range:
            print(f"q² range restriction: {q_range[0]:.4e} - {q_range[1]:.4e} nm⁻²")

        # Ensure basedata is prepared
        if self.df_basedata is None:
            print("[Step 1/4] Preparing basedata...")
            self.prepare_basedata()

        # Get mapping from filename to full path (case-insensitive for Linux)
        print("[Step 2/4] Extracting cumulant data from ALV files...")
        import glob
        datafiles = []
        datafiles.extend(glob.glob(os.path.join(self.data_folder, "*.asc")))
        datafiles.extend(glob.glob(os.path.join(self.data_folder, "*.ASC")))
        file_to_path = {os.path.basename(f): f for f in datafiles}

        # Extract cumulant data
        all_cumulant_data = []
        for filename in self.loaded_data['correlations'].keys():
            if filename in file_to_path:
                file_path = file_to_path[filename]
                extracted_cumulants = extract_cumulants(file_path)
                if extracted_cumulants is not None:
                    extracted_cumulants['filename'] = filename
                    all_cumulant_data.append(extracted_cumulants)

        if not all_cumulant_data:
            raise ValueError("No cumulant data could be extracted from files!")

        print(f"            Extracted cumulant data from {len(all_cumulant_data)} files")

        df_extracted_cumulants = pd.concat(all_cumulant_data, ignore_index=True)
        df_extracted_cumulants.index = df_extracted_cumulants.index + 1

        # Merge with basedata
        cumulant_method_A_data = pd.merge(
            self.df_basedata,
            df_extracted_cumulants,
            on='filename',
            how='outer'
        )
        cumulant_method_A_data = cumulant_method_A_data.reset_index(drop=True)
        cumulant_method_A_data.index = cumulant_method_A_data.index + 1

        # Store unfiltered data for post-fit refinement
        self.method_a_data = cumulant_method_A_data.copy()

        # Apply q-range filter if specified
        if q_range is not None:
            min_q, max_q = q_range
            mask = (cumulant_method_A_data['q^2'] >= min_q) & (cumulant_method_A_data['q^2'] <= max_q)
            cumulant_method_A_data = cumulant_method_A_data[mask].reset_index(drop=True)
            cumulant_method_A_data.index = cumulant_method_A_data.index + 1
            print(f"            Applied q² range filter: {min_q:.4e} - {max_q:.4e} nm⁻²")
            print(f"            {len(cumulant_method_A_data)} data points remaining")

        # Perform linear regression for each gamma column without showing plots
        print("[Step 3/4] Performing linear regression for each order...")
        gamma_cols = ['1st order frequency [1/ms]',
                     '2nd order frequency [1/ms]',
                     '3rd order frequency [1/ms]']

        results_list = []
        for i, gamma_col in enumerate(gamma_cols, 1):
            print(f"            Order {i}: {gamma_col}")
            if gamma_col in cumulant_method_A_data.columns:
                X = cumulant_method_A_data['q^2']
                Y = cumulant_method_A_data[gamma_col]
                X_with_const = sm.add_constant(X)
                model = sm.OLS(Y, X_with_const).fit()

                results_list.append({
                    'gamma_col': gamma_col,
                    'q^2_coef': model.params.iloc[1],
                    'q^2_se': model.bse.iloc[1],
                    'R_squared': model.rsquared,
                    'Normality': 'Normal' if model.rsquared > 0.9 else 'Check'
                })

        cumulant_method_A_diff = pd.DataFrame(results_list)

        # Create summary plot (without showing)
        self.method_a_summary_plot = create_summary_plot(
            cumulant_method_A_data,
            'q^2',
            gamma_cols,
            ['1st order', '2nd order', '3rd order'],
            '1/ms'
        )

        # Create DataFrame with diffusion coefficients
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

        # Debug: Print data before calling calculate_cumulant_results_A
        print(f"[CUMULANT METHOD A DEBUG] Data before calculate_cumulant_results_A:")
        print(f"  A_diff shape: {A_diff.shape}")
        print(f"  A_diff:\n{A_diff}")
        print(f"  cumulant_method_A_diff shape: {cumulant_method_A_diff.shape}")
        print(f"  cumulant_method_A_diff:\n{cumulant_method_A_diff}")
        print(f"  polydispersity_method_A_2: {polydispersity_method_A_2}")
        print(f"  polydispersity_method_A_3: {polydispersity_method_A_3}")
        print(f"  c_value: {self.c_value}")
        print(f"  delta_c: {self.delta_c}")

        # Calculate final results
        self.method_a_results = calculate_cumulant_results_A(
            A_diff,
            cumulant_method_A_diff,
            polydispersity_method_A_2,
            polydispersity_method_A_3,
            self.c_value,
            self.delta_c
        )

        print(f"[CUMULANT METHOD A DEBUG] Final results:")
        print(f"  Shape: {self.method_a_results.shape}")
        print(f"  Columns: {self.method_a_results.columns.tolist()}")
        print(self.method_a_results)

        # Store regression statistics for detailed output
        self.method_a_regression_stats = {
            'gamma_cols': gamma_cols,
            'regression_results': results_list,
            'diffusion_data': A_diff
        }

        return self.method_a_results

    def run_method_b(self, fit_limits: Tuple[float, float], q_range=None) -> pd.DataFrame:
        """
        Run Cumulant Method B

        Uses linear fit method

        Args:
            fit_limits: Tuple of (min_time, max_time) for fitting
            q_range: Optional tuple (min_q, max_q) to restrict Diffusion Analysis

        Returns:
            DataFrame with results (Rh, errors, R^2, PDI)
        """
        from cumulants import calculate_g2_B, analyze_diffusion_coefficient
        from gui.analysis.cumulant_plotting import plot_processed_correlations_no_show, create_summary_plot

        # Store fit_limits for post-fit refinement
        self.method_b_fit_limits = fit_limits

        print("\n" + "="*60)
        print("CUMULANT METHOD B - LINEAR FIT")
        print("="*60)
        print(f"Fit limits: {fit_limits[0]:.6f} - {fit_limits[1]:.6f} s")
        if q_range:
            print(f"q² range restriction: {q_range[0]:.4e} - {q_range[1]:.4e} nm⁻²")

        # Ensure data is prepared
        if self.df_basedata is None:
            print("[Step 1/5] Preparing basedata...")
            self.prepare_basedata()

        if self.processed_correlations is None:
            print("[Step 2/5] Preparing processed correlations...")
            self.prepare_processed_correlations()

        # Calculate sqrt(g2)
        print("[Step 3/5] Calculating sqrt(g2) and dropping negative values...")
        processed_correlations = calculate_g2_B(self.processed_correlations)

        # Define fit function (up to 1st moment extension)
        def fit_function(x, a, b, c):
            return 0.5 * np.log(a) - b * x + 0.5 * c * x**2

        # Plot and fit (without showing)
        print(f"[Step 4/5] Fitting {len(processed_correlations)} correlation functions...")
        cumulant_method_B_fit, self.method_b_plots = plot_processed_correlations_no_show(
            processed_correlations,
            fit_function,
            fit_limits
        )
        print(f"            Successfully fitted {len(cumulant_method_B_fit)} datasets")

        # Merge with basedata
        cumulant_method_B_data = pd.merge(
            self.df_basedata,
            cumulant_method_B_fit,
            on='filename',
            how='outer'
        )
        cumulant_method_B_data = cumulant_method_B_data.reset_index(drop=True)
        cumulant_method_B_data.index = cumulant_method_B_data.index + 1

        # Store unfiltered data for post-fit refinement
        self.method_b_data = cumulant_method_B_data.copy()

        # Apply q-range filter if specified
        if q_range is not None:
            min_q, max_q = q_range
            mask = (cumulant_method_B_data['q^2'] >= min_q) & (cumulant_method_B_data['q^2'] <= max_q)
            cumulant_method_B_data = cumulant_method_B_data[mask].reset_index(drop=True)
            cumulant_method_B_data.index = cumulant_method_B_data.index + 1
            print(f"            Applied q² range filter: {min_q:.4e} - {max_q:.4e} nm⁻²")
            print(f"            {len(cumulant_method_B_data)} data points remaining")

        # Analyze diffusion coefficient (without plotting)
        # We'll create our own summary plot
        print(f"[Step 5/5] Analyzing diffusion coefficient (Γ vs q²)...")
        import statsmodels.api as sm

        # Linear regression
        X = cumulant_method_B_data['q^2']
        Y = cumulant_method_B_data['b']
        X_with_const = sm.add_constant(X)
        model = sm.OLS(Y, X_with_const).fit()

        print(f"            Slope (D): {model.params.iloc[1]:.4e} s⁻¹·nm²")
        print(f"            R²: {model.rsquared:.6f}")

        # Create summary plot
        self.method_b_summary_plot = create_summary_plot(
            cumulant_method_B_data,
            'q^2',
            ['b'],
            ['Method B'],
            '1/s'
        )

        # Extract fit quality metrics
        self.method_b_fit_quality = {}
        for filename in cumulant_method_B_fit['filename']:
            row = cumulant_method_B_fit[cumulant_method_B_fit['filename'] == filename]
            if not row.empty:
                self.method_b_fit_quality[filename] = {
                    'R2': row['R-squared'].values[0] if 'R-squared' in row else 0,
                    'residuals': 'Normal'
                }

        # Calculate diffusion coefficients
        B_diff = pd.DataFrame()
        B_diff['D [m^2/s]'] = [model.params.iloc[1] * 10**(-18)]
        B_diff['std err D [m^2/s]'] = [model.bse.iloc[1] * 10**(-18)]

        # Calculate polydispersity
        cumulant_method_B_data['polydispersity'] = (
            cumulant_method_B_data['c'] / (cumulant_method_B_data['b'])**2
        )
        polydispersity_method_B = cumulant_method_B_data['polydispersity'].mean()

        # Calculate final results - use lists to ensure DataFrame rows are created
        print("\nCalculating hydrodynamic radius...")
        print(f"  D = {B_diff['D [m^2/s]'][0]:.4e} ± {B_diff['std err D [m^2/s]'][0]:.4e} m²/s")

        rh_value = self.c_value * (1 / B_diff['D [m^2/s]'][0]) * 10**9

        fractional_error_Rh_B = np.sqrt(
            (self.delta_c / self.c_value)**2 +
            (B_diff['std err D [m^2/s]'][0] / B_diff['D [m^2/s]'][0])**2
        )
        rh_error_value = fractional_error_Rh_B * rh_value

        # Create DataFrame with lists to ensure rows are created
        self.method_b_results = pd.DataFrame({
            'Rh [nm]': [rh_value],
            'Rh error [nm]': [rh_error_value],
            'R_squared': [model.rsquared],
            'Fit': ['Rh from linear cumulant fit'],
            'Residuals': ['N/A'],
            'PDI': [polydispersity_method_B]
        })

        print(f"\nFINAL RESULTS:")
        print(f"  Rh = {rh_value:.2f} ± {rh_error_value:.2f} nm")
        print(f"  PDI = {polydispersity_method_B:.4f}")
        print(f"  R² = {model.rsquared:.6f}")
        print("="*60 + "\n")

        # Store regression statistics as strings/dicts (not model object)
        self.method_b_regression_stats = {
            'summary': str(model.summary()),
            'params': model.params.to_dict(),
            'stderr_intercept': float(model.bse.iloc[0]),
            'stderr_slope': float(model.bse.iloc[1]),
            'rsquared': float(model.rsquared),
            'rsquared_adj': float(model.rsquared_adj),
            'fvalue': float(model.fvalue),
            'f_pvalue': float(model.f_pvalue),
            'aic': float(model.aic),
            'bic': float(model.bic)
        }

        return self.method_b_results

    def run_method_c(self, params: Dict[str, Any], q_range=None) -> pd.DataFrame:
        """
        Run Cumulant Method C

        Uses iterative non-linear fit method

        Args:
            params: Dictionary with parameters:
                - fit_limits: tuple of (min, max) times
                - fit_function: str ('fit_function2', 'fit_function3', 'fit_function4')
                - adaptive_initial_guesses: bool
                - adaptation_strategy: str ('individual', 'global', 'representative')
                - optimizer: str ('lm', 'trf', 'dogbox')
                - initial_parameters: list of initial parameter guesses
            q_range: Optional tuple (min_q, max_q) to restrict Diffusion Analysis

        Returns:
            DataFrame with results (Rh, errors, R^2, PDI)
        """
        from cumulants_C import get_adaptive_initial_parameters, get_meaningful_parameters
        from gui.analysis.cumulant_plotting import plot_processed_correlations_iterative_no_show, create_summary_plot

        # Ensure data is prepared
        if self.df_basedata is None:
            self.prepare_basedata()

        if self.processed_correlations is None:
            self.prepare_processed_correlations()

        # Define fit functions
        def fit_function2(x, a, b, c, f):
            inner_term = 1 + 0.5 * c * x**2
            term = f + a * (np.exp(-b * x) * inner_term)**2
            return term

        def fit_function3(x, a, b, c, d, f):
            inner_term = 1 + 0.5 * c * x**2 - (d * x**3) / 6
            term = f + a * (np.exp(-b * x) * inner_term)**2
            return term

        def fit_function4(x, a, b, c, d, e, f):
            inner_term = 1 + 0.5 * c * x**2 - (d * x**3) / 6 + ((e - 3 * c**2) * x**4) / 24
            term = f + a * (np.exp(-b * x) * inner_term)**2
            return term

        # Select fit function
        fit_func_map = {
            'fit_function2': fit_function2,
            'fit_function3': fit_function3,
            'fit_function4': fit_function4
        }
        chosen_fit_function = fit_func_map[params['fit_function']]

        # Get initial parameters
        if params['adaptive_initial_guesses']:
            initial_parameters = get_adaptive_initial_parameters(
                self.processed_correlations,
                chosen_fit_function,
                params['initial_parameters'],
                strategy=params['adaptation_strategy'],
                verbose=False
            )
        else:
            initial_parameters = get_meaningful_parameters(
                chosen_fit_function,
                params['initial_parameters']
            )

        # Run fitting (without showing plots)
        use_multiprocessing = params.get('use_multiprocessing', False)
        print(f"[Method C] Multiprocessing: {'Enabled' if use_multiprocessing else 'Disabled'}")

        self.method_c_fit, self.method_c_plots = plot_processed_correlations_iterative_no_show(
            self.processed_correlations,
            chosen_fit_function,
            params['fit_limits'],
            initial_parameters,
            method=params['optimizer'],
            use_multiprocessing=use_multiprocessing
        )

        # Merge with basedata
        cumulant_method_C_data = pd.merge(
            self.df_basedata,
            self.method_c_fit,
            on='filename',
            how='outer'
        )
        cumulant_method_C_data = cumulant_method_C_data.reset_index(drop=True)
        cumulant_method_C_data.index = cumulant_method_C_data.index + 1

        # Store unfiltered data for post-fit refinement
        self.method_c_data = cumulant_method_C_data.copy()

        # Store q_range for potential recomputation
        self.last_q_range = q_range

        # Apply q-range filter if specified
        if q_range is not None:
            min_q, max_q = q_range
            mask = (cumulant_method_C_data['q^2'] >= min_q) & (cumulant_method_C_data['q^2'] <= max_q)
            cumulant_method_C_data = cumulant_method_C_data[mask].reset_index(drop=True)
            cumulant_method_C_data.index = cumulant_method_C_data.index + 1
            print(f"[METHOD C] Applied q² range filter: {min_q} - {max_q} nm⁻², {len(cumulant_method_C_data)} points remaining")

        # Create summary plot
        self.method_c_summary_plot = create_summary_plot(
            cumulant_method_C_data,
            'q^2',
            ['best_b'],
            ['Method C'],
            '1/s'
        )

        # Extract fit quality metrics
        self.method_c_fit_quality = {}
        for filename in self.method_c_fit['filename']:
            row = self.method_c_fit[self.method_c_fit['filename'] == filename]
            if not row.empty:
                self.method_c_fit_quality[filename] = {
                    'R2': row['R_squared'].values[0] if 'R_squared' in row else 0,
                    'residuals': row['RMSE'].values[0] if 'RMSE' in row else 0
                }

        # Linear regression for diffusion coefficient
        import statsmodels.api as sm

        # Debug: Check data before regression
        print(f"[CUMULANT METHOD C DEBUG] Data for regression:")
        print(f"  q^2 shape: {cumulant_method_C_data['q^2'].shape}")
        print(f"  best_b shape: {cumulant_method_C_data['best_b'].shape}")
        print(f"  q^2 sample: {cumulant_method_C_data['q^2'].head()}")
        print(f"  best_b sample: {cumulant_method_C_data['best_b'].head()}")
        print(f"  q^2 has NaN: {cumulant_method_C_data['q^2'].isna().any()}")
        print(f"  best_b has NaN: {cumulant_method_C_data['best_b'].isna().any()}")

        X = cumulant_method_C_data['q^2']
        Y = cumulant_method_C_data['best_b']
        X_with_const = sm.add_constant(X)
        model = sm.OLS(Y, X_with_const).fit()

        print(f"[CUMULANT METHOD C DEBUG] Regression results:")
        print(f"  params: {model.params}")
        print(f"  R²: {model.rsquared}")
        print(f"  slope (params[1]): {model.params.iloc[1]}")

        # Create DataFrame with diffusion coefficients
        C_diff = pd.DataFrame()
        C_diff['D [m^2/s]'] = [model.params.iloc[1] * 10**(-18)]
        C_diff['std err D [m^2/s]'] = [model.bse.iloc[1] * 10**(-18)]

        print(f"[CUMULANT METHOD C DEBUG] Diffusion coefficient:")
        print(f"  D [m^2/s]: {C_diff['D [m^2/s]'][0]}")
        print(f"  c_value: {self.c_value}")
        print(f"  Rh calculation: {self.c_value} * (1 / {C_diff['D [m^2/s]'][0]}) * 10^9")

        # Calculate polydispersity
        cumulant_method_C_data['polydispersity'] = (
            cumulant_method_C_data['best_c'] / (cumulant_method_C_data['best_b'])**2
        )
        polydispersity_method_C = cumulant_method_C_data['polydispersity'].mean()

        # Calculate final results
        self.method_c_results = pd.DataFrame()

        # Calculate Rh step by step for debugging
        rh_value = self.c_value * (1 / C_diff['D [m^2/s]'][0]) * 10**9
        print(f"[CUMULANT METHOD C DEBUG] Rh calculation result:")
        print(f"  Calculated Rh: {rh_value}")
        print(f"  Type: {type(rh_value)}")

        self.method_c_results['Rh [nm]'] = [rh_value]

        fractional_error_Rh_C = np.sqrt(
            (self.delta_c / self.c_value)**2 +
            (C_diff['std err D [m^2/s]'][0] / C_diff['D [m^2/s]'][0])**2
        )
        rh_error = fractional_error_Rh_C * rh_value
        print(f"  Calculated Rh error: {rh_error}")

        self.method_c_results['Rh error [nm]'] = [rh_error]
        self.method_c_results['R_squared'] = [model.rsquared]
        self.method_c_results['Fit'] = ['Rh from iterative non-linear cumulant fit']
        self.method_c_results['Residuals'] = ['N/A']
        self.method_c_results['PDI'] = [polydispersity_method_C]

        print(f"[CUMULANT METHOD C DEBUG] Final DataFrame:")
        print(self.method_c_results)

        # Store regression statistics as strings/dicts (not model object)
        self.method_c_regression_stats = {
            'summary': str(model.summary()),
            'params': model.params.to_dict(),
            'stderr_intercept': float(model.bse.iloc[0]),
            'stderr_slope': float(model.bse.iloc[1]),
            'rsquared': float(model.rsquared),
            'rsquared_adj': float(model.rsquared_adj),
            'fvalue': float(model.fvalue),
            'f_pvalue': float(model.f_pvalue),
            'aic': float(model.aic),
            'bic': float(model.bic)
        }

        return self.method_c_results

    def recompute_method_c_diffusion(self, included_files: list, q_range=None) -> pd.DataFrame:
        """
        Recompute Method C Diffusion Analysis with only selected files

        Used for post-fit filtering - recompute with only good fits

        Args:
            included_files: List of filenames to include
            q_range: Optional tuple (min_q, max_q) to restrict Diffusion Analysis

        Returns:
            DataFrame with recomputed results (Rh, errors, R^2, PDI)
        """
        import statsmodels.api as sm
        from gui.analysis.cumulant_plotting import create_summary_plot

        # Check if Method C data exists
        if not hasattr(self, 'method_c_fit') or self.method_c_fit is None:
            raise ValueError("Method C must be run first before recomputing!")

        # Filter the fit data to only include selected files
        cumulant_method_C_fit_filtered = self.method_c_fit[
            self.method_c_fit['filename'].isin(included_files)
        ].copy()

        if len(cumulant_method_C_fit_filtered) == 0:
            raise ValueError("No files selected for recomputation!")

        # Merge with basedata
        cumulant_method_C_data = pd.merge(
            self.df_basedata,
            cumulant_method_C_fit_filtered,
            on='filename',
            how='inner'  # Only include files that exist in both
        )
        cumulant_method_C_data = cumulant_method_C_data.reset_index(drop=True)
        cumulant_method_C_data.index = cumulant_method_C_data.index + 1

        # Apply q-range filter if specified
        if q_range is not None:
            min_q, max_q = q_range
            mask = (cumulant_method_C_data['q^2'] >= min_q) & (cumulant_method_C_data['q^2'] <= max_q)
            cumulant_method_C_data = cumulant_method_C_data[mask].reset_index(drop=True)
            cumulant_method_C_data.index = cumulant_method_C_data.index + 1
            print(f"[METHOD C RECOMPUTE] Applied q² range filter: {min_q} - {max_q} nm⁻², {len(cumulant_method_C_data)} points remaining")

        print(f"[METHOD C RECOMPUTE] Recomputing with {len(cumulant_method_C_data)} data points")

        # Linear regression for diffusion coefficient
        X = cumulant_method_C_data['q^2']
        Y = cumulant_method_C_data['best_b']
        X_with_const = sm.add_constant(X)
        model = sm.OLS(Y, X_with_const).fit()

        # Create DataFrame with diffusion coefficients
        C_diff = pd.DataFrame()
        C_diff['D [m^2/s]'] = [model.params.iloc[1] * 10**(-18)]
        C_diff['std err D [m^2/s]'] = [model.bse.iloc[1] * 10**(-18)]

        # Calculate polydispersity
        cumulant_method_C_data['polydispersity'] = (
            cumulant_method_C_data['best_c'] / (cumulant_method_C_data['best_b'])**2
        )
        polydispersity_method_C = cumulant_method_C_data['polydispersity'].mean()

        # Calculate final results
        recomputed_results = pd.DataFrame()
        rh_value = self.c_value * (1 / C_diff['D [m^2/s]'][0]) * 10**9
        recomputed_results['Rh [nm]'] = [rh_value]

        fractional_error_Rh_C = np.sqrt(
            (self.delta_c / self.c_value)**2 +
            (C_diff['std err D [m^2/s]'][0] / C_diff['D [m^2/s]'][0])**2
        )
        rh_error = fractional_error_Rh_C * rh_value
        recomputed_results['Rh error [nm]'] = [rh_error]
        recomputed_results['R_squared'] = [model.rsquared]
        recomputed_results['Fit'] = [f'Method C filtered (N={len(included_files)})']
        recomputed_results['Residuals'] = ['N/A']
        recomputed_results['PDI'] = [polydispersity_method_C]

        print(f"[METHOD C RECOMPUTE] Recomputed Rh: {rh_value:.2f} ± {rh_error:.2f} nm")
        print(f"[METHOD C RECOMPUTE] R²: {model.rsquared:.4f}")

        return recomputed_results

    def get_combined_results(self) -> pd.DataFrame:
        """
        Get combined results from all methods

        Returns:
            DataFrame with all results concatenated
        """
        results = []

        if self.method_a_results is not None:
            results.append(self.method_a_results)

        if self.method_b_results is not None:
            results.append(self.method_b_results)

        if self.method_c_results is not None:
            results.append(self.method_c_results)

        if results:
            return pd.concat(results, ignore_index=True)
        else:
            return pd.DataFrame()
