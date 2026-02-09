# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 13:53:48 2025

@author: vinci

cumulant analysis: general, methods A and B
"""
import pandas as pd
import re
import numpy as np
from scipy.optimize import curve_fit
import inspect
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm


#takes cumulant data from ALV-datafile and ensures robust encoding handling
#function to extract cumulants
def extract_cumulants(file_path, encodings=['Windows-1252', 'utf-8', 'latin1', 'ISO-8859-1']):
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                lines = file.readlines()

                #extract cumulant 1st order
                cumulant_1st_order = None
                found_cumulant = False  
                for i, line in enumerate(lines):
                    if "Cumulant 1.Order" in line:
                        found_cumulant = True
                        try:
                            next_line = lines[i + 1]
                            match = re.search(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", next_line)
                            if match:
                                cumulant_1st_order = float(match.group(0))
                            else:
                                return None 
                        except (ValueError, IndexError):
                            print(f"Error extracting 1st order cumulant from {file_path}")
                        break 

                if not found_cumulant:
                    print(f"1st order cumulant not found in {file_path}")

                #extract cumulant 2nd order
                cumulant_2nd_order = None
                found_cumulant = False  
                for i, line in enumerate(lines):
                    if "Cumulant 2.Order" in line:
                        found_cumulant = True
                        try:
                            next_line = lines[i + 1]
                            match = re.search(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", next_line)
                            if match:
                                cumulant_2nd_order = float(match.group(0))
                            else:
                                return None 
                        except (ValueError, IndexError):
                            print(f"Error extracting 2nd order cumulant from {file_path}")
                        break 

                if not found_cumulant:
                    print(f"2nd order cumulant not found in {file_path}")

                #extract cumulant 2nd order expansion parameter
                cumulant_2nd_order_exp_param = None
                found_cumulant = False  
                for i, line in enumerate(lines):
                    if "Cumulant 2.Order" in line:
                        found_cumulant = True
                        try:
                            next_line = lines[i + 4]
                            match = re.search(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", next_line)
                            if match:
                                cumulant_2nd_order_exp_param = float(match.group(0))
                            else:
                                return None 
                        except (ValueError, IndexError):
                            print(f"Error extracting 2nd order cumulant expansion parameter from {file_path}")
                        break 

                if not found_cumulant:
                    print(f"2nd order cumulant not found in {file_path}")
                    
                #extract cumulant 3rd order
                cumulant_3rd_order = None
                found_cumulant = False  
                for i, line in enumerate(lines):
                    if "Cumulant 3.Order" in line:
                        found_cumulant = True
                        try:
                            next_line = lines[i + 1]
                            match = re.search(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", next_line)
                            if match:
                                cumulant_3rd_order = float(match.group(0))
                            else:
                                return None 
                        except (ValueError, IndexError):
                            print(f"Error extracting 3rd order cumulant from {file_path}")
                        break 

                if not found_cumulant:
                    print(f"3rd order cumulant not found in {file_path}")
                    
                #extract cumulant 3rd order expansion parameter
                cumulant_3rd_order_exp_param = None
                found_cumulant = False  
                for i, line in enumerate(lines):
                    if "Cumulant 3.Order" in line:
                        found_cumulant = True
                        try:
                            next_line = lines[i + 4]
                            match = re.search(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", next_line)
                            if match:
                                cumulant_3rd_order_exp_param = float(match.group(0))
                            else:
                                return None 
                        except (ValueError, IndexError):
                            print(f"Error extracting 3rd order cumulant expansion parameter from {file_path}")
                        break 

                if not found_cumulant:
                    print(f"3rd order cumulant not found in {file_path}")
                    
                data = {'1st order frequency [1/ms]': [cumulant_1st_order], 
                        '2nd order frequency [1/ms]': [cumulant_2nd_order], 
                        '3rd order frequency [1/ms]': [cumulant_3rd_order],
                        '2nd order frequency exp param [ms^2]': [cumulant_2nd_order_exp_param],
                        '3rd order frequency exp param [ms^2]': [cumulant_3rd_order_exp_param]}
                return pd.DataFrame(data)

        except UnicodeDecodeError:
            continue
        except FileNotFoundError as e:
            print(f"File not found: {file_path}. Error: {e}")
            return None
        except Exception as e:
            print(f"An error occurred while processing {file_path}: {e}")
            return None
    
    print(f"Failed to extract cumulants from {file_path} with any encoding")
    return None

#calculation of cumulant results, specific for method A
def calculate_cumulant_results_A(A_diff, cumulant_method_A_diff, polydispersity_method_A_2, polydispersity_method_A_3, c, delta_c):
    results = []
    orders = [1, 2, 3]
    pdi_values = [None, polydispersity_method_A_2, polydispersity_method_A_3]

    for i, order in enumerate(orders):
        # Calculate values as scalars
        rh_value = c * (1 / A_diff['D [m^2/s]'][i]) * 10**9
        fractional_error_Rh = np.sqrt((delta_c / c)**2 + (A_diff['std err D [m^2/s]'][i] / A_diff['D [m^2/s]'][i])**2)
        rh_error_value = fractional_error_Rh * rh_value
        r2_value = cumulant_method_A_diff['R_squared'][i]
        fit_name = f'Rh from {order}st order cumulant fit' if order == 1 else f'Rh from {order}nd order cumulant fit' if order == 2 else f'Rh from {order}rd order cumulant fit'
        normality_value = cumulant_method_A_diff['Normality'][i]

        # Create DataFrame with lists to ensure rows are created
        result = pd.DataFrame({
            'Rh [nm]': [rh_value],
            'Rh error [nm]': [rh_error_value],
            'R_squared': [r2_value],
            'Fit': [fit_name],
            'Residuals': [normality_value]
        })

        if i > 0:  # PDI is only for 2nd and 3rd order
            result['PDI'] = [pdi_values[i]]

        results.append(result)

    df_cumulant_method_A_results = pd.concat(results, ignore_index=True)
    return df_cumulant_method_A_results

#create zero DataFrame directly, in case of not wanting to perform the cumulant fit mit method A
def create_zero_cumulant_results_A():
    results = []
    orders = [1, 2, 3]
    
    for i, order in enumerate(orders):
        result = pd.DataFrame({
            'Rh [nm]': [0],
            'Rh error [nm]': [0], 
            'R_squared': [0],
            'Fit': [f'Rh from {order}st order cumulant fit' if order == 1 else f'Rh from {order}nd order cumulant fit' if order == 2 else f'Rh from {order}rd order cumulant fit'],
            'Residuals': [0]
        })
        if i > 0:
            result['PDI'] = [0]
        results.append(result)
    
    return pd.concat(results, ignore_index=True)

#calculation of g2-values for cumulant-method-B
def calculate_g2_B(dataframes_dict):
    processed_dataframes = {}

    for name, df in dataframes_dict.items():
        try:
            df_copy = df.copy(deep=True) 

            g2_values = df_copy['g(2)']
            
            #drop negative values
            neg_or_zero_mask = g2_values <= 0
            if neg_or_zero_mask.any():
                num_neg_or_zero = neg_or_zero_mask.sum()
                print(f"Warning: DataFrame '{name}' contains {num_neg_or_zero} non-positive g(2) values. These rows will be dropped.")

            rows_to_keep = g2_values > 0
            df_copy = df_copy[rows_to_keep]  
            df_copy = df_copy.reset_index(drop=True) 

            df_copy['g(2)_mod'] = np.sqrt(df_copy['g(2)'])

            processed_dataframes[name] = df_copy  

        except (KeyError, TypeError, ValueError) as e:
            print(f"Warning: Error processing DataFrame '{name}': {e}")

    return processed_dataframes


#plot Gamma vs. q^2
def analyze_diffusion_coefficient(data_df, q_squared_col, gamma_cols, method_names=None, gamma_unit='1/s', x_range=None, show_plots=True):
    #validate inputs
    if not isinstance(gamma_cols, list):
        gamma_cols = [gamma_cols]
    
    #set up method names if not provided
    if method_names is None:
        method_names = gamma_cols
    elif not isinstance(method_names, list):
        method_names = [method_names]
    
    #make sure method_names has the same length as gamma_cols
    if len(method_names) < len(gamma_cols):
        method_names.extend(['' for _ in range(len(gamma_cols) - len(method_names))])
    
    #check if q_squared_col exists in data_df
    if q_squared_col not in data_df.columns:
        raise ValueError(f"Column '{q_squared_col}' not found in the DataFrame")
    
    #X-data
    X_full = data_df[q_squared_col]
    
    all_results = []
    
    #iterate through each gamma column
    for i, gamma_col in enumerate(gamma_cols):
        #check if column exists in the dataframe
        if gamma_col not in data_df.columns:
            print(f"Column '{gamma_col}' not found in the DataFrame. Skipping...")
            continue
        
        method_name = method_names[i] if i < len(method_names) else ''
        
        #Y-data (full dataset for plotting)
        Y_full = data_df[gamma_col]
        
        #filter data for fitting if x_range is specified
        if x_range is not None:
            if not isinstance(x_range, (tuple, list)) or len(x_range) != 2:
                raise ValueError("x_range must be a tuple or list of length 2: (min_x, max_x)")

            min_x, max_x = x_range
            mask = (X_full >= min_x) & (X_full <= max_x)
            # Reset index to avoid alignment issues with boolean indexing
            X_fit = X_full[mask].reset_index(drop=True)
            Y_fit = Y_full[mask].reset_index(drop=True)
            
            if len(X_fit) < 2:
                print(f"Warning: Only {len(X_fit)} data points in specified x_range for {gamma_col}. Need at least 2 points for fitting.")
                continue
                
            fit_range_text = f" (fit range: {min_x:.3f} to {max_x:.3f})"
        else:
            X_fit = X_full
            Y_fit = Y_full
            fit_range_text = ""
        
        #plot all data points
        plt.figure(figsize=(8, 4))
        plt.scatter(X_full, Y_full, alpha=0.6, label='All data')
        
        # Highlight fitting data if x_range is specified
        if x_range is not None:
            plt.scatter(X_fit, Y_fit, color='red', alpha=0.8, label='Fitting data', s=30)
        
        #perform linear regression on fitting data
        X_fit_with_constant = sm.add_constant(X_fit)
        model = sm.OLS(Y_fit, X_fit_with_constant).fit()
        
        #plot regression line over the full x-range for visualization
        X_line = np.linspace(X_full.min(), X_full.max(), 100)
        X_line_with_constant = sm.add_constant(X_line)
        y_predicted_line = model.predict(X_line_with_constant)
        plt.plot(X_line, y_predicted_line, 'r-', label=f'Regression Line{fit_range_text}')
        
        #print regression results
        print(f"\n--- Linear Regression Results {method_name}{fit_range_text} ---")
        print(f"Number of data points used for fitting: {len(X_fit)}")
        print(model.summary())
        print("\n")
        
        #extract statistical results
        r_squared = model.rsquared
        jb_p_value = float(model.summary().tables[2].data[2][3])
        residuals = model.resid
        skewness = stats.skew(residuals)
        kurtosis = stats.kurtosis(residuals)
        omnibus_p_value = float(model.summary().tables[2].data[1][1])
        
        #check normality conditions
        conditions_met = [
            jb_p_value >= 0.05, 
            omnibus_p_value >= 0.05, 
            skewness <= 1, 
            kurtosis <= 0
        ]
        conditions_not_met = [
            jb_p_value < 0.05, 
            omnibus_p_value < 0.05, 
            skewness > 1, 
            kurtosis > 0
        ]
        
        if all(conditions_met):
            normality_status = "Normally distributed"
        elif all(conditions_not_met):
            normality_status = "Not normally distributed"
        else:
            normality_status = "Possibly not normally distributed"
        
        #store regression results
        results_dict = {
            'Column': gamma_col,
            'Method': method_name,
            'R_squared': r_squared, 
            'Normality': normality_status,
            'Skewness': skewness,
            'Kurtosis': kurtosis,
            'JB_p_value': jb_p_value,
            'Omnibus_p_value': omnibus_p_value,
            'N_points_fit': len(X_fit),
            'X_range_min': X_fit.min() if len(X_fit) > 0 else None,
            'X_range_max': X_fit.max() if len(X_fit) > 0 else None
        }
        
        #add coefficients and standard errors
        for param, coef, se in zip(model.params.index, model.params, model.bse):
            results_dict[f'{param}_coef'] = coef
            results_dict[f'{param}_se'] = se
        
        all_results.append(results_dict)
        
        #set labels and title
        plt.xlabel(r'q$^2$ [nm$^{-2}$]')
        plt.ylabel(fr'$\Gamma$ [{gamma_unit}]')

        title = r'q$^2$ vs. $\Gamma$'
        if method_name:
            title += f' ({method_name})'
        plt.title(title)
        
        plt.legend()
        plt.grid(True)
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    #create dataframe with results
    if not all_results:
        print("No valid columns were analyzed.")
        return pd.DataFrame()
    
    return pd.DataFrame(all_results)

#cleaning the dataframe from bad-fits
def remove_rows_by_index(df, indices_str):
    try:
        if indices_str:  # Check if the string is not empty
            indices = [int(index.strip()) for index in indices_str.split(',')]
            valid_indices = [index for index in indices if index in df.index]

            if valid_indices:
                df = df.drop(valid_indices)
                print("Rows removed.")
            else:
                print("No valid indices found in the DataFrame.")
        else:
            print("No indices provided.")
    except ValueError:
        print("Invalid input. Please provide comma-separated integers.")
    return df

#plot and fit for method B
def plot_processed_correlations(dataframes_dict, fit_function, fit_x_limits):
    all_fit_results = []
    plot_number = 1
    
    for name, df in dataframes_dict.items():
        try:
            fit_result = {'filename': name}

            #main processing
            x_data = df['t (s)']
            y_data = df['g(2)_mod']
            x_fit = x_data[(x_data >= fit_x_limits[0]) & (x_data <= fit_x_limits[1])]
            y_fit = y_data[(x_data >= fit_x_limits[0]) & (x_data <= fit_x_limits[1])]
                
            #perform the fit with bounds
            popt, pcov = curve_fit(fit_function, x_fit, y_fit, method='lm', maxfev=50000)
                
            #calculate parameter errors from covariance matrix
            perr = np.sqrt(np.diag(pcov))
                
            #generate fit curve
            y_fit_values = fit_function(x_data, *popt)
                
            #calculate residuals for Q-Q plot
            residuals = y_fit - fit_function(x_fit, *popt)
                
            #extract parameter names and store fit parameters
            param_names = inspect.getfullargspec(fit_function).args[1:]
            for i, param_name in enumerate(param_names):
                fit_result[param_name] = popt[i]
                fit_result[f'{param_name}_error'] = perr[i]
                fit_result[f'{param_name}_relative_error'] = (perr[i] / abs(popt[i])) * 100 if popt[i] != 0 else np.inf
                
            # Calculate fit quality metrics
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y_fit - np.mean(y_fit))**2)
            r_squared = 1 - (ss_res / ss_tot)
            fit_result['R-squared'] = r_squared
                
            #create subplot layout
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                
            #left plot: original data and fit
            ax1.plot(x_data, y_data, marker='.', linestyle='', label='Data')
            ax1.plot(x_data, y_fit_values, 'r-', label=f'Fit')
            ax1.set_xlabel('lag time (s)')
            ax1.set_ylabel(r"$\sqrt{g(2)-1}$")
            ax1.set_title(f'[{plot_number}]: g(2)-1 vs. lag time for {name}')
            ax1.grid(True)
            ax1.set_yscale('log')
            ax1.set_xlim(0, 0.002)
            ax1.legend()
                
            #right plot: Q-Q plot
            stats.probplot(residuals, dist="norm", plot=ax2)
            ax2.set_title(f'[{plot_number}]: Q-Q Plot of Residuals')
            ax2.grid(True)
                
            #add R^2 as text in the left plot
            param_text = f"R² = {r_squared:.4f}"
                
            #position the text in the upper right of the left plot
            ax1.text(0.95, 0.95, param_text, transform=ax1.transAxes, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                
            plt.tight_layout()
            plt.show()
            plot_number += 1
                
            #store all results
            all_fit_results.append(fit_result)
                
        except (KeyError, TypeError) as e:
            print(f"Error processing DataFrame '{name}': {e}")
            fit_result['Error'] = str(e)
            all_fit_results.append(fit_result)
        except RuntimeError as e:
            print(f"Fit error for DataFrame '{name}': {e}")
            fit_result['Error'] = str(e)
            all_fit_results.append(fit_result)
    
    final_results_df = pd.DataFrame(all_fit_results)
    return final_results_df


