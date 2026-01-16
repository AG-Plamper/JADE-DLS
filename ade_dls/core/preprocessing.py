# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 09:24:53 2025

@author: vinci

preprocessing
"""
import pandas as pd
import os
from io import StringIO
import matplotlib.pyplot as plt

#get folder name
def get_folder_name(filepath):
  try:
    normalized_path = os.path.normpath(filepath)
    parts = normalized_path.split(os.sep)
    if len(parts) <= 1 or (len(parts) == 2 and parts[0] == '' and parts[1] == ''):
        return None

    return parts[-2]


  except Exception as e:
    print(f"An error occurred: {e}")
    return None

#function for base-data extraction
def extract_data(file_path):
    """
    Extract base data from .asc file with robust error handling

    Args:
        file_path: Path to .asc file

    Returns:
        pd.DataFrame with extracted data or None on error
    """
    # Handling of encoding issues
    encodings = ['utf-8', 'latin-1', 'windows-1252', 'ISO-8859-1']

    for encoding_idx, encoding in enumerate(encodings):
        try:
            # Initialize data dictionary
            data = {
                'angle [°]': [None],
                'temperature [K]': [None],
                'wavelength [nm]': [None],
                'refractive_index': [None],
                'viscosity [cp]': [None]
            }

            # Read file line by line (memory efficient)
            # Only read first 200 lines (metadata is always at the top)
            with open(file_path, 'r', encoding=encoding, errors='replace') as file:
                for line_num, line in enumerate(file):
                    # Stop after 200 lines (all metadata should be found by then)
                    if line_num > 200:
                        break

                    try:
                        # Extract angle
                        if data['angle [°]'][0] is None and "Angle [°]       :" in line:
                            try:
                                data['angle [°]'][0] = float(line.split(":")[1].strip())
                            except (ValueError, IndexError):
                                pass  # Continue to next field

                        # Extract temperature
                        elif data['temperature [K]'][0] is None and "Temperature [K] :" in line:
                            try:
                                data['temperature [K]'][0] = float(line.split(":")[1].strip())
                            except (ValueError, IndexError):
                                pass

                        # Extract wavelength
                        elif data['wavelength [nm]'][0] is None and "Wavelength [nm] :" in line:
                            try:
                                data['wavelength [nm]'][0] = float(line.split(":")[1].strip())
                            except (ValueError, IndexError):
                                pass

                        # Extract refractive index
                        elif data['refractive_index'][0] is None and "Refractive Index:" in line:
                            try:
                                data['refractive_index'][0] = float(line.split(":")[1].strip())
                            except (ValueError, IndexError):
                                pass

                        # Extract viscosity
                        elif data['viscosity [cp]'][0] is None and "Viscosity [cp]  :" in line:
                            try:
                                data['viscosity [cp]'][0] = float(line.split(":")[1].strip())
                            except (ValueError, IndexError):
                                pass

                        # Early exit if all data found
                        if all(v[0] is not None for v in data.values()):
                            break

                    except Exception as line_error:
                        # Skip problematic lines
                        continue

            # Verify we got ALL required data (not just some)
            # All fields must be present for valid DLS analysis
            if any(v[0] is None for v in data.values()):
                missing_fields = [k for k, v in data.items() if v[0] is None]
                if encoding_idx == len(encodings) - 1:
                    print(f"WARNING: Incomplete metadata in {os.path.basename(file_path)}")
                    print(f"         Missing fields: {', '.join(missing_fields)}")
                    return None
                else:
                    continue  # Try next encoding

            # Successfully extracted complete data, return DataFrame
            return pd.DataFrame(data)

        except UnicodeDecodeError:
            # This encoding doesn't work, try the next one
            if encoding_idx == len(encodings) - 1:
                print(f"ERROR: Failed to decode {os.path.basename(file_path)} with all encodings")
                return None
            continue

        except MemoryError:
            print(f"ERROR: Out of memory while processing {os.path.basename(file_path)}")
            return None

        except FileNotFoundError:
            print(f"ERROR: File not found: {os.path.basename(file_path)}")
            return None

        except PermissionError:
            print(f"ERROR: Permission denied: {os.path.basename(file_path)}")
            return None

        except Exception as e:
            print(f"ERROR: Unexpected error processing {os.path.basename(file_path)}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return None

    # Should never reach here
    return None


#function to extract countrate
#first finding the countrate in file
def find_countrate_row(filename, encoding='Windows-1252'):
    try:
        with open(filename, 'r', encoding=encoding) as f:
            for i, line in enumerate(f):
                if '"Count Rate"' in line:
                    return i
        return None 
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None
    except UnicodeDecodeError:
        print(f"Error: Encoding error with '{filename}'. Try a different encoding.")
        return None

#extract from file
def extract_countrate(filename, encoding='Windows-1252'):
    row_num_count_rate = find_countrate_row(filename, encoding)

    if row_num_count_rate is not None:
        try:
            data = []
            with open(filename, 'r', encoding=encoding) as f:
                for i, line in enumerate(f):
                    if i > row_num_count_rate:
                        if "Monitor Diode" in line:
                            break
                        data.append(line.strip())

            if data:
                df = pd.read_csv(StringIO("\n".join(data)), sep="\t", header=None)
                return df
            else:
                print(f"No data between 'Count Rate' and 'Monitor Diode' in {filename}")
                return None

        except (pd.errors.EmptyDataError, ValueError):
            print(f"Error creating DataFrame for {filename}.")
            return None
        except UnicodeDecodeError:
            print(f"Encoding error with '{filename}'.")
            return None
    else:
        print(f"'Count Rate' not found in {filename}")
        return None


#plot all countrates over time with indices in titles and enable CLI exclusion selection
def plot_countrates(all_countrates, ncols=3, show_indices=True): 
    n_plots = len(all_countrates)
    nrows = (n_plots + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 5*nrows))

    if isinstance(axes, plt.Axes): 
        axes = [axes]
    else:
        axes = axes.flatten()

    #create list of dataset names for index reference
    dataset_names = list(all_countrates.keys())
    
    i = 0
    for name, df in all_countrates.items():
        if i < len(axes): 
            ax = axes[i]

            for column in df.columns[1:]:
                ax.plot(df[df.columns[0]], df[column], label=column, linewidth=1) 

            ax.set_xlabel(str(df.columns[0]))
            ax.set_ylabel("Countrate [kHz]")
            
            #add index to title if requested
            if show_indices:
                idx = dataset_names.index(name)
                ax.set_title(f"[{idx}] {name}", fontsize=10)
            else:
                ax.set_title(name, fontsize=10)
                
            ax.legend(fontsize=8)  
            ax.grid(True)
            ax.tick_params(axis='both', labelsize=8) 
            ax.xaxis.get_major_locator().set_params(nbins=3) 
            ax.yaxis.get_major_locator().set_params(nbins=3) 
            i += 1
        else:
            break 
    for j in range(i, len(axes)):
        axes[j].set_axis_off()

    plt.tight_layout(pad=0.5) 
    plt.show()

def cli_countrate_exclusion(all_countrates):
    #show all plots with indices in titles
    plot_countrates(all_countrates, show_indices=True)
    
    #get user input for exclusion
    selection = input("\nEnter indices of datasets to EXCLUDE (comma separated, or 'none'): ")
    
    #list of dataset names for reference
    dataset_names = list(all_countrates.keys())
    
    if selection.lower() in ['none', '', 'n']:
        print(f"No datasets excluded. Using all {len(all_countrates)} datasets")
        return all_countrates
    else:
        try:
            #parse exclusion indices
            excluded_indices = [int(idx.strip()) for idx in selection.split(',')]
            excluded_names = [dataset_names[i] for i in excluded_indices if 0 <= i < len(dataset_names)]
            
            #create filtered dataset excluding selected ones
            filtered_data = {name: data for name, data in all_countrates.items() 
                           if name not in excluded_names}
            
            #report results
            if excluded_names:
                print(f"Excluded {len(excluded_names)} datasets: {', '.join(excluded_names)}")
                print(f"Continuing with {len(filtered_data)} datasets")
            else:
                print("No valid exclusions. Using all datasets.")
            
            #option to visualize the filtered selection
            if filtered_data and len(filtered_data) != len(all_countrates):
                show_filtered = input("Show filtered datasets? (y/n): ").lower()
                if show_filtered == 'y' or show_filtered == 'yes':
                    plot_countrates(filtered_data, show_indices=True)
            
            return filtered_data
        except (ValueError, IndexError) as e:
            print(f"Error in selection: {e}. Using all datasets.")
            return all_countrates


#function to extract correlation-data
#first find correlation data
def find_correlation_row(filename, encoding='Windows-1252'):
    try:
        with open(filename, 'r', encoding=encoding) as f:
            for i, line in enumerate(f):
                if '"Correlation"' in line:
                    return i
        return None 
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None
    except UnicodeDecodeError:
        print(f"Error: Encoding error with '{filename}'. Try a different encoding.")
        return None

#extract correlation data
def extract_correlation(filename, encoding='Windows-1252'):
    row_num_correlation = find_correlation_row(filename, encoding)

    if row_num_correlation is not None:
        try:
            data = []
            with open(filename, 'r', encoding=encoding) as f:
                for i, line in enumerate(f):
                    if i > row_num_correlation:
                        if '"Count Rate"' in line:
                            break
                        data.append(line.strip())

            if data:
                df = pd.read_csv(StringIO("\n".join(data)), sep="\t", header=None)
                return df
            else:
                print(f"No data between 'Correlation' and 'Count Rate' in {filename}")
                return None

        except (pd.errors.EmptyDataError, ValueError):
            print(f"Error creating DataFrame for {filename}.")
            return None
        except UnicodeDecodeError:
            print(f"Encoding error with '{filename}'.")
            return None
    else:
        print(f"'Correlation' not found in {filename}")
        return None
    
#plot all correlation columns except the ones that are 0 and enable CLI exclusion selection similar as before
def plot_correlations(all_correlations, ncols=3, show_indices=True):
    n_plots = len(all_correlations)
    nrows = (n_plots + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 5*nrows))

    if isinstance(axes, plt.Axes):
        axes = [axes]
    else:
        axes = axes.flatten()

    dataset_names = list(all_correlations.keys())
    
    i = 0
    for name, df in all_correlations.items():
        if i < len(axes):
            ax = axes[i]

            for column in df.columns[1:]: 
                if not (df[column] == 0).all(): 
                    ax.plot(df[df.columns[0]], df[column], label=column, linewidth=1)

            ax.set_xlabel(str(df.columns[0]))
            ax.set_ylabel("Correlation")
            
            # Add index to title if requested
            if show_indices:
                idx = dataset_names.index(name)
                ax.set_title(f"[{idx}] {name}", fontsize=10)
            else:
                ax.set_title(name, fontsize=10)
                
            ax.set_xscale('log')
            ax.legend(fontsize=8)
            ax.grid(True)
            ax.tick_params(axis='both', labelsize=8)
            ax.xaxis.set_major_locator(plt.LogLocator(base=10, numticks=4))
            ax.yaxis.get_major_locator().set_params(nbins=5)
            i += 1
        else:
            break

    for j in range(i, len(axes)):
        axes[j].set_axis_off()

    plt.tight_layout(pad=0.5)
    plt.show()

#also similar to as before
def cli_correlation_exclusion(all_correlations):
    plot_correlations(all_correlations, show_indices=True)
    
    selection = input("\nEnter indices of correlation datasets to EXCLUDE (comma separated, or 'none'): ")
    
    dataset_names = list(all_correlations.keys())
    
    if selection.lower() in ['none', '', 'n']:
        print(f"No correlation datasets excluded. Using all {len(all_correlations)} datasets")
        return all_correlations
    else:
        try:
            excluded_indices = [int(idx.strip()) for idx in selection.split(',')]
            excluded_names = [dataset_names[i] for i in excluded_indices if 0 <= i < len(dataset_names)]
            
            filtered_data = {name: data for name, data in all_correlations.items() 
                           if name not in excluded_names}
            
            if excluded_names:
                print(f"Excluded {len(excluded_names)} correlation datasets: {', '.join(excluded_names)}")
                print(f"Continuing with {len(filtered_data)} correlation datasets")
            else:
                print("No valid exclusions. Using all correlation datasets.")
            
            if filtered_data and len(filtered_data) != len(all_correlations):
                show_filtered = input("Show filtered correlation datasets? (y/n): ").lower()
                if show_filtered == 'y' or show_filtered == 'yes':
                    plot_correlations(filtered_data, show_indices=True)
            
            return filtered_data
        except (ValueError, IndexError) as e:
            print(f"Error in selection: {e}. Using all correlation datasets.")
            return all_correlations

#data removal
def remove_from_data(dataframe, to_remove):
    if 'filename' not in dataframe.columns:
        raise ValueError("data must have a 'filename' column.")

    df_filtered = dataframe[~dataframe['filename'].isin(to_remove)].copy()
    return df_filtered

#dataframe removal
def remove_dataframes(dataframes_dict, to_remove):
    new_dict = dataframes_dict.copy()  
    for frame_name in to_remove:
        if frame_name in new_dict:
            del new_dict[frame_name]
        else:
                print(f"Warning: DataFrame '{frame_name}' not found in the dictionary.")
    return new_dict


#process correlation data for easier evaluation
def process_correlation_data(input_dict, columns_to_remove = None):
    if not isinstance(input_dict, dict):
        print("Error: Input must be a dictionary.")
        return {}

    output_dict = {}

    for key, df in input_dict.items():
        if not isinstance(df, pd.DataFrame):
            print(f"Error: '{key}' is not a Pandas DataFrame.")
            return {}
        try:
            #create a copy to avoid modifying the original DataFrame
            new_df = df.copy()  
            #time in s
            new_df['t (s)'] = df['time [ms]']*10**(-3)
            #calculates the mean of the two correlation detectors
            new_df['g(2)'] = (df['correlation 1']+df['correlation 2'])/2
            
            #remove columns
            if columns_to_remove:
                for col in columns_to_remove:
                    if col in new_df.columns:  #check if the column exists before removing
                        new_df = new_df.drop(columns=col)
                    else:
                        print(f"Warning: Column '{col}' not found in DataFrame '{key}'.")
                        
            output_dict[key] = new_df

        except (KeyError, TypeError, AttributeError) as e:  #catch potential errors
            print(f"Error processing DataFrame for key '{key}': {e}")
            continue

    return output_dict
