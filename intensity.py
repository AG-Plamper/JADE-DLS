# -*- coding: utf-8 -*-
"""
intensity.py
============
Extraction and visualization of angle-resolved scattering intensity data
from ALV .ASC files for use in static light scattering (SLS) analysis.

Functions
---------
extract_intensity(file_path)
    Extract angle, MeanCR0, MeanCR1 and monitor diode reading from an
    ALV .ASC file with robust encoding fallback. Returns a single-row
    DataFrame per file.

plot_meancr(df, x_col, y_col, title, figsize, save_path)
    Scatter plot of corrected mean intensity vs angle for a compiled
    multi-angle intensity DataFrame.

Dependencies: pandas, matplotlib
"""
import pandas as pd
import matplotlib.pyplot as plt

#extract intensity-related data from ALV DLS file.
def extract_intensity(file_path):
    encodings = ['Windows-1252', 'utf-8', 'latin-1', 'ISO-8859-1']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                lines = file.readlines()
                
                #extract angle
                angle = None
                for line in lines:
                    if "Angle [" in line and "]" in line and ":" in line:
                        try:
                            angle = float(line.split(":")[1].strip())
                            break
                        except (ValueError, IndexError):
                            print(f"Error extracting angle from {file_path}")

                #extract MeanCR0
                meancr0 = None
                for line in lines:
                    if "MeanCR0 [kHz]" in line:
                        try:
                            meancr0 = float(line.split(":")[1].strip())
                            break
                        except (ValueError, IndexError):
                            print(f"Error extracting MeanCR0 from {file_path}")
                              
                #extract MeanCR1
                meancr1 = None
                for line in lines:
                    if "MeanCR1 [kHz]" in line:
                        try:
                            meancr1 = float(line.split(":")[1].strip())
                            break
                        except (ValueError, IndexError):
                            print(f"Error extracting MeanCR1 from {file_path}")
                              
                #extract monitor diode
                monitordiode = None
                for line in lines:
                    if "Monitor Diode" in line:
                        try:
                            # Split by whitespace and get the last element
                            monitordiode = float(line.split()[-1])
                            break
                        except (ValueError, IndexError):
                            print(f"Error extracting monitor diode from {file_path}")    
                              
                #create DataFrame
                data = {
                    'angle [°]': [angle], 
                    'meancr0 [kHz]': [meancr0], 
                    'meancr1 [kHz]': [meancr1], 
                    'monitordiode [cps]': [monitordiode]
                }
                return pd.DataFrame(data)
              
        except UnicodeDecodeError:
            if encoding == encodings[-1]:
                print(f"Failed to decode {file_path} with all attempted encodings")
                return None
            continue
        except FileNotFoundError as e:
            print(f"File not found: {file_path}. Error: {e}")
            return None
        except Exception as e:
            print(f"An error occurred while processing {file_path}: {e}")
            return None

#plot mean intensities vs angle
def plot_meancr(df, x_col, y_col, title='', figsize=(8, 4), save_path=None):
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(df[x_col], df[y_col], alpha=0.7, s=50)
    ax.set_xlabel('angle [°]')
    ax.set_ylabel('corrected mean intensity [kHz]')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return fig, ax
