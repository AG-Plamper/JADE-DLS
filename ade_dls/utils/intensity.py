# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 12:08:00 2025

@author: vinci

intensity processing
"""
import pandas as pd
import matplotlib.pyplot as plt

#extraction of data required for intensity calculations
def extract_intensity(file_path):
  encodings = ['utf-8', 'latin-1', 'windows-1252', 'ISO-8859-1']
  
  for encoding in encodings:
    try:
      with open(file_path, 'r', encoding=encoding) as file:
        lines = file.readlines()
        
        #extract angle
        angle = None
        for line in lines:
          if "Angle [°]       :" in line: 
            try:
              angle = float(line.split(":")[1].strip()) 
              break 
            except (ValueError, IndexError):
              print(f"Error extracting angle from {file_path}")

        #extract MeanCR0
        meancr0 = None
        for line in lines:
          if "MeanCR0 [kHz]   :" in line: 
            try:
              meancr0 = float(line.split(":")[1].strip()) 
              break 
            except (ValueError, IndexError):
              print(f"Error extracting MeanCR0 from {file_path}")
              
        #extract MeanCR1
        meancr1 = None
        for line in lines:
          if "MeanCR1 [kHz]   :" in line: 
            try:
              meancr1 = float(line.split(":")[1].strip()) 
              break 
            except (ValueError, IndexError):
              print(f"Error extracting MeanCR2 from {file_path}")
              
        #extract monitor diode
        monitordiode = None
        for line in lines:
          if "Monitor Diode" in line:
            try:
             monitordiode = float(line.split()[-1]) 
             break  
            except (ValueError, IndexError):
              print(f"Error extracting monitor diode from {file_path}")    
              
        #create DataFrame
        data = {'angle [°]': [angle], 'meancr0 [kHz]': [meancr0], 'meancr1 [kHz]': [meancr1], 
                'monitordiode [cps]': [monitordiode]}
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

#plotting mean intensities for each angle
def plot_meancr(df, x_col, y_col):
    plt.figure(figsize=(8, 4))
    plt.scatter(df[x_col], df[y_col], alpha=0.7, s=50)
    plt.xlabel('angle [°]')
    plt.ylabel('corrected mean intensity [kHz]')
    plt.title('')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    