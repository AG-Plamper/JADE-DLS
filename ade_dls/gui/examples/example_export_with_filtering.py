# JADE-DLS Analysis Script
# Auto-generated on 2025-11-10 14:30:00
# Generated from GUI session
#
# This example demonstrates how filtering steps are tracked and exported
# for full reproducibility of your analysis

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ade_dls.core.preprocessing import extract_data, extract_correlation
from ade_dls.analysis.cumulants import analyze_diffusion_coefficient
from ade_dls.analysis.regularized import nnls_reg_all
import glob
import os

# ========== Step 1: Load Data ==========

# Step: Load Data
# Timestamp: 2025-11-10 14:25:00

# Load data from folder
data_folder = r"/path/to/your/data"
datafiles = glob.glob(os.path.join(data_folder, "*.asc"))

# Filter out averaged files
filtered_files = [f for f in datafiles
                  if "averaged" not in os.path.basename(f).lower()]

print(f"Found {len(filtered_files)} .asc files in {data_folder}")

# Initialize data dictionaries
countrates_data = {}
correlations_data = {}
df_basedata = pd.DataFrame()


# ========== Step 2: Extract Data ==========

# Step: Extract Data
# Timestamp: 2025-11-10 14:25:15

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

# ========== Step 3: Filter Countrates ==========

# Step: Filter Countrates
# Timestamp: 2025-11-10 14:26:30

# Filtering step: Exclude 2 files based on countrates
# Original files: 15, Remaining: 13
files_to_exclude_countrates = [
    'sample_001.asc',
    'sample_007.asc'
]

# Filter countrates data
countrates_data = {k: v for k, v in countrates_data.items()
                if k not in files_to_exclude_countrates}

print(f"Filtered countrates: {len(files_to_exclude_countrates)} files excluded, {len(countrates_data)} files remaining")

# Update basedata to match (if it exists)
if 'df_basedata' in locals() and len(df_basedata) > 0:
    df_basedata = df_basedata[~df_basedata['filename'].isin(files_to_exclude_countrates)]
    df_basedata = df_basedata.reset_index(drop=True)
    df_basedata.index = df_basedata.index + 1


# ========== Step 4: Filter Correlations ==========

# Step: Filter Correlations
# Timestamp: 2025-11-10 14:28:15

# Filtering step: Exclude 1 files based on correlations
# Original files: 13, Remaining: 12
files_to_exclude_correlations = [
    'sample_010.asc'
]

# Filter correlations data
correlations_data = {k: v for k, v in correlations_data.items()
                if k not in files_to_exclude_correlations}

print(f"Filtered correlations: {len(files_to_exclude_correlations)} files excluded, {len(correlations_data)} files remaining")

# Update basedata to match (if it exists)
if 'df_basedata' in locals() and len(df_basedata) > 0:
    df_basedata = df_basedata[~df_basedata['filename'].isin(files_to_exclude_correlations)]
    df_basedata = df_basedata.reset_index(drop=True)
    df_basedata.index = df_basedata.index + 1


# ========== Step 5: Cumulant Analysis ==========

# Step: Cumulant Analysis
# Timestamp: 2025-11-10 14:29:00
# ... (analysis code would follow)

# ========== Summary ==========
print("\n" + "="*60)
print("ANALYSIS COMPLETE - FILTERING SUMMARY")
print("="*60)
print(f"Original files found: 15")
print(f"Files excluded from countrates: {len(files_to_exclude_countrates)}")
print(f"Files excluded from correlations: {len(files_to_exclude_correlations)}")
print(f"Final files analyzed: 12")
print("\nExcluded files:")
print("  Countrates:", files_to_exclude_countrates)
print("  Correlations:", files_to_exclude_correlations)
print("="*60)

# This script is fully reproducible - running it independently will give
# the same results as the GUI analysis, with the same files excluded.
