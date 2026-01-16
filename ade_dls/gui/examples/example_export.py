"""
Example: Exported Analysis from JADE-DLS GUI

This shows what an exported Python script would look like.
It demonstrates the transparency feature of the GUI.
"""

# JADE-DLS Analysis Script
# Auto-generated on 2025-11-09 20:00:00
# Generated from GUI session

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ade_dls.core.preprocessing import extract_data, extract_correlation
from ade_dls.analysis.cumulants import analyze_diffusion_coefficient
from ade_dls.analysis.regularized import nnls_reg_all
import glob
import os


# ========== Step 1: Load Data ==========
# Timestamp: 2025-11-09 19:55:00

data_folder = r"C:/path/to/data"
datafiles = glob.glob(os.path.join(data_folder, "*.asc"))
filtered_files = [f for f in datafiles
                  if "averaged" not in os.path.basename(f).lower()]

print(f"Loaded {len(filtered_files)} data files")


# ========== Step 2: Extract Base Data ==========
# Timestamp: 2025-11-09 19:55:10

all_data = []
for file in filtered_files:
    extracted_data = extract_data(file)
    if extracted_data is not None:
        filename = os.path.basename(file)
        extracted_data['filename'] = filename
        all_data.append(extracted_data)

if all_data:
    df_basedata = pd.concat(all_data, ignore_index=True)
    df_basedata.index = df_basedata.index + 1
else:
    print("No data extracted!")

# Calculate q and q^2
df_basedata['q'] = abs(
    ((4*np.pi*df_basedata['refractive_index']) /
     (df_basedata['wavelength [nm]'])) *
    np.sin(np.radians(df_basedata['angle [°]'])/2)
)
df_basedata['q^2'] = df_basedata['q']**2


# ========== Step 3: Extract Correlation Data ==========
# Timestamp: 2025-11-09 19:55:30

all_correlations = {}
file_to_path = {os.path.basename(file): file for file in filtered_files}

for filename in df_basedata['filename']:
    if filename in file_to_path:
        file_path = file_to_path[filename]
        extracted_correlation = extract_correlation(file_path)
        if extracted_correlation is not None:
            all_correlations[filename] = extracted_correlation

print(f"Extracted correlation data for {len(all_correlations)} files")


# ========== Step 4: Cumulant Analysis C ==========
# Timestamp: 2025-11-09 19:56:00
# Method: Iterative non-linear fit (4th order)
# Parameters:
#   fit_range: (1e-9, 10)
#   optimization: lm
#   adaptive: True
#   strategy: individual

from ade_dls.analysis.cumulants_C import plot_processed_correlations_iterative

# Process correlations
from ade_dls.core.preprocessing import process_correlation_data
columns_to_drop = ['time [ms]', 'correlation 1', 'correlation 2',
                   'correlation 3', 'correlation 4']
processed_correlations = process_correlation_data(all_correlations, columns_to_drop)

# Define fit function (4th order cumulant)
def fit_function4(x, a, b, c, d, e, f):
    inner_term = 1 + 0.5 * c * x**2 - (d * x**3) / 6 + ((e - 3 * c**2) * x**4) / 24
    term = f + a * (np.exp(-b * x) * inner_term)**2
    return term

# Initial parameters
base_initial_parameters = [0.8, 10000, 0, 0, 0, 0]

# Fit
fit_limits = (1e-9, 10)
cumulant_C_fit = plot_processed_correlations_iterative(
    processed_correlations,
    fit_function4,
    fit_limits,
    base_initial_parameters,
    method='lm'
)


# ========== Step 5: Calculate Diffusion Coefficient ==========
# Timestamp: 2025-11-09 19:57:00

cumulant_C_data = pd.merge(df_basedata, cumulant_C_fit, on='filename', how='outer')

cumulant_C_diff = analyze_diffusion_coefficient(
    data_df=cumulant_C_data,
    q_squared_col='q^2',
    gamma_cols=['best_b'],
    method_names=['Cumulant C']
)


# ========== Step 6: Calculate Hydrodynamic Radius ==========
# Timestamp: 2025-11-09 19:57:30

from scipy.constants import k  # Boltzmann constant

# Calculate mean temperature and viscosity
mean_temperature = df_basedata['temperature [K]'].mean()
mean_viscosity = df_basedata['viscosity [cp]'].mean()

# Calculate constant c = k_B*T / (6*pi*eta)
c = (k * mean_temperature) / (6 * np.pi * mean_viscosity * 1e-3)

# Calculate diffusion coefficient in m^2/s
D = cumulant_C_diff['q^2_coef'].iloc[0] * 1e-18
D_error = cumulant_C_diff['q^2_se'].iloc[0] * 1e-18

# Calculate hydrodynamic radius
Rh = c / D * 1e9  # in nm
Rh_error = Rh * (D_error / D)  # Propagated error

# Calculate polydispersity index
cumulant_C_data['polydispersity'] = (
    cumulant_C_data['best_c'] / (cumulant_C_data['best_b']**2)
)
PDI = cumulant_C_data['polydispersity'].mean()


# ========== Results ==========

print("\n" + "="*60)
print("JADE-DLS Analysis Results - Cumulant Method C")
print("="*60)
print(f"Hydrodynamic Radius: {Rh:.2f} ± {Rh_error:.2f} nm")
print(f"Diffusion Coefficient: {D:.2e} ± {D_error:.2e} m²/s")
print(f"Polydispersity Index: {PDI:.4f}")
print(f"R-squared: {cumulant_C_diff['R_squared'].iloc[0]:.6f}")
print("="*60)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(cumulant_C_data['q^2'], cumulant_C_data['best_b'], 'o',
         label='Data', markersize=8)
plt.xlabel('q² [nm⁻²]')
plt.ylabel('Γ [s⁻¹]')
plt.title('Cumulant Method C: Γ vs q²')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('cumulant_C_results.png', dpi=300)
plt.show()

print("\nAnalysis complete! Results saved to cumulant_C_results.png")
