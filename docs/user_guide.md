# JADE-DLS GUI

**J**upyter-based **A**ngular **D**ependent **E**valuator for **D**ynamic **L**ight **S**cattering - GUI Edition

A transparent, user-friendly graphical interface for Dynamic Light Scattering (DLS) data analysis.

## Features

### ✨ Core Capabilities

- **Multi-Method Analysis**: Cumulant (A/B/C), NNLS, and Regularized inverse Laplace fitting
- **Interactive Visualization**: Real-time plotting of correlation data, fits, and results
- **Transparent Pipeline**: See exactly what code is being executed
- **Reproducible Research**: Export analysis as Jupyter notebooks or Python scripts
- **Parameter Tracking**: Full history of all analysis parameters

### 🔍 Transparency Features

Unlike "black box" software, JADE-DLS GUI shows you:
- **Generated Code**: View the exact Python code for every analysis step
- **Parameter Inspector**: See all parameters and their values
- **Built-in Documentation**: Understand what each method does
- **Export Workflow**: Generate reproducible Jupyter notebooks or scripts

## Installation

### Prerequisites

- Python 3.7 or higher
- Existing JADE-DLS analysis modules (cumulants.py, preprocessing.py, etc.)

### Install Dependencies

```bash
pip install -r requirements_gui.txt
```

Or install individually:

```bash
pip install PyQt5 numpy pandas scipy matplotlib nbformat
```

## Usage

### Quick Start

1. **Launch the GUI:**

```bash
python jade_dls_gui.py
```

2. **Load Data:**
   - Click `File > Load Data...` or use `Ctrl+O`
   - Select folder containing `.asc` files from ALV correlator

3. **Run Analysis:**
   - Select analysis method from the workflow panel
   - Click "▶ Run Selected" or "▶ Run All"
   - View results in the central panel

4. **Export Results:**
   - `File > Export as Jupyter Notebook...` (Ctrl+J)
   - `File > Export as Python Script...` (Ctrl+P)

### Workflow Panel (Left)

The workflow panel shows available analysis steps:

- 📂 **Load Data** - Load .asc data files
- 🔍 **Preprocess** - Extract and filter data
- 📊 **Cumulant A** - ALV software cumulant data
- 📈 **Cumulant B** - Linear cumulant fit
- 📉 **Cumulant C** - Iterative non-linear fit
- 🔬 **NNLS** - Inverse Laplace NNLS
- ⚙️ **Regularized** - Tikhonov-Phillips regularization
- 📋 **Compare** - Compare all methods

### Analysis View (Center)

Tabbed interface showing:

- **Data Overview** - Loaded files and statistics
- **Plots** - Interactive visualization
- **Results** - Analysis results table
- **Comparison** - Side-by-side method comparison

### Inspector Panel (Right)

Transparency features:

- **💻 Code** - Auto-generated Python code
- **⚙️ Parameters** - All analysis parameters
- **📖 Docs** - Method documentation

## Analysis Methods

### Cumulant Analysis

**Method A:** Uses pre-calculated cumulant fits from ALV software
- Fast, reliable for monodisperse samples
- Provides 1st, 2nd, and 3rd order fits

**Method B:** Linear fit of ln[g(τ)^0.5] vs τ
- Simple, transparent approach
- Best for quick analysis

**Method C:** Iterative non-linear least squares
- Most accurate for monodisperse samples
- Adaptive parameter estimation

### Inverse Laplace Methods

**NNLS:** Non-Negative Least Squares
- Provides size distributions
- Good starting point for polydisperse samples

**Regularized:** NNLS with Tikhonov-Phillips regularization
- Smooth, stable distributions
- Tunable regularization parameter (α)

## Export Options

### Jupyter Notebook (.ipynb)

Exports include:
- Markdown documentation of each step
- Executable code cells
- Parameter documentation
- Ready to run or modify

### Python Script (.py)

Standalone script with:
- All imports
- Step-by-step code
- Comments explaining each step

### PDF Report (Coming Soon)

Complete analysis report with:
- Plots and visualizations
- Results tables
- Parameter summary

## Transparency Philosophy

JADE-DLS GUI is designed to prevent "black box" analysis:

1. **Code Visibility**: Every action generates visible Python code
2. **Parameter Tracking**: All parameters are logged and displayed
3. **Export Capability**: Export to reproducible notebooks/scripts
4. **Documentation**: Built-in explanations of methods and equations
5. **Open Source**: All code is visible and modifiable

## Project Structure

```
JADE-DLS/
├── jade_dls_gui.py           # Main application launcher
├── gui/
│   ├── main_window.py        # Main GUI window
│   ├── core/
│   │   └── pipeline.py       # Transparent pipeline system
│   ├── widgets/
│   │   ├── workflow_panel.py # Left panel (workflow steps)
│   │   ├── analysis_view.py  # Center panel (plots, results)
│   │   └── inspector_panel.py # Right panel (code, params)
│   └── dialogs/              # Dialog windows
├── preprocessing.py          # Data loading (existing)
├── cumulants.py             # Cumulant analysis (existing)
├── cumulants_C.py           # Iterative cumulant (existing)
├── regularized.py           # NNLS & regularized (existing)
└── intensity.py             # Intensity processing (existing)
```

## Tips

- **Check R² values**: Good fits should have R² > 0.99
- **Inspect residuals**: Look for normal distribution
- **Compare methods**: Use the comparison tab to validate results
- **Export your work**: Always export notebooks for documentation
- **Use alpha analyzer**: Find optimal regularization for your data

## Troubleshooting

### GUI won't start

```bash
# Check PyQt5 installation
python -c "import PyQt5; print(PyQt5.__version__)"

# Reinstall if needed
pip install --upgrade PyQt5
```

### Import errors

Ensure you're running from the JADE-DLS directory:

```bash
cd /path/to/JADE-DLS
python jade_dls_gui.py
```

### Plots not showing

Install matplotlib for Qt5:

```bash
pip install matplotlib PyQt5
```

## Future Enhancements

- [ ] Batch processing multiple experiments
- [ ] Custom fit function editor
- [ ] Advanced plotting options
- [ ] Database integration for data management
- [ ] PDF report generation with plots
- [ ] Interactive tutorial system
- [ ] Cloud data storage integration

## Contributing

This is a transparent, scientific tool. Contributions welcome!

## License

See main JADE-DLS README for license information.

## Credits

Based on the original JADE-DLS Jupyter notebook analysis framework.

GUI Development: 2025

---

**Philosophy:** Science should be transparent, reproducible, and accessible.
