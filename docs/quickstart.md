# JADE-DLS GUI Quick Start Guide

## Installation (5 minutes)

### Step 1: Install Dependencies

```bash
cd JADE-DLS
pip install -r requirements_gui.txt
```

### Step 2: Launch the GUI

**Option A - Using launcher script (Linux/Mac):**
```bash
./start_gui.sh
```

**Option B - Direct Python:**
```bash
python jade_dls_gui.py
```

**Option C - Windows:**
```cmd
python jade_dls_gui.py
```

## First Analysis (10 minutes)

### 1. Load Your Data

```
File > Load Data... (or Ctrl+O)
→ Select folder with .asc files
→ GUI shows: "Loaded X files"
```

### 2. Check Data Overview

```
Click "Data Overview" tab
→ See statistics
→ Verify files loaded correctly
```

### 3. Run Cumulant Analysis

```
Left Panel > Click "Cumulant C"
→ Click "▶ Run Selected"
→ Wait for analysis to complete
→ View results in "Results" tab
```

### 4. Inspect the Code (Transparency!)

```
Right Panel > Click "💻 Code" tab
→ See the exact Python code
→ Click "📋 Copy Code" to copy
```

### 5. Export Your Analysis

```
File > Export as Jupyter Notebook... (Ctrl+J)
→ Choose filename (e.g., my_analysis.ipynb)
→ Open in Jupyter to run/modify
```

## Layout Overview

```
┌─────────────────────────────────────────────────────────────┐
│  JADE-DLS Analysis Tool                    [File] [Analysis]│
├───────────┬─────────────────────────────┬───────────────────┤
│           │                             │                   │
│ WORKFLOW  │     ANALYSIS VIEW           │    INSPECTOR      │
│           │                             │                   │
│ Steps:    │  Tabs:                      │  Tabs:            │
│ ✓ Load    │  • Data Overview            │  • 💻 Code        │
│ ⚙️ Process │  • 📈 Plots                 │  • ⚙️ Params      │
│ ⏸️ Cumulant│  • 📋 Results               │  • 📖 Docs        │
│ ⏸️ NNLS    │  • ⚖️ Comparison            │                   │
│           │                             │  [Copy] [Export]  │
│ [▶ Run]   │                             │                   │
│           │                             │                   │
└───────────┴─────────────────────────────┴───────────────────┘
```

## Understanding the Three Panels

### Left: Workflow Panel
- Shows analysis steps
- Click to select a step
- "▶ Run Selected" to execute
- "▶ Run All" for full analysis

### Center: Analysis View
- **Data Overview**: Loaded files, statistics
- **Plots**: Interactive visualizations
- **Results**: Tables with Rh, D, PDI, etc.
- **Comparison**: Side-by-side method comparison

### Right: Inspector Panel
- **💻 Code**: Auto-generated Python code
- **⚙️ Parameters**: All settings and values
- **📖 Docs**: Explanation of methods

## Transparency Features

### 1. See the Code
Every action generates Python code you can see, copy, and run independently.

```python
# Example: What the GUI does when you click "Run Cumulant C"
from cumulants_C import plot_processed_correlations_iterative

fit_function = lambda x, a, b, c, d, e, f: ...
result = plot_processed_correlations_iterative(
    data,
    fit_function,
    fit_range=(1e-9, 10),
    method='lm'
)
```

### 2. Track Parameters
All parameters are logged:
- What values were used
- When they were changed
- Why (if you add notes)

### 3. Export Everything
- **Jupyter Notebook**: Fully reproducible analysis
- **Python Script**: Standalone executable code
- **PDF Report**: Complete documentation (coming soon)

## Common Workflows

### Workflow 1: Quick Monodisperse Analysis

1. Load data
2. Run "Cumulant C" (iterative fit)
3. Check R² > 0.99 and PDI < 0.1
4. Export results

**Time: ~5 minutes**

### Workflow 2: Polydisperse Sample

1. Load data
2. Run "Cumulant C" (baseline)
3. Run "Regularized" (for distribution)
4. Compare results in "Comparison" tab
5. Export notebook for documentation

**Time: ~10 minutes**

### Workflow 3: Compare All Methods

1. Load data
2. Click "▶ Run All" in workflow panel
3. Go to "Comparison" tab
4. Review which method gives best fit
5. Export complete report

**Time: ~15 minutes**

## Tips for Success

### ✅ Do's

- **Always check R²**: Good fits have R² > 0.99
- **Inspect residuals**: Should be normally distributed
- **Export your work**: Document everything
- **Use the code viewer**: Learn what's happening
- **Compare methods**: Validate with multiple approaches

### ❌ Don'ts

- **Don't ignore bad fits**: Low R² = unreliable results
- **Don't use just one method**: Cross-validate
- **Don't forget to export**: Reproducibility is key
- **Don't blindly trust**: Understand the code

## Keyboard Shortcuts

- `Ctrl+O`: Load Data
- `Ctrl+J`: Export as Jupyter Notebook
- `Ctrl+P`: Export as Python Script
- `Ctrl+Q`: Quit
- `F1`: Help/Tutorial

## Troubleshooting

### "Module not found" errors
```bash
pip install --upgrade -r requirements_gui.txt
```

### GUI looks ugly
The GUI uses Qt Fusion style. To change:
```python
# Edit jade_dls_gui.py
app.setStyle('Windows')  # or 'Fusion', 'WindowsVista'
```

### Plots not showing
```bash
pip install --upgrade matplotlib PyQt5
```

### Can't find my results
Check the Analysis View > Results tab. Also check the code viewer to see where files are saved.

## Next Steps

1. **Read the full README**: `GUI_README.md`
2. **Check example export**: `gui/examples/example_export.py`
3. **Explore parameter dialogs**: Right-click on steps (coming soon)
4. **Join the community**: Report issues on GitHub

## Philosophy

**JADE-DLS GUI is designed to be transparent, not a black box.**

Every click generates code. Every parameter is visible. Every analysis is exportable.

**Science should be reproducible. Software should be transparent.**

---

Need help? Check `GUI_README.md` or open an issue on GitHub.

Happy analyzing! 🔬
