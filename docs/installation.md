# JADE-DLS Installation Guide

## Requirements

- Python 3.8 or higher
- pip package manager

## Installation Steps

### 1. Install Python Dependencies

Open a terminal/command prompt in the JADE-DLS directory and run:

```bash
pip install -r requirements.txt
```

This will install all required packages:
- numpy: Numerical computing
- scipy: Scientific computing
- pandas: Data manipulation
- matplotlib: Plotting
- statsmodels: Statistical analysis
- PyQt5: GUI framework
- scikit-learn: Machine learning (for peak clustering and robust regression)
- joblib: Parallel processing
- openpyxl: Excel file support

### 2. Verify Installation

Test that all packages are installed correctly:

```bash
python -c "import numpy, scipy, pandas, matplotlib, statsmodels, PyQt5, sklearn, joblib, openpyxl; print('All packages installed successfully!')"
```

### 3. Run JADE-DLS

```bash
python main.py
```

## Troubleshooting

### "No module named 'sklearn'" Error

This means scikit-learn is not installed. Install it with:

```bash
pip install scikit-learn
```

### "joblib not available" Warning

This means joblib is not installed, which will disable multiprocessing. Install it with:

```bash
pip install joblib
```

### Multiprocessing Not Working on Windows

If you see "falling back to sequential processing" on Windows:

1. Make sure joblib is installed: `pip install joblib`
2. Check the console output for the specific error message
3. Try reinstalling the dependencies: `pip install --upgrade --force-reinstall -r requirements.txt`

### PyQt5 Issues

If you have problems with PyQt5:

```bash
pip install --upgrade PyQt5
```

On Linux, you might need system packages:
```bash
sudo apt-get install python3-pyqt5
```

### Slow Performance

For best performance with large datasets:

1. Ensure joblib is installed for multiprocessing
2. Ensure scikit-learn is installed for optimized algorithms
3. Use an SSD for data storage
4. Close other applications to free up RAM

## Optional Dependencies

For better Excel export performance:

```bash
pip install xlsxwriter
```

## Development Setup

For development, you may also want:

```bash
pip install pytest  # For running tests
pip install black   # For code formatting
```

## Contact

For issues and bug reports, please contact the JADE-DLS development team.
