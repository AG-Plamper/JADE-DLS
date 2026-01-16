# ADE-DLS: Angular Dependent Evaluator for Dynamic Light Scattering

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

ADE-DLS is a comprehensive Python package for analyzing Dynamic Light Scattering (DLS) data to determine hydrodynamic radius and particle size distributions.

## Features

### Analysis Methods

- **Cumulant Analysis** (3 methods)
  - Method A: Uses ALV software pre-calculated fits
  - Method B: Linear fit of ln[g(τ)^0.5] vs τ
  - Method C: Iterative non-linear least squares

- **Inverse Laplace Transform Methods**
  - NNLS (Non-Negative Least Squares)
  - Regularized NNLS (Tikhonov-Phillips regularization)
  - Advanced constraints: normalization, sparsity, unimodality

- **Data Processing**
  - Automated file loading (.ASC files from ALV correlator)
  - Preprocessing and filtering
  - Peak clustering for multi-modal distributions
  - Diffusion coefficient calculation
  - Hydrodynamic radius determination

### GUI Application

- **Interactive PyQt5-based interface**
- **Transparent analysis pipeline** - generates reproducible Jupyter notebooks and Python scripts
- **Comprehensive visualization** - real-time plots and result inspection
- **Batch processing** - analyze multiple files at different angles
- **Export capabilities** - Excel, CSV, Jupyter notebooks, Python scripts

## Installation

### From PyPI (coming soon)

```bash
pip install ade-dls
```

### From Source

```bash
git clone https://github.com/traianuschem/JADE-DLS.git
cd JADE-DLS
pip install -e .
```

### With GUI Support

```bash
pip install ade-dls[gui]
```

### Full Installation (all features)

```bash
pip install ade-dls[all]
```

## Quick Start

### GUI Application

Launch the graphical interface:

```bash
ade-dls
```

Or with Python:

```python
python -m ade_dls.gui.main_window
```

### Python API

```python
import pandas as pd
from ade_dls.core import preprocessing
from ade_dls.analysis import cumulants, regularized

# Load data
dataframes = preprocessing.load_and_preprocess_data(
    folder_path="path/to/data",
    file_extension=".ASC"
)

# Cumulant analysis
cumulant_params = {
    'method': 'C',
    'wavelength': 632.8,  # nm
    'temperature': 298.15,  # K
    'viscosity': 0.89,  # cP
    'refractive_index': 1.33
}
results = cumulants.cumulant_method_c_all(dataframes, cumulant_params)

# NNLS analysis
import numpy as np
nnls_params = {
    'decay_times': np.logspace(-6, 0, 100),
    'prominence': 0.05,
    'distance': 5
}
nnls_results = regularized.nnls_all(dataframes, nnls_params)

print(results)
```

## Documentation

- [Installation Guide](docs/installation.md)
- [User Guide](docs/user_guide.md)
- [GUI Quick Start](docs/quickstart.md)
- [Data Filtering Guide](docs/FILTERING_GUIDE.md)
- [Contributing Guidelines](CONTRIBUTING.md)

## Requirements

- Python 3.8+
- NumPy >= 1.22.0
- SciPy >= 1.8.0
- Pandas >= 1.5.0
- Matplotlib >= 3.5.0
- scikit-learn >= 1.0.0
- statsmodels >= 0.13.0

**For GUI:**
- PyQt5 >= 5.15.0

## Supported Data Formats

Currently supports:
- **ALV correlator** (.ASC files)

Extensible architecture allows easy addition of custom data loaders for other instruments.

## Citation

If you use ADE-DLS in your research, please cite:

```bibtex
@software{ade_dls,
  title = {ADE-DLS: Angular Dependent Evaluator for Dynamic Light Scattering},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/traianuschem/JADE-DLS}
}
```

## License

This project is licensed under the GNU General Public License v3.0 or later - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Originally developed as JADE-DLS (Jupyter-based Angular Dependent Evaluator)
- Evolved from Jupyter notebooks to a full-featured GUI application

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Contact

For questions, issues, or suggestions:
- Open an issue: [GitHub Issues](https://github.com/traianuschem/JADE-DLS/issues)
- Repository: [https://github.com/traianuschem/JADE-DLS](https://github.com/traianuschem/JADE-DLS)

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and changes.
