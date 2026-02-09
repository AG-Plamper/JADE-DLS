# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2025-01-16

### Changed
- **BREAKING**: Renamed project from JADE-DLS to ADE-DLS (Angular Dependent Evaluator for Dynamic Light Scattering)
  - Removed "Jupyter-based" from name as project now features a full PyQt5 GUI
- **BREAKING**: Complete package restructuring to follow Python best practices
  - Reorganized as proper installable package `ade_dls`
  - Added `pyproject.toml` for modern Python packaging
  - Moved all modules into structured package hierarchy
- **BREAKING**: Changed license from CC-BY-NC-SA 4.0 to GPL-3.0+ (required by PyQt5)

### Added
- Proper package structure with `ade_dls` namespace
  - `ade_dls.core`: Data loading and preprocessing
  - `ade_dls.analysis`: Analysis algorithms (cumulants, regularized methods, peak clustering)
  - `ade_dls.gui`: PyQt5 GUI application
  - `ade_dls.utils`: Utility functions
- Installation support via pip (`pip install ade-dls`)
- Command-line entry point: `ade-dls` launches GUI
- Enhanced documentation structure
- CONTRIBUTING.md for contributor guidelines
- Optional dependency groups: `[gui]`, `[export]`, `[dev]`, `[all]`

### Improved
- Consolidated requirements into `pyproject.toml`
- Better import organization
- Preparation for test suite and CI/CD

### Fixed
- Import paths now follow standard Python package conventions

## [1.0.0] - 2024

### Added
- Initial release as JADE-DLS
- Jupyter notebook-based analysis
- PyQt5 GUI application
- Cumulant analysis methods (A, B, C)
- NNLS and regularized NNLS methods
- Peak clustering algorithms
- ALV .ASC file support
- Data filtering and preprocessing
- Export to Jupyter notebooks and Python scripts
- Batch processing capabilities
- Comprehensive visualization

### Analysis Methods
- Method A: ALV software pre-calculated fits
- Method B: Linear fit of ln[g(τ)^0.5] vs τ
- Method C: Iterative non-linear least squares
- NNLS (Non-Negative Least Squares)
- Regularized NNLS with Tikhonov-Phillips regularization
- Advanced constraints: normalization, sparsity, unimodality

---

## Migration Guide: v1.0 → v2.0

### Import Changes

**Old (v1.0):**
```python
import preprocessing
import cumulants
import regularized
from gui.main_window import MainWindow
```

**New (v2.0):**
```python
from ade_dls.core import preprocessing
from ade_dls.analysis import cumulants, regularized
from ade_dls.gui.main_window import MainWindow
```

### Installation Changes

**Old (v1.0):**
```bash
git clone repo
python jade-dls-gui.py
```

**New (v2.0):**
```bash
pip install ade-dls[all]
ade-dls  # or: python -m ade_dls.gui.main_window
```

### Configuration

All functionality remains the same; only import paths have changed.

---

[2.0.0]: https://github.com/traianuschem/JADE-DLS/compare/v1.0.0...v2.0.0
[1.0.0]: https://github.com/traianuschem/JADE-DLS/releases/tag/v1.0.0
