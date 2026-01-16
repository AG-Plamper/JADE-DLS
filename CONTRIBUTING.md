# Contributing to ADE-DLS

Thank you for your interest in contributing to ADE-DLS! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Adding New Data Loaders](#adding-new-data-loaders)

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help create a welcoming environment for all contributors
- Scientific rigor and reproducibility are priorities

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/JADE-DLS.git
   cd JADE-DLS
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/traianuschem/JADE-DLS.git
   ```

## Development Setup

### Install in Development Mode

```bash
# Install with all dependencies including dev tools
pip install -e ".[all,dev]"

# Or install minimal dev setup
pip install -e .
pip install pytest pytest-cov black isort flake8 mypy
```

### Verify Installation

```bash
python -c "import ade_dls; print(ade_dls.__version__)"
pytest tests/  # Once tests are added
```

## How to Contribute

### Reporting Bugs

1. **Search existing issues** to avoid duplicates
2. **Create a new issue** with:
   - Clear, descriptive title
   - Steps to reproduce
   - Expected vs actual behavior
   - Python version, OS, package versions
   - Minimal code example if applicable

### Suggesting Features

1. **Check existing feature requests**
2. **Open an issue** describing:
   - The problem you're trying to solve
   - Proposed solution
   - Alternatives considered
   - Impact on existing functionality

### Contributing Code

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**:
   - Follow coding standards (see below)
   - Add tests for new functionality
   - Update documentation

3. **Test your changes**:
   ```bash
   pytest tests/
   black ade_dls/ tests/
   flake8 ade_dls/ tests/
   ```

4. **Commit with clear messages**:
   ```bash
   git add .
   git commit -m "Add feature: brief description

   More detailed explanation of what changed and why."
   ```

5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Open a Pull Request** on GitHub

## Coding Standards

### Python Style

- Follow **PEP 8** style guide
- Use **Black** for code formatting (line length: 100)
- Use **isort** for import sorting
- Maximum line length: 100 characters

```bash
# Format code
black ade_dls/ tests/

# Sort imports
isort ade_dls/ tests/

# Check style
flake8 ade_dls/ tests/
```

### Code Quality

- Write clear, self-documenting code
- Add docstrings for all public functions/classes (NumPy style)
- Use type hints where appropriate
- Keep functions focused and modular
- Avoid premature optimization

### Docstring Format (NumPy Style)

```python
def analyze_correlation(tau, g2, method='cumulant'):
    """
    Analyze autocorrelation function to extract decay rate.

    Parameters
    ----------
    tau : np.ndarray
        Array of lag times in seconds
    g2 : np.ndarray
        Normalized autocorrelation function g(2)(τ) - 1
    method : str, optional
        Analysis method: 'cumulant', 'nnls', or 'regularized'
        Default is 'cumulant'

    Returns
    -------
    dict
        Dictionary containing analysis results with keys:
        - 'decay_rate': Decay rate Γ in Hz
        - 'radius': Hydrodynamic radius in nm
        - 'polydispersity': Polydispersity index

    Raises
    ------
    ValueError
        If tau and g2 arrays have different lengths

    Examples
    --------
    >>> tau = np.logspace(-6, 0, 100)
    >>> g2 = np.exp(-2 * tau / 1e-4)
    >>> results = analyze_correlation(tau, g2)
    >>> print(results['decay_rate'])
    10000.0
    """
    pass
```

## Testing

### Writing Tests

- Place tests in `tests/` directory
- Mirror package structure: `tests/analysis/test_cumulants.py`
- Use pytest framework
- Aim for >80% code coverage

```python
# tests/analysis/test_cumulants.py
import pytest
import numpy as np
from ade_dls.analysis import cumulants

def test_cumulant_method_b_basic():
    """Test basic functionality of Method B"""
    # Arrange
    tau = np.logspace(-6, 0, 50)
    g2 = np.exp(-2 * tau / 1e-4)

    # Act
    result = cumulants.method_b(tau, g2)

    # Assert
    assert 'gamma' in result
    assert result['gamma'] > 0
    np.testing.assert_allclose(result['gamma'], 10000, rtol=0.1)
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=ade_dls --cov-report=html

# Run specific test file
pytest tests/analysis/test_cumulants.py

# Run specific test
pytest tests/analysis/test_cumulants.py::test_cumulant_method_b_basic
```

## Submitting Changes

### Pull Request Guidelines

1. **Target the correct branch**: Usually `main` or `develop`
2. **One feature per PR**: Keep changes focused
3. **Write clear PR description**:
   - What does this PR do?
   - Why is this change needed?
   - How was it tested?
   - Any breaking changes?

4. **Ensure CI passes**: All tests and checks must pass
5. **Respond to feedback**: Address review comments promptly
6. **Keep PR updated**: Rebase on latest main if needed

### PR Template

```markdown
## Description
Brief description of changes

## Motivation
Why is this change needed?

## Changes
- List of specific changes
- Made in this PR

## Testing
How was this tested?
- [ ] Unit tests added/updated
- [ ] Manual testing performed
- [ ] Examples verified

## Checklist
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] Tests pass locally
- [ ] CHANGELOG.md updated (if applicable)

## Breaking Changes
List any breaking changes and migration instructions
```

## Adding New Data Loaders

To support a new DLS instrument, extend the data loader system:

### 1. Create New Loader Class

```python
# ade_dls/core/data_loader.py

class MalvernDataLoader:
    """
    Data loader for Malvern Zetasizer files.

    Supports .txt and .dts file formats from Malvern instruments.
    """

    def load(self, filepath):
        """
        Load Malvern data file.

        Parameters
        ----------
        filepath : str or Path
            Path to Malvern data file

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: 't (s)', 'g(2)', 'count_rate'
        """
        pass

    def get_metadata(self, filepath):
        """
        Extract metadata from file header.

        Returns
        -------
        dict
            Metadata including temperature, angle, wavelength, etc.
        """
        pass
```

### 2. Add Tests

```python
# tests/core/test_data_loader.py

def test_malvern_loader():
    loader = MalvernDataLoader()
    df = loader.load("tests/fixtures/malvern_sample.txt")

    assert 't (s)' in df.columns
    assert 'g(2)' in df.columns
    assert len(df) > 0
```

### 3. Update Documentation

- Add loader to README.md supported formats
- Create usage example in `examples/`
- Update `docs/extending_loaders.md`

### 4. Submit PR

Include:
- Loader implementation
- Tests with fixture data
- Documentation updates
- Example usage

## Questions?

If you have questions:
- Open a [Discussion](https://github.com/traianuschem/JADE-DLS/discussions)
- Check existing [Issues](https://github.com/traianuschem/JADE-DLS/issues)
- Contact maintainers

Thank you for contributing to ADE-DLS!
