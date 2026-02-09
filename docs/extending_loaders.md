# Extending Data Loaders

ADE-DLS is designed with an extensible architecture to support DLS data from various instruments. This guide shows how to add support for new data formats.

## Current Support

- **ALV Correlator** (.ASC files) - Fully supported via `ALVDataLoader`

## Architecture Overview

The data loading system is located in `ade_dls.core.data_loader` and follows a simple class-based pattern.

## Adding a New Loader

### Step 1: Create Loader Class

Create a new loader class in `ade_dls/core/data_loader.py`:

```python
class MalvernDataLoader:
    """
    Data loader for Malvern Zetasizer files.

    Supports .txt and .dts file formats from Malvern instruments.
    """

    def __init__(self):
        """Initialize Malvern data loader."""
        self.supported_extensions = ['.txt', '.dts']

    def load_metadata(self, filepath):
        """
        Extract metadata from Malvern file.

        Parameters
        ----------
        filepath : str or Path
            Path to Malvern data file

        Returns
        -------
        pd.DataFrame
            DataFrame with metadata columns:
            - angle [°]
            - temperature [K]
            - wavelength [nm]
            - refractive_index
            - viscosity [cp]
            - filename, folder
        """
        import pandas as pd

        # Parse Malvern header format
        metadata = {}

        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith('Temperature'):
                    metadata['temperature [K]'] = float(line.split(':')[1].strip())
                elif line.startswith('Angle'):
                    metadata['angle [°]'] = float(line.split(':')[1].strip())
                # ... parse other fields

        return pd.DataFrame([metadata])

    def load_correlation(self, filepath):
        """
        Extract correlation function from Malvern file.

        Parameters
        ----------
        filepath : str or Path
            Path to Malvern data file

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            - t (s): lag times
            - g(2): normalized correlation function g(2)(τ) - 1
        """
        import pandas as pd
        import numpy as np

        # Parse Malvern correlation data
        tau = []
        g2 = []

        with open(filepath, 'r') as f:
            in_data_section = False
            for line in f:
                if 'Correlation Function' in line:
                    in_data_section = True
                    continue
                if in_data_section and line.strip():
                    parts = line.split()
                    tau.append(float(parts[0]))
                    g2.append(float(parts[1]))

        return pd.DataFrame({
            't (s)': np.array(tau),
            'g(2)': np.array(g2)
        })

    def is_supported(self, filepath):
        """Check if file format is supported."""
        from pathlib import Path
        path = Path(filepath)
        return path.suffix in self.supported_extensions
```

### Step 2: Export in Module

Add your loader to `ade_dls/core/data_loader.py`:

```python
__all__ = [
    'ALVDataLoader',
    'MalvernDataLoader',  # Add your loader
    'extract_data',
    'extract_correlation',
    'extract_countrate',
]
```

### Step 3: Create Tests

Add tests in `tests/core/test_data_loader.py`:

```python
import pytest
from ade_dls.core.data_loader import MalvernDataLoader

def test_malvern_loader_metadata():
    """Test Malvern metadata extraction."""
    loader = MalvernDataLoader()
    metadata = loader.load_metadata('tests/fixtures/malvern_sample.txt')

    assert 'temperature [K]' in metadata.columns
    assert 'angle [°]' in metadata.columns
    assert len(metadata) == 1

def test_malvern_loader_correlation():
    """Test Malvern correlation function loading."""
    loader = MalvernDataLoader()
    df = loader.load_correlation('tests/fixtures/malvern_sample.txt')

    assert 't (s)' in df.columns
    assert 'g(2)' in df.columns
    assert len(df) > 0
    assert (df['t (s)'] > 0).all()  # Positive lag times
```

### Step 4: Add Test Fixtures

Place sample data files in `tests/fixtures/`:

```
tests/
└── fixtures/
    ├── alv_sample.ASC
    └── malvern_sample.txt  # Your test file
```

### Step 5: Update Documentation

1. Update `README.md` supported formats section
2. Add usage example to `examples/`
3. Update this guide with specific notes

## Integration with GUI

To integrate your loader with the GUI, modify `ade_dls/gui/core/data_loader.py`:

```python
from ade_dls.core.data_loader import ALVDataLoader, MalvernDataLoader

class DataLoader:
    def __init__(self):
        self.loaders = {
            'ALV': ALVDataLoader(),
            'Malvern': MalvernDataLoader(),  # Add your loader
        }

    def auto_detect_format(self, filepath):
        """Auto-detect file format."""
        for name, loader in self.loaders.items():
            if loader.is_supported(filepath):
                return name
        return None
```

## API Requirements

Your loader should implement:

### Required Methods

- `load_metadata(filepath)` → pd.DataFrame with metadata
- `load_correlation(filepath)` → pd.DataFrame with ['t (s)', 'g(2)']
- `is_supported(filepath)` → bool

### Required Metadata Fields

Your `load_metadata()` must return a DataFrame with these columns:

- `angle [°]` (float): Scattering angle in degrees
- `temperature [K]` (float): Temperature in Kelvin
- `wavelength [nm]` (float): Laser wavelength in nanometers
- `refractive_index` (float): Solvent refractive index
- `viscosity [cp]` (float): Solvent viscosity in centipoise
- `filename` (str): Original filename
- `folder` (str): Parent folder name

Optional fields:
- `duration [s]` (float): Measurement duration
- `q` (float): Scattering vector magnitude
- `q^2` (float): Squared scattering vector

### Correlation Data Format

Your `load_correlation()` must return:

```python
pd.DataFrame({
    't (s)': np.array([...]),     # Lag times in seconds
    'g(2)': np.array([...])       # g(2)(τ) - 1 (normalized, baseline-subtracted)
})
```

## Common Patterns

### Handling Multiple Encodings

```python
def load_with_encoding(filepath, encodings=['utf-8', 'latin-1', 'windows-1252']):
    for encoding in encodings:
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                return parse_file(f)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode {filepath} with any encoding")
```

### Parsing Headers

```python
def parse_header(filepath):
    metadata = {}
    with open(filepath, 'r') as f:
        for line in f:
            if ':' in line:
                key, value = line.split(':', 1)
                metadata[key.strip()] = value.strip()
            if 'End of Header' in line:
                break
    return metadata
```

### Validating Data

```python
def validate_correlation(df):
    """Validate correlation data format."""
    assert 't (s)' in df.columns, "Missing 't (s)' column"
    assert 'g(2)' in df.columns, "Missing 'g(2)' column"
    assert len(df) > 10, "Too few data points"
    assert (df['t (s)'] > 0).all(), "Negative lag times"
    assert df['g(2)'].notna().all(), "NaN values in correlation"
```

## Example: Brookhaven Loader

Here's a complete example for Brookhaven BI-9000AT correlator:

```python
class BrookhavenDataLoader:
    """Loader for Brookhaven BI-9000AT .cor files."""

    def __init__(self):
        self.supported_extensions = ['.cor']

    def load_metadata(self, filepath):
        import pandas as pd
        from pathlib import Path

        metadata = {
            'filename': Path(filepath).name,
            'folder': Path(filepath).parent.name,
        }

        with open(filepath, 'r') as f:
            lines = f.readlines()

            # Parse Brookhaven header format
            for line in lines[:20]:  # Header is in first 20 lines
                if 'Temperature' in line:
                    metadata['temperature [K]'] = float(line.split()[-1]) + 273.15
                elif 'Angle' in line:
                    metadata['angle [°]'] = float(line.split()[-1])
                elif 'Wavelength' in line:
                    metadata['wavelength [nm]'] = float(line.split()[-1])
                elif 'Viscosity' in line:
                    metadata['viscosity [cp]'] = float(line.split()[-1])
                elif 'Refractive Index' in line:
                    metadata['refractive_index'] = float(line.split()[-1])

        return pd.DataFrame([metadata])

    def load_correlation(self, filepath):
        import pandas as pd
        import numpy as np

        tau = []
        g2 = []

        with open(filepath, 'r') as f:
            lines = f.readlines()

            # Find start of data section
            data_start = 0
            for i, line in enumerate(lines):
                if 'Channel' in line and 'Delay' in line:
                    data_start = i + 1
                    break

            # Parse data
            for line in lines[data_start:]:
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 3:
                        tau.append(float(parts[1]))  # Delay time
                        g2.append(float(parts[2]))   # Correlation

        return pd.DataFrame({
            't (s)': np.array(tau),
            'g(2)': np.array(g2)
        })

    def is_supported(self, filepath):
        from pathlib import Path
        return Path(filepath).suffix in self.supported_extensions
```

## Testing Your Loader

```python
# Quick test
from ade_dls.core.data_loader import YourLoader

loader = YourLoader()

# Test metadata
metadata = loader.load_metadata('sample_file.ext')
print(metadata)

# Test correlation
correlation = loader.load_correlation('sample_file.ext')
print(correlation.head())

# Verify required columns
assert 'temperature [K]' in metadata.columns
assert 't (s)' in correlation.columns
assert 'g(2)' in correlation.columns
```

## Contributing

When contributing a new loader:

1. Follow the API requirements above
2. Include test files (anonymized/synthetic if needed)
3. Add docstrings (NumPy style)
4. Update README.md
5. Create a PR with:
   - Loader implementation
   - Tests
   - Sample data
   - Documentation updates

## Questions?

- Open an issue with tag `data-loader`
- Discuss in GitHub Discussions
- Check existing loaders for examples

## Future: Base Class

In future versions, we plan to add an abstract base class:

```python
from abc import ABC, abstractmethod

class DLSDataLoader(ABC):
    @abstractmethod
    def load_metadata(self, filepath):
        pass

    @abstractmethod
    def load_correlation(self, filepath):
        pass

    @abstractmethod
    def is_supported(self, filepath):
        pass
```

This is deferred to avoid breaking changes during the initial restructuring.
