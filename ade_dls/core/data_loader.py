"""
Data loader for DLS instruments

Currently supports ALV correlator .ASC files.
Extensible architecture for adding support for other instruments.
"""

from .preprocessing import (
    extract_data,
    extract_correlation,
    extract_countrate,
    find_correlation_row,
    find_countrate_row,
    get_folder_name
)

__all__ = [
    'ALVDataLoader',
    'extract_data',
    'extract_correlation',
    'extract_countrate',
]


class ALVDataLoader:
    """
    Data loader for ALV correlator files (.ASC format).

    Handles:
    - Base metadata extraction (angle, temperature, wavelength, etc.)
    - Correlation function data
    - Count rate data
    - Multiple encoding support for different ALV software versions

    Examples
    --------
    >>> loader = ALVDataLoader()
    >>> metadata = loader.load_metadata('sample.ASC')
    >>> correlation = loader.load_correlation('sample.ASC')
    >>> countrate = loader.load_countrate('sample.ASC')
    """

    def __init__(self):
        """Initialize ALV data loader."""
        self.supported_extensions = ['.ASC', '.asc']

    def load_metadata(self, filepath, encoding='Windows-1252'):
        """
        Extract metadata from ALV .ASC file.

        Parameters
        ----------
        filepath : str or Path
            Path to .ASC file
        encoding : str, optional
            File encoding, default 'Windows-1252'

        Returns
        -------
        pd.DataFrame
            DataFrame with metadata including:
            - angle [°]
            - temperature [K]
            - wavelength [nm]
            - refractive_index
            - viscosity [cp]
            - duration [s]
            - q, q^2
            - filename, folder
        """
        return extract_data(filepath)

    def load_correlation(self, filepath, encoding='Windows-1252'):
        """
        Extract correlation function from ALV .ASC file.

        Parameters
        ----------
        filepath : str or Path
            Path to .ASC file
        encoding : str, optional
            File encoding, default 'Windows-1252'

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            - t (s): lag times
            - g(2): normalized correlation function g(2)(τ) - 1
        """
        return extract_correlation(filepath, encoding=encoding)

    def load_countrate(self, filepath, encoding='Windows-1252'):
        """
        Extract count rate data from ALV .ASC file.

        Parameters
        ----------
        filepath : str or Path
            Path to .ASC file
        encoding : str, optional
            File encoding, default 'Windows-1252'

        Returns
        -------
        pd.DataFrame
            DataFrame with count rate time series
        """
        return extract_countrate(filepath, encoding=encoding)

    def is_supported(self, filepath):
        """
        Check if file format is supported.

        Parameters
        ----------
        filepath : str or Path
            Path to file

        Returns
        -------
        bool
            True if file extension is supported
        """
        from pathlib import Path
        path = Path(filepath)
        return path.suffix in self.supported_extensions

    def __repr__(self):
        return f"ALVDataLoader(supported={self.supported_extensions})"
