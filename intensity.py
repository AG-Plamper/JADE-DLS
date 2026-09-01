# -*- coding: utf-8 -*-
"""
intensity.py
============
Retired (v2.4) -- merged into preprocessing.py.

extract_intensity() is gone; preprocessing.extract_data() now extracts
MeanCR0, MeanCR1 and Monitor Diode in the same file pass as angle,
temperature, wavelength, refractive_index, viscosity and duration (one
file read instead of two per measurement). plot_meancr() moved to
preprocessing.py unchanged.

Update any `from intensity import ...` to `from preprocessing import ...`.
"""
