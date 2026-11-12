# JADE-DLS: Jupyter-based Angular Dependent Evaluator for Dynamic Light Scattering

JADE-DLS is a Python tool in a Jupyter notebook for the angle-resolved analysis of dynamic light scattering (DLS) data.
The intensity autocorrelation functions g(2)(τ)−1 of all measured angles are fitted to obtain the decay rate Γ, and the
translational diffusion coefficient D follows from the linear regression of Γ against q². The hydrodynamic radius is then
calculated with the Stokes–Einstein equation. Every step is shown in the notebook and can be inspected.

**Methods:** cumulant analysis (ALV software values, linear and nonlinear fits), multimodal cumulant analysis (Dirac delta
modes), NNLS and Tikhonov–Phillips regularized fit, cross-angle clustering of populations and population-resolved
static light scattering (Guinier analysis).

**Input:** .ASC files of ALV correlators.

**ADE-DLS**, a graphical user interface with automated analysis presets and provenance recording, is available at
https://github.com/traianuschem/JADE-DLS.

## Example data
- monomodal polymer sample
- bimodal polymer sample
- gold nanoparticles with diameters of 30, 100 and 200 nm (single components, binary and ternary mixtures)
- synthetic data generator with benchmark results (obtained with version 3.0)

## Requirements
Python 3.x with numpy, scipy, pandas, matplotlib, scikit-learn and jupyter.

## License and citation
GPL-3.0. If you use JADE-DLS, please cite:

@software{JADE-DLS,
  author  = {Schildknecht, Vincent and Neubert, Richard},
  title   = {{JADE-DLS}: Jupyter-based Angular Dependent Evaluator for Dynamic Light Scattering},
  version = {3.1},
  year    = {2026},
  url     = {https://github.com/AG-Plamper/JADE-DLS},
  license = {GPL-3.0}
  doi     = {10.5281/zenodo.16841738}
}
