# 2D SIP Inversion with Two Rectangular Anomalies

This repository contains the Python scripts used to generate synthetic frequency-domain spectral induced polarization (FDIP/SIP) data, define a two-anomaly 2D forward model, and perform Bayesian inversion using MCMC. The workflow is built around **PyGIMLi**, **pyBERT**, and **emcee**.

The project implements a synthetic 2D subsurface model containing **two rectangular anomalies** embedded in a background medium, simulates complex resistivity responses over multiple frequencies, and then inverts the data with a probabilistic Markov chain Monte Carlo framework.

---

## Repository Contents

- `createSyntheticData_2D2Anoms.py`  
  Builds the synthetic 2D model with two rectangular anomalies, creates the mesh, defines the Cole-Cole parameters, simulates FDIP data, and exports the synthetic dataset.

- `MCMCfunctions_2D2Anoms.py`  
  Contains the inversion class and the MCMC likelihood/prior machinery used to sample the posterior distribution of model parameters.

- `peltonfdip_2D2Anoms.py`  
  Defines the forward model used in the Bayesian inversion. This file links the Cole-Cole parameterization to PyGIMLi’s ERT forward response.

- `myBISIPinversion_2D2Anom.py`  
  Main script for running the inversion. It loads the synthetic dataset, initializes the forward model, and launches the MCMC sampling.

- `MCMCPlots.py`  
  Produces chain plots, posterior diagnostics, and recovered model sections from the inversion results.

---

## Scientific Background

The inversion is based on the **Cole-Cole model** for frequency-dependent complex resistivity:

- \( \rho_0 \): DC resistivity
- \( m \): chargeability
- \( \tau \): time constant
- \( c \): relaxation exponent

The model is applied in a spatially varying 2D geometry with two rectangular anomalies. Synthetic complex resistivity data are generated across a set of frequencies and then inverted in a Bayesian framework.

---

## Workflow

1. **Create the synthetic model**  
   Run `createSyntheticData_2D2Anoms.py` to generate:
   - geometry
   - mesh
   - true Cole-Cole parameter fields
   - synthetic FDIP data
   - CSV files and mesh files for inversion

2. **Run the inversion**  
   Run `myBISIPinversion_2D2Anom.py` to:
   - load the synthetic data
   - initialize the forward operator
   - define the parameter priors
   - run MCMC sampling using `emcee`

3. **Visualize the results**  
   Run `MCMCPlots.py` to inspect:
   - trace plots
   - posterior chains
   - recovered parameter sections
   - inversion quality

---

## Requirements

The code is written in Python and depends on the following main packages:

- `numpy`
- `matplotlib`
- `pygimli`
- `pybert`
- `emcee`

Depending on your environment, you may also need:

- `scipy`
- `h5py`
- `pandas`

---

## Installation

A typical workflow is to create a dedicated Python environment and install the required packages.

Example using `conda`:

```bash
conda create -n sip-inversion python=3.11
conda activate sip-inversion

Then install the required libraries:

conda install -c gimli pygimli pybert
pip install emcee matplotlib numpy scipy h5py pandas
Note: PyGIMLi and pyBERT installation may depend on your operating system and package manager. Please refer to the official project documentation if installation issues occur.

How to Run
1. Generate synthetic data
python createSyntheticData_2D2Anoms.py

This script creates the synthetic 2D model, mesh files, Cole-Cole parameter files, and the synthetic SIP dataset.

2. Run the inversion
python myBISIPinversion_2D2Anom.py

This script runs the Bayesian inversion using the generated synthetic dataset.

3. Plot and inspect the results
python MCMCPlots.py

This script reads the MCMC output and produces diagnostic plots and recovered models.

Output Files

The scripts may generate files such as:

.bms mesh files
.poly geometry files
.csv synthetic model and data files
.h5 MCMC backend files
.png and .pdf figures for model and inversion results
Notes
The repository is designed for synthetic testing and method development.
The inversion is highly nonlinear and may be sensitive to:
parameter bounds
mesh discretization
noise level
number of walkers and MCMC steps
the forward mapping between model cells and data response

The code structure separates:

synthetic data generation
forward simulation
inversion
plotting

This makes it easier to modify the model geometry or parameterization independently.




