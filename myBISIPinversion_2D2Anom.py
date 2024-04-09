#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 10:45:33 2022

@author: sadegh
SIP inversion uning MCMC
"""
# %%   FORWARD

from pygimli.physics.SIP.models import modelColeColeRho
from peltonfdip_2D2Anoms import fdipPeltonNew
import numpy as np
import pygimli as pg
import pybert as pb
from pygimli.physics import ert
import emcee

# %% calling forward results
ModelName = '2D2Anom_wa'

fdip = pb.FDIP(f"synthSlag_{ModelName}")
mesh = pg.load(f"mesh_{ModelName}.bms")
print(fdip)


# %% initialize forward model
filepath = f'SIPDataAll_{ModelName}.csv'
SIPmodel = fdipPeltonNew(filepath=filepath, mesh=mesh, frvec=fdip.freq,
                           data=fdip.data,
                           n_cells=3, nsteps=1000, nwalkers=32)  # initializing the class
# %% testing if our model retuns correctly (output shold be equals synthSlag)
if 0:
    rhovec = np.array([100, 200, 400])  # resistivity in Ohmm
    mvec = np.array([0.0, 0.8, 0.6])
    tauvec = np.array([0.01,0.1, 1.0])  # time constants in s
    cvec = np.array([0.25, 0.4, 0.5])   # relaxation exponent
    
    model = np.concatenate([np.log10(rhovec), mvec, np.log10(tauvec), cvec])
    
    output = SIPmodel.forward(model)  # calling the function
    print(output.shape)
    print(output)
        
# %% Set up the backend to save the resulted chain to a file
# Don't forget to clear it in case the file already exists
filename = f"1000_{ModelName}.h5"
backend = emcee.backends.HDFBackend(filename)
# backend.reset(SIPmodel.nwalkers, SIPmodel.ndim)    

# %% Fit the model to this data file
SIPmodel.fit(backend=backend)
# model._data     # load data
# %% Chi-Square
# np.savetxt('chiSquareVec_Homogenous.csv', SIPmodel.chiSquareVec, delimiter=',')

