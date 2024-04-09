#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 16:24:20 2022

@author: sadegh
"""

#%%    

import numpy as np
import pygimli as pg
import pygimli.meshtools as mt
import pybert as pb
from pygimli.physics import ert

# %% load data(sheme) container
data = ert.createData(elecs=3.5*np.arange(50.1)-90, schemeName='wa')  
# will change 10.1 to 50.1 later # *10 makes the electrode spacing 10 times bigger
# A B N M
# data["a"], data["b"] = data["b"], data["a"]
data["k"] = ert.geometricFactors(data)
print(data)
# print(data["k"][:5])
pg.x(data)    # number of electrods and their positions

# %% create geometry (poly object)
ModelName = '2D2Anom_wa'

geo = mt.createWorld(start=[-150, 0], end=[150, -80], worldMarker=True, marker=0)
# Create a heterogeneous block
blockA = mt.createRectangle(start=[15, 0], end=[50, -7.0],
                            marker=1,  boundaryMarker=2, area=5)
blockB = mt.createRectangle(start=[-50, 0], end=[-15, -7.0],
                            marker=2,  boundaryMarker=3, area=5)
# Merge geometrical entities
plc = geo + blockA + blockB

for sen in data.sensors():   # to refine the model at the surface
    plc.createNode(sen)      # we create a node at any sensor position
    plc.createNode(sen-[0, 1.0])
pg.show(plc, boundaryMarker=True);
# res[mesh.cellMarkers()]  # resistiviry of each cell
# %% save the anomalies 
# We save the anomalies for later plotting lines on results
ano = plc
ano.exportPLC(f"geo_{ModelName}.poly")
# %% mesh the geometry
mesh = mt.createMesh(plc)
mesh.save(f"mesh_{ModelName}.bms")
ax, cb = pg.show(mesh, markers=True, showMesh=True, colorBar=True, dpi=200);
ax.figure.savefig(f"mesh_{ModelName}.png", dpi=200)
ax.figure.savefig(f"mesh_{ModelName}.pdf")
mesh.cellMarkers()   # number of cells in the mesh

 # %% Cole Cole Parameters for each layer
# the synthetic model
frvec = [0.3, 0.7, 1.0, 3.0, 7.0, 10.0, 20.0, 40.0, 80.0, 100.0,] # should be changed as following

# frvec = [0.3, 1., 3., 10., 30, 100.] # should be changed as following
# frvec = [10., 100] # should be changed as following

# frvec = [0.156, 0.312, 0.625, 1.25, 2.5, 5, 10, 20, 40, 80, 125,
#          250, 500, 1000]  # SIP256C frequencies #frequency vector
rhovec = np.array([80, 300, 400])  # resistivity in Ohmm
mvec = np.array([0.2, 0.45, 0.50])    # chargeability in V/V # chargables are non-zero
tauvec = np.array([0.4, 0.9, 0.99]) # time constants in s
cvec = np.array([0.6, 0.50, 0.40])   # relaxation exponent typically [0.1 .. 0.6]

fdip = pb.FDIP(f=frvec, data=data) 
synModel =np.hstack((np.log10(rhovec).ravel(), mvec.ravel(), np.log10(tauvec).ravel(), cvec.ravel()))
np.savetxt(f'synModel_{ModelName}.csv', synModel, delimiter=',')
# %% Plot True Cole-Cole values
if 1:
    fig, ax = pg.plt.subplots(nrows=4, figsize=(8, 12), sharex=True, sharey=True)
    cm = mesh.cellMarkers()
    pg.show(mesh, rhovec[cm], ax=ax[0], cMin=0, cMax=500, logScale=True,
        cMap="Spectral_r")
    pg.show(mesh, mvec[cm], ax=ax[1], cMin=0, cMax=0.8, logScale=0, cMap="plasma")
    pg.show(mesh, tauvec[cm], ax=ax[2], cMin=0.01, cMax=1, logScale=1, cMap="magma")
    pg.show(mesh, cvec[cm], ax=ax[3], cMin=0, cMax=0.5, logScale=0, cMap="viridis")
    ax[0].set_xlim(-100, 100)
    ax[0].set_ylim(-70, 0);

# %% check complex resistivity of each medium based on the predefined synthetic model
# Frequency-domain Cole-Cole impedance for each anomaly:
# Z = (1. - m * (1. - relaxationTerm(f, tau, c, a))) * rho
from pygimli.physics.SIP.models import modelColeColeRho
for i in range(len(frvec)):
    res = modelColeColeRho(frvec[i], rhovec, mvec, tauvec, cvec)
    print('f = ',frvec[i])
    print('Z =',res)    

# %% Forward modelling, FDIP simulation
# pybert.FDIP.simulate can do the forward modelling for many frequencies taking a Cole-Cole model as input
# now the actual simulation:
fdip.simulate(mesh, rhovec, mvec, tauvec, cvec, noiseAbs=0, noiseLevel=0.0, verbose=True); # noiseLevel=error percentage of the model

nDatapoint = 1   # for example
spec=fdip.getDataSpectrum(nDatapoint) # Return SIP spectrum class for single data number
spec.showData()
# %% Show pseudosections of a single frequency data
fdip.showSingleFrequencyData(4);
# fdip.showSingleFrequencyData(3);
# %%  Show decay curves
fdip.showDataSpectra(data, ab=[1, 2]); # resistivity phase is normaly negative / conductivity phase is normaly pasitive
# %% save rhoa, phia and array data
fdip.basename = f"synthSlag_{ModelName}"
fdip.saveData()

# %% save pdf files
fdip.generateDataPDF()
fdip.generateSpectraPDF()

# %% Forward Modelling Results to be used for the inversion
np.random.seed(42)
rhoa = np.copy(fdip.RHOA)  # make a copy for not to be noisified every time we call it
phia = np.copy(fdip.PHIA)
ampError = 0.05                              # relative error level (in percent)
ampNoise = ampError * pg.randn(rhoa.shape) # ampNoise: add Gaussian noise to error
rhoa *= ampNoise + 1.0           # noisified data   # why +1 ? because it is a relative error 
phiError = 0.001         # 1 miliradians
phiNoise = phiError * pg.randn(phia.shape)  
phia += phiNoise

# %%   extracting the required matrix to be used for the inversion
headers = np.array([('freq', 'amp', 'phia', 'amp_err', 'phia_err')])
frarray = np.array(frvec)   # frequency array

# frvecAll = np.tile(np.flipud(frarray),len(rhoa))
frvecAll = np.tile(frarray,len(rhoa))
# rhoaAllDatapoint = (np.fliplr(rhoa)).ravel() #using all of the apparent resistivity values
rhoaAllDatapoint = (rhoa).ravel() #using all of the apparent resistivity values
# phiaAllDatapoint = (np.fliplr(phia)).ravel() * 1000 #*1000 since unit is mrad
phiaAllDatapoint = (phia).ravel() * 1000 #*1000 since unit is mrad
amp_errAllDatapoint = rhoaAllDatapoint * ampError
phia_errAllDatapoint = np.ones_like(phiaAllDatapoint) * phiError * 1000 # *1000 since unit is mrad
SIPDataAll = np.column_stack((frvecAll, rhoaAllDatapoint, phiaAllDatapoint, amp_errAllDatapoint, phia_errAllDatapoint))# SIP data without header/ Frequency descending
SIPDataAllHeder = np.vstack([headers, SIPDataAll]) #  SIP data with header
# %% saving SIPDAta as a csv file
np.savetxt(f'SIPDataAll_{ModelName}.csv', SIPDataAllHeder,fmt='%s', delimiter=',')
# %% Test chi square
# rhoaPure = (np.fliplr(np.copy(fdip.RHOA))).ravel()
rhoaPure = (np.copy(fdip.RHOA)).ravel()
# phiaPeure = (np.fliplr(np.copy(fdip.PHIA))).ravel() * 1000 
phiaPeure = (np.copy(fdip.PHIA)).ravel() * 1000 
SIPDataAllPure = (np.array(list(zip(*[frvecAll, rhoaPure, phiaPeure,
                                        amp_errAllDatapoint, phia_errAllDatapoint ])))) # SIP data without header/ Frequency descending
SIPDataAllHederPure = np.vstack([headers, SIPDataAllPure])
np.savetxt(f'SIPDataAllPure_{ModelName}.csv', SIPDataAllHederPure,fmt='%s', delimiter=',')

pureDataReIm = np.array((SIPDataAllPure[:, 1]*np.cos(SIPDataAllPure[:, 2]/1000), SIPDataAllPure[:, 1]*np.sin(SIPDataAllPure[:, 2]/1000)))


