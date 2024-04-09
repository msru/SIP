#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 10:28:16 2022

@author: sadegh
Inversion class
"""

import emcee
import numpy as np
#from bisip import Inversion
from MCMCfunctions_2D2Anoms import Inversion
from pybert import FDIP
# the whole complex forward calculation is in pygimli.physics ert
from pygimli.utils import squeezeComplex, toPolar
from pygimli.physics import ert
from pygimli.physics.SIP.models import modelColeColeRho

class fdipPeltonNew(Inversion):
    """A generalized ColeCole inversion scheme for SIP data.
    Args:
        *args: Arguments passed to the Inversion class.
        mesh: a two-dimensional mesh created with pygimli.meshtools
        frvec: (:obj:`list`) frequency vector
        data: data container created with pygimli.physics.ert.ertScheme
        n_cells: number of model cells : number of the mediums
        **kwargs: Additional keyword arguments passed to the Inversion class.
     """

    def __init__(self, *args, mesh, frvec, data, n_cells=1, **kwargs):
        super().__init__(*args, **kwargs)

        self.n_cells = n_cells
        # Add ColeCole parameters for each cell to dict
        range_n_cells = list(range(self.n_cells))
        # the length of the model vector is defind as follows:
        # print(n_cells)
        self.params.update({f'log_r0{i}': [1.2, 3.0] for i in range_n_cells})
        self.params.update({f'm{i}': [0.0, 1.0] for i in range_n_cells})
        self.params.update({f'log_tau{i}': [-1.5, 0.5] for i in range_n_cells})
        self.params.update({f'c{i}': [0.0, 1.0] for i in range_n_cells})

        # self._bounds = np.array(self.param_bounds).T
        self.mesh = mesh
        # we need to initialize the frequency vecctor and synthetic data vector (data contaider)
        self.freq = frvec
        # the forward operator is initialize only once
        self.fop = ert.ERTModelling(sr=True, verbose=False)
        self.fop.data = data
        self.fop.setComplex(True)
        self.fop.setMesh(mesh, ignoreRegionManager=True)  # call mesh

    def forward(self, model):
        """Returns a ColeCole impedance.
        Args:
            mesh: 
            rhovec: ndarray of R0
            mvec: ndarray of m
            tauvec: ndarray of tau
            cvec: ndarray of c
        """
        mod = np.reshape(model, (4, -1))
        # model is a long vector and we need to bring it to the shape
        # print(mod)
        # making a matrix for the amp
        AMP = np.zeros((self.fop.data.size(), len(self.freq)))
        PHI = np.zeros_like(AMP)  # matrix for for phase
        
        
        # np.random.seed(42)
        # synmodel = ([2.30102999566398, 0.4, -0.698970004336018, 0.5])
        # dist = np.reshape(np.array(np.random.normal(synmodel, 0.1, 4)),(4, -1))
        # dist2 = np.reshape(np.array(synmodel),(4, -1))
        
        for i, f in enumerate(self.freq):
            # print(i, f)
            # compute Cole-Cole model
            # compute complex resistivity as a function of frequency f
            # res = modelColeColeRho(f, *mod)
            res = modelColeColeRho(f, 10**(mod[0]), mod[1], 10**(mod[2]), mod[3]) # complex resistivity as a resulf of a cole cole model parameters at each step
            # res = modelColeColeRho(f, 10**(dist2[0]), dist2[1], 10**(dist2[2]), dist2[3])
            # res = modelColeColeRho(f, 10**(dist[0]), dist[1], 10**(dist[2]), dist[3])
            # print(f, res)
            # assign res to all number of cells in the mesh
            resvec = res[self.mesh.cellMarkers()]
            # print(f, resvec, resvec.shape)
            bla = squeezeComplex(resvec)
            # print(f, bla, bla.shape)
            response = self.fop.response(bla) # the response complex 'apparent' resistivity (dödata:Re & Im)
            # Re_response , Im_response = (np.array(response)).reshape(2,int(len(response)/2))

            # print(i, f, response, response.shape)
            # print(response)
            # print(response.shape)

            rhoa, phia = toPolar(response) #toPolar Converts complex values array into amplitude and phase in radiant
            # print(rhoa, phia)
            AMP[:, i] = rhoa
            # phiai[phiai > np.pi/2] = np.pi - phiai[phiai > np.pi/2]
            PHI[:, i] = -phia  #* 1000
            # print(np.vstack((AMP.ravel(), PHI.ravel())))
        return np.vstack((AMP.ravel(), PHI.ravel()))
    
