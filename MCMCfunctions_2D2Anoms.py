#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 11:52:49 2022

@author: sadegh
"""
import numpy as np
import emcee
from multiprocessing import Pool
from pygimli.utils import squeezeComplex, toPolar
import warnings


ModelName = '2D2Anom_dd'

class Inversion():
    """A class to perform Bayesian inversion of SIP data.

    Args:
        filepath (:obj:`str`): The path to the SIP data file to perform inversion on.
        nwalkers (:obj:`int`): Number of walkers to use to explore the
            parameter space. Defaults to 32.
        nsteps (:obj:`int`): Number of steps to perform in the MCMC
            simulation. Defaults to 1000.
        headers (:obj:`int`): The number of header lines in the SIP data  file.
            Defaults to 1.
        ph_units (:obj:`str`): The units of the phase shift measurements.
            Choices: 'mrad', 'rad', 'deg'. Defaults to 'mrad'.
            """

    def __init__(self, filepath, nwalkers=32, nsteps=1000, headers=1,
                 ph_units='mrad'):

        # Get arguments
        self.filepath = filepath
        self.nwalkers = nwalkers
        self.nsteps = nsteps
        self.headers = headers
        self.ph_units = ph_units

        # Set default attributes
        self._p0 = None
        self._params = {}
        self.__fitted = False
        self.chiSquareVec = []
        self.thetaVec =[]
        
        # Load data
        self._data = self.load_data(self.filepath, self.headers, self.ph_units)

    def load_data(self, filename, headers=1, ph_units='mrad'):
        """Imports a data file and prepares it for inversion.

        Args:
            filepath (:obj:`str`): The path to the data file.
            headers (:obj:`int`): The number of header lines in the file.
                Defaults to 1.
            ph_units (:obj:`str`): The units of the phase shift measurements.
                Choices: 'mrad', 'rad', 'deg'. Defaults to 'mrad'.

        """
        # Importation des données .DAT
        dat_file = np.loadtxt(f'{filename}', skiprows=headers, delimiter=',')
        labels = ['freq', 'amp', 'pha', 'amp_err', 'pha_err']
        data = {l: dat_file[:, i] for (i, l) in enumerate(labels)}

        if ph_units == 'mrad':
            data['pha'] = data['pha']/1000  # mrad to rad
            data['pha_err'] = data['pha_err']/1000  # mrad to rad
        if ph_units == 'deg':
            data['pha'] = np.radians(data['pha'])  # deg to rad
            data['pha_err'] = np.radians(data['pha_err'])  # deg to rad

        # data['Z'] = data['amp']*(np.cos(data['pha']) + 1j*np.sin(data['pha']))
        # EI = np.sqrt(((data['amp']*np.cos(data['pha'])*data['pha_err'])**2)
        #              + (np.sin(data['pha'])*data['amp_err'])**2)
        # ER = np.sqrt(((data['amp']*np.sin(data['pha'])*data['pha_err'])**2)
        #              + (np.cos(data['pha'])*data['amp_err'])**2)
        # data['Z_err'] = ER + 1j*EI

        # z = data['Z'] # data in Re/Im form
        # z_ReIm = squeezeComplex(z)
        # d_amp, d_ph = toPolar(z_ReIm)

        # z_e = data['Z_err'] # data err in Re/Im form
        # z_eReIm = squeezeComplex(z_e)
        # d_err_amp, d_err_ph = toPolar(z_eReIm)

        data['z'] = np.array([data['amp'], data['pha']])
        data['z_err'] = np.array([data['amp_err'], data['pha_err']])
        data['N'] = len(data['freq'])
        data['w'] = 2*np.pi*data['freq']

        return data

    def _log_likelihood(self, theta, forward, d_obs, d_obserr):
        """ 
        Returns the conditional log-likelihood of the observations.
        theta : The parameter values/ theta (:obj:`ndarray`): Ordered array of log_R0, m, log_tau,c
        w : The angular frequency vector (w) which is repeated n times (n=number of data)
            Array of angular frequencies to compute the impedance for (w = 2*pi*f).
        d_bs : Observed measurement or 'Noisified synthetic data', Real and Imaginary parts separated.
        d_obserr : Concatenated Amplitude error and phase error
        """
        # print(theta)
        sigma2 = d_obserr**2
        self.thetaVec.append(theta)
        np.savetxt(f'thetaVec_{ModelName}.csv', self.thetaVec, delimiter=',')

        # fwd = np.log(forward(theta))
        fwd = forward(theta)
        # chiSquare = np.mean((np.log(d_obs) - fwd)**2 / sigma2)
        chiSquare = np.mean((d_obs - fwd)**2 / sigma2)
        self.chiSquareVec.append(chiSquare)
        np.savetxt(f'chiSquareVec_{ModelName}.csv', self.chiSquareVec, delimiter=',')


        # return np.sum(-0.5*(d_obs - fwd)**2 / sigma2 - np.log10(sigma2) - 0.5 - np.pi)
        # return np.sum(-0.5*(d_obs - fwd)**2 / sigma2) - 0.5*np.log(np.prod(sigma2)) - 0.5*np.log(2*np.pi)*(np.sum(fwd)/np.mean(fwd)) #sum/mean =n and n=number of data*number of freq*2
        # return np.sum(-0.5*(d_obs - fwd)**2 / sigma2) - 0.5*np.log(2*np.pi)*(np.sum(fwd)/np.mean(fwd)) #sum/mean =n and n=number of data*number of freq*2
       
        # p1 = -0.5*np.sum((np.log(d_obs) - fwd)**2 / sigma2)
        # p2 = -0.5*np.log(2*np.pi)*(np.sum(fwd)/np.mean(fwd)) # sum/mean =n and n=number of data*number of freq*2
        # p3 = - 0.5*np.log(0.05)*(np.sum(fwd)/np.mean(fwd))
        # p4 = - 0.5*np.log(0.01)*(np.sum(fwd)/np.mean(fwd))
        # p5 = - np.sum(np.log(fwd[0]))
        # p6 = - np.sum(np.log(fwd[1]))
        # return  p1 + p2 + p3 + p4 + p5 +p6
        
        return np.sum(-0.5*(d_obs - fwd)**2 / sigma2) - 0.5*np.sum(np.log(sigma2)) - 0.5*np.log(2*np.pi)*(np.sum(fwd)/np.mean(fwd)) #sum/mean =n and n=number of data*number of freq*2


    def _log_prior(self, theta, bounds):
        """Returns the prior log-probability of the model parameters. """
        if not ((bounds[0] < theta).all() and (theta < bounds[1]).all()):
            return -np.inf
        else:
            return 0.0

    def _log_probability(self, theta, model, bounds, w, d_obs, d_obserr):
        """Returns the Bayes numerator log-probability. """
        lp=self._log_prior(theta, bounds)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self._log_likelihood(theta, model, d_obs, d_obserr)

    def _check_if_fitted(self):
        """Checks if the model has been fitted. """
        if not self.fitted:
            raise AssertionError('Model is not fitted! Fit the model to a '
                                 'dataset before attempting to plot results.')

    def fit(self, p0=None, pool=None, moves=None, backend=None, blobs_dtype=None, seed=42):
        """Samples the posterior distribution to fit the model to the data.

        Args:
            p0(:obj:`ndarray`): Starting parameter values. Should be a 2D
                array with shape (nwalkers, ndim). If None, random values will
                be uniformly drawn from the parameter bounds. Defaults to None.
                ndim : Number of model parameters
            pool (:obj:`pool`, optional): A pool object from the
                Python multiprocessing library. See
                https://emcee.readthedocs.io/en/stable/tutorials/parallel/.
                Defaults to None.
            moves (:obj:`moves`, optional): A `emcee` Moves class (see
                https://emcee.readthedocs.io/en/stable/user/moves/). If None,
                the emcee algorithm `StretchMove` is used. Defaults to None.

        """
        self._p0=p0
        # self._bounds = self.param_bounds
        self.ndim=self.param_bounds.shape[1] # ndim:number of parameters

        if self._p0 is None:
            np.random.seed(42)
            self._p0=np.random.uniform(*self.param_bounds,
                                         (self.nwalkers, self.ndim)) #initialization
        np.savetxt(f'initial_parameter-values_{ModelName}.csv', self._p0, delimiter=',')

        model_args=(self.forward, self.param_bounds, self._data['w'],
                      self._data['z'], self._data['z_err'])

        self._sampler=emcee.EnsembleSampler(self.nwalkers,
                                              self.ndim,
                                              self._log_probability,
                                              args=model_args,
                                              pool=pool,
                                              backend=backend)

        self._sampler.run_mcmc(self._p0, self.nsteps, progress=True)
        self.__fitted=True

    def get_chain(self, **kwargs):
        """Get the stored chain of MCMC samples from a fitted model.

        Keyword Args:
            discard (:obj:`int`): Number of steps to discard (burn-in period).
            thin (:obj:`int`): Thinning factor.
            flat (:obj:`bool`): .Flatten the chain across the ensemble. If flat
                is False, the output chain will have shape (nsteps, nwalkers,
                ndim). If flat is True, the output chain will have shape
                (nsteps*nwalkers, ndim).

        Returns:
            :obj:`ndarray`: The MCMC chain(s).

        """
        self._check_if_fitted()
        return self._sampler.get_chain(**kwargs)

    def get_last_sample(self, **kwargs):
        """Access the most recent sample in the chain """
        self._check_if_fitted()
        return self.sampler.get_last_sample(**kwargs)

    def get_log_prob(self, **kwargs):
        """ Get the chain of log probabilities evaluated at the MCMC samples

        Keyword Args:
        flat (Optional[bool]) – Flatten the chain across the ensemble. (default: False)
        thin (Optional[int]) – Take only every thin steps from the chain. (default: 1)
        discard (Optional[int]) – Discard the first discard steps in the chain as burn-in. (default: 0)

        Returns:
        :obj:`ndarray`: The chain of log probabilities.
        """
        self._check_if_fitted()
        return self.sampler.get_log_prob(**kwargs)


    @ property
    def p0(self):
        """:obj:`ndarray`: Starting parameter values. Should be a 2D array with
            shape (nwalkers, ndim)."""
        return self._p0

    @ property
    def params(self): #parameters are defined withe their bounds based on updates
        """:obj:`dict`: Parameter names and their bounds."""
        return self._params

    @ params.setter
    def params(self, var):
        self._params=var

    @ property
    def sampler(self):
        """:obj:`EnsembleSampler`: A `emcee` sampler object (see
            https://emcee.readthedocs.io/en/stable/user/sampler/)."""
        self._check_if_fitted()
        return self._sampler

    @ property
    def data(self):
        """:obj:`dict`: The input data dictionary."""
        return self._data

    @ property
    def fitted(self):
        """:obj:`bool`: Whether the model has been fitted or not."""
        return self.__fitted

    @ property
    def param_names(self):
        """:obj:`list` of :obj:`str`: Ordered names of the parameters."""
        return list(self.params.keys())

    @ property
    def param_bounds(self): # parameter bounds are defined based on the updates
        """:obj:`list` of :obj:`float`: Ordered bounds of the parameters."""
        return np.array(list(self.params.values())).T
    
    # def parse_chain(self, chain, **kwargs):
    #     if chain is None:
    #         # if discard is not None and thin is not None:
    #         kwargs['flat'] = True
    #         chain = self.get_chain(**kwargs)
    #         if 'discard' not in kwargs and 'thin' not in kwargs:
    #             warnings.warn(('No samples were discarded from the chain.\n'
    #                            'Pass discard and thin keywords to remove '
    #                            'burn-in samples and reduce autocorrelation.'),
    #                           UserWarning)
    #     else:
    #         if chain.ndim > 2:
    #             raise ValueError('Flatten chain by passing flat=True.')

    #         if 'discard' in kwargs or 'thin' in kwargs:
    #             raise ValueError('Please pass either a chain obtained with '
    #                              'the get_chain() method or pass '
    #                              'discard and thin keywords to parse '
    #                              'the full chain. Do not pass both.')
    #     return chain
    
    # def get_param_mean(self, chain=None, **kwargs):
    #     """Gets the mean of the model parameters for a MCMC chain.

    #     Args:
    #         chain (:obj:`ndarray`): A numpy array containing the MCMC chain to
    #             plot. Should be a 2D array (nsteps, ndim). If None and no
    #             kwargs are passed to discard iterations, will raise a warning
    #             and the full chain will be used. Defaults to None.

    #     Keyword Args:
    #         **kwargs: See kwargs of the get_chain method.

    #     """
    #     chain = self.parse_chain(chain, **kwargs)
    #     return np.mean(chain, axis=0)
    
    # def eget_param_std(self, chain=None, **kwargs):
    #     """Gets the standard deviation of the model parameters.

    #     Args:
    #         chain (:obj:`ndarray`): A numpy array containing the MCMC chain to
    #             plot. Should be a 2D array (nsteps, ndim). If None and no
    #             kwargs are passed to discard iterations, will raise a warning
    #             and the full chain will be used. Defaults to None.

    #     Keyword Args:
    #         **kwargs: See kwargs of the get_chain method.

    #     """
    #     chain = self.parse_chain(chain, **kwargs)
    #     return np.std(chain, axis=0)