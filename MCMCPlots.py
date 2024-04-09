#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 10:49:35 2022

@author: sadegh
"""
#%%
import emcee
from peltonfdip_2D2Anoms import fdipPeltonNew
import pygimli as pg
import pybert as pb
import matplotlib.pyplot as plt
import numpy as np

# %% loading the results
Modelname = '2D2Anom_wa'

filepath = f'SIPDataAll_{Modelname}.csv'
filename = f"1000_{Modelname}.h5"
mesh = pg.load(f"mesh_{Modelname}.bms")
fdip = pb.FDIP(f"synthSlag_{Modelname}")


# %% loading MCMC chains and plot traces
synModel = np.loadtxt(f'synModel_{Modelname}.csv', delimiter=',')
reader = emcee.backends.HDFBackend(filename)
samples = reader.get_chain(discard=0)
SIPmodel = fdipPeltonNew(filepath=filepath, mesh=mesh, frvec=fdip.freq,
                           data=fdip.data,
                           n_cells=3, nsteps=1000, nwalkers=32)  # initializing the class
    
labels = SIPmodel.param_names
fig, axes = plt.subplots(len(samples[0][0]), figsize=(8, 15), sharex=True, dpi=200)

for i in range(len(samples[0][0])):
    ax = axes[i]
    ax.plot(samples[:, :, i], c='k', alpha=0.3)
    for j in synModel:
        ax.axhline(y=synModel[i], linestyle='--', c='r')
    ax.set_xlim(0, len(samples))
    ax.set_ylim(SIPmodel.param_bounds[:, i])
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)
axes[-1].set_xlabel('Steps')
axes[-12].set_title(f'Original MCMC chains for a {Modelname}')
ax.figure.savefig(f"MCMC-chains-{Modelname}.png", dpi=300)
sd
# %%
# import matplotlib.pyplot as plt

# chi = np.loadtxt('chiSquareVec_2D2AnomsModel8.csv')
# plt.hist(chi, bins=10)
# plt.xscale('log')
# plt.show()
# %% the original chain of log_probabilities. (data fit)
if 1:
    disc = 00
    logProb_samples=-reader.get_log_prob(discard=disc, thin=1) # negative data fit
    # print(samples_logProb)
    print(logProb_samples.shape)
    np.savetxt(f'log_probabilities_{Modelname}.csv', logProb_samples, delimiter=',')
    
    # reader2=emcee.backends.Backend(filename)
    # reader2.get_blobs(flat=True)
    # logPrior_samples=reader2.get_blobs(discard=disc, thin=1) # negative data fit
    # print(logPrior_samples)
    
    
    fig = plt.figure(dpi=100, figsize=(10, 7))
    ax = fig.add_subplot()
    fig.subplots_adjust(top=0.85)
    ax.set_title(f'Discard first {disc} steps of the original log_prob,{Modelname}', fontsize=14, fontweight='bold')
    ax.semilogy(logProb_samples) # log scale
    # ax.set_ylim([0, 60e3])
    plt.ylabel('log_probability', fontsize=12)
    plt.xlabel('Steps', fontsize=12)
    plt.grid(True)
    ax.figure.savefig(f"log_probability_{Modelname}.png", dpi=300)



# %% discarding the bad data fit
lastones = logProb_samples[-1, :]
bestfit_log_prob =lastones[lastones < 1e3]  # 30% more than the min
print(bestfit_log_prob.shape)

# %% the modified chain of log_probabilities. (data fit) bad data fit are deleted
# bestdatafit = logProb_samples[-1, lastones < 0, :]
bestdatafit = logProb_samples[:, lastones < 1e3]

fig = plt.figure(dpi=100, figsize=(10, 7))
ax = fig.add_subplot()
fig.subplots_adjust(top=0.85)
ax.set_title(f'Discard first {disc} steps of the modified log_prob, models with good data fit', fontsize=14, fontweight='bold')
ax.semilogy(bestdatafit) # log scale
# ax.set_ylim([0, 60e3])
plt.ylabel('log_probability', fontsize=12)
plt.xlabel('Steps', fontsize=12)
plt.grid(True)
ax.figure.savefig(f"log_probability_modified_{Modelname}.png", dpi=300)

# %% Histogram
_ = plt.hist(bestdatafit[-1, :], bins=10)
plt.show()
ax.figure.savefig(f"Histogram_modified_{Modelname}.png", dpi=300)


# %% loading the modified MCMC chains and plot traces_ using best fitted data
lastones = logProb_samples[-1, :]
bestsamples = samples[:, lastones < 1e3, :]
abels = SIPmodel.param_names
fig, axes = plt.subplots(len(bestsamples[0][0]), figsize=(8, 15), sharex=True, dpi=200)

for i in range(len(bestsamples[0][0])):
    ax = axes[i]
    ax.plot(bestsamples[:, :, i], c='k', alpha=0.3)
    for j in synModel:
        ax.axhline(y=synModel[i], linestyle='--', c='r')
    ax.set_xlim(0, len(bestsamples))
    ax.set_ylim(SIPmodel.param_bounds[:, i])
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)
axes[-1].set_xlabel('Steps')
axes[-8].set_title(f'Discard first {disc} steps of the modified chains, {Modelname}', fontsize=12, fontweight='bold')
fig.figure.savefig(f"MCMC-chains_modified_{Modelname}.png", dpi=300)


#%% mean / std / true parameters values
synModel = np.loadtxt(f'synModel_{Modelname}.csv', delimiter=',')
samples_mean_std = reader.get_chain(discard=000, thin=1, flat=True)
values = samples_mean_std.mean(axis=(0))
uncertainties = samples_mean_std.std(axis=(0))

for n, v, u, s in zip(SIPmodel.param_names, values, uncertainties, synModel):
  print(f'{n}:    mean:{v:.5f}   +/-   std:{u:.5f}   /  true_value:{s:.5f}')
#real-r0-value = r0*model.data['norm_factoog_probabilities_1000_steps_NewLikelihood-war']

# %% Print best parameters 'modified parameter chains', Print the mean and std of the parameters values
bestsamples = samples[:, lastones < 1e3] # need to flatten
dis = 800
bestsamplesDiscardN = bestsamples[dis:,:,:]
bestsamplesFlatten = bestsamplesDiscardN.reshape(len(bestsamplesDiscardN[:, 0, 0])*len(bestsamplesDiscardN[0, :, 0]),len(bestsamplesDiscardN[0, 0, :]))# flatting
values = bestsamplesFlatten.mean(axis=(0))
uncertainties = bestsamplesFlatten.std(axis=(0))
for n, v, u, s in zip(SIPmodel.param_names, values, uncertainties, synModel):
  print(f'{n}:    mean:{v:.5f}   +/-   std:{u:.5f}   /  true_value:{s:.5f}')
#real-r0-value = r0*model.data['norm_factor']

#%% Inspecting the posterior
# visualize the posterior distribution of all parameters using a corner plot 
import corner
fig = corner.corner(samples_mean_std, labels=SIPmodel.param_names)
_ = fig.suptitle(f"Discard first {disc} steps, {Modelname}", fontsize=36)
fig.figure.savefig(f"corner plot , {Modelname}.png", dpi=300)


#%% Inspecting the posterior modified
# visualize the posterior distribution of all parameters using a corner plot 
import corner
fig = corner.corner(bestsamplesFlatten, labels=SIPmodel.param_names)
_ = fig.suptitle(f"Discard first {dis} steps,Modified {Modelname}", fontsize=36)
fig.figure.savefig(f"corner plot-modified ,{Modelname}.png", dpi=300)

#%% Plot histograms of the MCMC simulation chains.
labels = SIPmodel.param_names
fig, axes = plt.subplots(len(samples[0][0]), figsize=(5, 1.5*samples_mean_std.shape[1]))
for i in range(len(samples[0][0])):
    ax = axes[i]
    ax.hist(samples_mean_std[:, i], bins=25, fc='w', ec='k')
    ax.set_xlabel(labels[i])
    ax.ticklabel_format(axis='x', scilimits=[-2, 2])
    fig.suptitle('Histograms of the MCMC simulation chains', fontsize=14)
fig.tight_layout()
fig.figure.savefig(f"histograms of the MCMC simulation chains , {Modelname}.png", dpi=300)
#%% Plot histograms of the modified MCMC simulation chains.
labels = SIPmodel.param_names
fig, axes = plt.subplots(len(samples[0][0]), figsize=(5, 1.5*samples_mean_std.shape[1]))
for i in range(len(samples[0][0])):
    ax = axes[i]
    ax.hist(bestsamplesFlatten[:, i], bins=25, fc='w', ec='k')
    ax.set_xlabel(labels[i])
    ax.ticklabel_format(axis='x', scilimits=[-2, 2])
    fig.suptitle('Histograms of the modified MCMC simulation chains', fontsize=14)
fig.tight_layout()
fig.figure.savefig(f"histograms of the modified MCMC simulation chains , {Modelname}.png", dpi=300)

#%% Plot histograms of the MCMC simulation chains. modified
# labels = SIPmodel.param_names
# fig, axes = plt.subplots(len(samples[0][0]), figsize=(5, 1.5*bestsamplesFlatten.shape[1]))
# for i in range(len(samples[0][0])):
#     ax = axes[i]
#     ax.hist(bestsamplesFlatten[:, i], bins=25, fc='w', ec='k')
#     ax.set_xlabel(labels[i])
#     ax.ticklabel_format(axis='x', scilimits=[-2, 2])
# fig.tight_layout()
# %% Section
if 1:
    rho_best, m_best, tau_best, c_best =np.reshape(values, (4, -1))
    fig, ax = pg.plt.subplots(nrows=4, figsize=(8, 12), sharex=True, sharey=True)
    cm = mesh.cellMarkers()
    pg.show(mesh, (10**rho_best)[cm], ax=ax[0], cMin=0, cMax=500, logScale=True,
        cMap="Spectral_r")
    pg.show(mesh, m_best[cm], ax=ax[1], cMin=0, cMax=0.8, logScale=0, cMap="plasma")
    pg.show(mesh, (10**tau_best)[cm], ax=ax[2], cMin=0.01, cMax=1, logScale=1, cMap="magma")
    pg.show(mesh, c_best[cm], ax=ax[3], cMin=0, cMax=0.5, logScale=0, cMap="viridis")
    ax[0].set_xlim(-100, 100)
    ax[0].set_ylim(-70, 0);
    fig.suptitle('Section', fontsize=14)
    fig.figure.savefig(f"Section ,{Modelname}.png", dpi=300)
