# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 16:41:09 2021

@author: dingxu
"""

import numpy as np
import emcee
import matplotlib.pyplot as pl
from matplotlib.pyplot import cm 
import matplotlib.mlab as mlab
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

def q(fbg,f0, rc, x):
    return fbg+f0/(1+(x/rc)**2)

def mobandata(Age,E,mM):
    agedata = np.loadtxt('agefeh09.dat')
    ageGBPRP = agedata[:,[2,28,29,30]]
    selectdata = ageGBPRP[np.round(ageGBPRP[:,0],2)== np.round(Age,2)]
    selectG = selectdata[:,1]
    selectBPRP = selectdata[:,2]-selectdata[:,3]
    print(str(len(selectBPRP))+'it is ok!')
    #print(selectG,selectBPRP)
    return selectG,selectBPRP

def distancecompute(selectG, selectBPRP,ydata):
    yuanBPRPG = [(ydata[1,:], ydata[0,:])]
    npyBPRPG = np.array(yuanBPRPG)[0].T

    mBPRPG = [(selectBPRP,selectG)]
    nmBPRPG = np.array(mBPRPG)[0].T
    
    #print (npyBPRPG)
    #print (nmBPRPG)
    
    kdt = cKDTree(nmBPRPG)#nmBPRPG[:,1:]
    dist, indices = kdt.query(npyBPRPG)
    
    temp = []
    for i in range (len(indices)):
        index = indices[i] 
        temp.append(nmBPRPG[index])
    
    nptemp = np.array(temp)    

    pipeidata = np.column_stack((npyBPRPG, nptemp))
    d1 = (pipeidata[:,0]-pipeidata[:,2])**2
    d2 = (pipeidata[:,1]-pipeidata[:,3])**2
    d = np.sqrt((d1+d2)/0.01)
    distance = np.sum(d)
    return distance


ydata = np.loadtxt('BPRPG.txt')

sigma = 0.05

nwalkers = 30
niter = 250
init_dist = [(7.,10.),(0.5,1),(10,20)]
ndim = len(init_dist)
priors = init_dist

def rpars(init_dist):
    return [np.random.rand() * (i[1]-i[0]) + i[0] for i in init_dist]

def lnprior(priors, values):
    
    lp = 0.
    for value, prior in zip(values, priors):
        if value >= prior[0] and value <= prior[1]:
            lp+=0
        else:
            lp+=-np.inf 
    return lp

def lnprob(z):
    
    lnp = lnprior(priors,z)
    if not np.isfinite(lnp):
            return -np.inf
  
    # make a model using the values the sampler generated
    selectG,selectBPRP = mobandata(z[0],z[1],z[2])

    # use chi^2 to compare the model to the data:
    chi2 = distancecompute(selectG, selectBPRP,ydata)

    # calculate lnp
    lnprob = -0.5*chi2 + lnp

    return lnprob

tempsigma = []
def run(init_dist, nwalkers, niter, ndim):

    # Generate initial guesses for all parameters for all chains
    p0 = np.array([rpars(init_dist) for i in range(nwalkers)])
    #print(p0)

    # Generate the emcee sampler. Here the inputs provided include the 
    # lnprob function. With this setup, the first parameter
    # in the lnprob function is the output from the sampler (the paramter 
    # positions).
    sampler = emcee.EnsembleSampler(nwalkers,ndim,lnprob)

    pos, prob, state = sampler.run_mcmc(p0, niter)

    for i in range(ndim):
        pl.figure(i+1)
        y = sampler.flatchain[:,i]
        n, bins, patches = pl.hist(y, 200, normed=1, color="b", alpha=0.45)
        pl.title("Dimension {0:d}".format(i))
        
        mu = np.average(y)
        tempsigma.append(mu)
        sigma = np.std(y)  
        tempsigma.append(sigma)
        print ("mu,", "sigma = ", mu, sigma)

        bf = norm.pdf(bins, mu, sigma)
        l = pl.plot(bins, bf, 'k--', linewidth=2.0)
        
    pl.show()
    return pos,tempsigma


pos,tempsigma = run(init_dist, nwalkers, niter, ndim)



