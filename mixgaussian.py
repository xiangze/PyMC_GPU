# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 03:01:47 2015

@author: xiangze
"""

import numpy as np
from scipy import linalg
import theano
from theano import tensor as T

def covinv(dim,rng):
    cov = np.array(rng.rand(dim, dim), dtype=theano.config.floatX)
    cov = (cov + cov.T) / 2.
    cov[np.arange(dim), np.arange(dim)] = 1.0
    return linalg.inv(cov)
    
def gaussian(dim,size=10,seed=123,rand=True,mu=2):
    rng = np.random.RandomState(seed)
    if(rand):
        mu  = np.array(rng.rand(dim) * size, dtype=theano.config.floatX)
    return lambda x:(T.dot((x - mu), covinv(dim,rng)) * (x - mu)).sum(axis=1)/2

def mixgaussian(dim,size=10,seed=123,rand=True,mu0=2,mu1=-2):
    rng = np.random.RandomState(seed)
    if(rand):
        mu0  = np.array(rng.rand(dim) * size, dtype=theano.config.floatX)
        mu1  = np.array(rng.rand(dim) * size, dtype=theano.config.floatX)
    g0=gaussian(dim,size,seed    ,rand,mu0)
    g1=gaussian(dim,size,seed+dim,rand,mu1)
#    _gaussian=lambda x,mu,cov_inv:(T.dot((x - mu), cov_inv) * (x - mu)).sum(axis=1)/2
    return lambda x: g0(x)+g1(x)
           