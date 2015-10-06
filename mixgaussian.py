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
    return lambda x: -T.log(T.exp(-g0(x))+T.exp(-g1(x)))
           
           
if __name__ == "__main__":    
    dim=2
    #x=T.vector("x")#NG
    x=T.matrix("x")
    fm=mixgaussian(dim,size=1,rand=False)
#    f1=gaussian(dim,size=1,rand=False)
    ff=theano.function([x],fm(x),
                       on_unused_input='warn',
                       allow_input_downcast=True)
    rng = np.random.RandomState(12)
    for i in xrange(100):
        a=np.array([rng.rand(dim)])
        print a[:],ff(a)[0]
