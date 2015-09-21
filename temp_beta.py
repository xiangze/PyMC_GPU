# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 02:23:49 2015

@author: xiangze
"""

import theano
from theano import shared
import theano.tensor as T
import numpy as np
import theano_common as thc
theano.config.exception_verbosity="high"

shared_list=lambda a: shared(np.array(a).T.astype(theano.config.floatX))

batchsize=6
dim=3

Es=shared_list(range(batchsize))
betas=T.vector("betas")

#def MH(E, En, beta,rng):
#    return (T.exp(beta*(E - En)) - rng.uniform(size=E.shape)) >= 0

ff=lambda beta:beta*Es
    
def genr(f,n_steps):
    Esn,u1=thc.recursion(f,[Es],n_steps)
    return thc.func_withupdate([betas],[Es],{Es:Esn[-1]})    

f=genr(ff,4)

print Es.get_value()
print f(range(batchsize))
print Es.get_value()