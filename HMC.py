# -*- coding: utf-8 -*-
"""
Created on Sat Sep 05 09:04:22 2015

@author: xiangze
"""

import numpy
from theano import function, shared
from theano import tensor as T
import theano
import numpy as np


sharedX = lambda X, name: shared(numpy.asarray(X, dtype=theano.config.floatX), name=name)
shared_list=lambda a: shared(np.array(a).T.astype(theano.config.floatX))


def kinetic_energy(v):
    return 0.5 * (v ** 2).sum(axis=1)

def hamiltonian(x, v, logP):
    # assuming mass = 1
    return logP(x) + kinetic_energy(v)
    
def MHorg(E, En, rng):
    return (T.exp(E - En) - rng.uniform(size=E.shape)) >= 0

def MH(E, En, beta,rng):
    return (T.exp(beta*(E - En)) - rng.uniform(size=E.shape)) >= 0

def gradsum(E,x):
    return T.grad(E(x).sum(),x)
    
def dynamics(x0, v0, stepsize, n_steps, logP):
    def leapfrog(x, v, step):
        vn=v-step*gradsum(logP,x)
        xn=x+vn*step
        return [xn,vn], {}

    v_half=v0-stepsize/2*gradsum(logP,x0)
    xn=x0+stepsize*v_half

    (xs, vs), scan_updates = theano.scan(leapfrog,
            outputs_info=[
                dict(initial=xn),
                dict(initial=v_half),
                ],
            non_sequences=[stepsize],
            n_steps=n_steps - 1)

    final_x,final_v = xs[-1],vs[-1]

    assert not scan_updates
    E=logP(final_x)
    final_v = final_v - stepsize/2 * T.grad(E.sum(), final_x)

    return final_x, final_v
    
def hmc_move(rng, x0, logP, beta, stepsize, n_steps):
    v0 = rng.normal(size=x0.shape)
    
    xn,vn = dynamics(
            x0=x0,  v0=v0,
            stepsize=stepsize,  n_steps=n_steps,
            logP=logP)

    E =hamiltonian(x0, v0, logP)
    En=hamiltonian(xn, vn, logP)
    accept = MH(E,En,beta,rng=rng)
    
    return accept,xn,T.switch(accept,En,E)
        
def hmc_updates(x, stepsize, avg_acceptance_rate, final_x, accept,
                 target_acceptance_rate, stepsize_inc, stepsize_dec,
                 stepsize_min, stepsize_max, avg_acceptance_slowness):

    ## POSITION UPDATES ##
    accept_matrix = accept.dimshuffle(0, *(('x',) * (final_x.ndim - 1)))
    xn = T.switch(accept_matrix, final_x, x)

    ## STEPSIZE UPDATES ##
    _new_stepsize = T.switch(avg_acceptance_rate > target_acceptance_rate,
                              stepsize * stepsize_inc, stepsize * stepsize_dec)
    new_stepsize  = T.clip(_new_stepsize, stepsize_min, stepsize_max)

    ## ACCEPT RATE U+PDATES ##
    mean_dtype = theano.scalar.upcast(accept.dtype, avg_acceptance_rate.dtype)
    new_acceptance_rate = T.add(
            avg_acceptance_slowness * avg_acceptance_rate,
            (1.0 - avg_acceptance_slowness) * accept.mean(dtype=mean_dtype))

    return [(x, xn),
            (stepsize, new_stepsize),
            (avg_acceptance_rate, new_acceptance_rate)]
    
def run(xs,rng,
        logP,
        beta,
        stepsize,
        n_steps):

       accept, xn, En= hmc_move(
                rng,xs,
                logP,beta,
                stepsize,n_steps)
       accept_matrix = accept.dimshuffle(0, *(('x',) * (xn.ndim - 1)))
       xn = T.switch(accept_matrix, xn, xs)
       return xn,En
       
def adaptiverun(xs_shared,rng,
                logP,
                beta,
                stepsize,
                n_steps,
                avg_acceptance_rate,
                step_min,
                step_max,
                step_inc,
                step_dec,
                target_acceptance_rate,
                avg_acceptance_slowness):

       accept, final_pos, finalE= hmc_move(
                rng,
                xs_shared,
                logP,
                beta,
                stepsize,
                n_steps)
    
       xs,stepsizes,acceptrates = hmc_updates(
                xs_shared,
                stepsize,
                avg_acceptance_rate,
                final_x=final_pos,
                accept=accept,
                stepsize_min=step_min,
                stepsize_max=step_max,
                stepsize_inc=step_inc,
                stepsize_dec=step_dec,
                target_acceptance_rate=target_acceptance_rate,
                avg_acceptance_slowness=avg_acceptance_slowness)
 
       return finalE,xs,stepsizes,acceptrates




    
if __name__ == "__main__":
    import theano_common as thc
    from scipy import linalg
    seed=120
    dim=3
    batchsize=6
    n_steps=4    
    xs=shared_list([range(batchsize)]*dim)
    Es=shared_list(range(batchsize))

    rng = np.random.RandomState(seed)
    mu0  = np.array(rng.rand(dim) * 10, dtype=theano.config.floatX)
    mu1  = np.array(rng.rand(dim) * 10, dtype=theano.config.floatX)

    cov = np.array(rng.rand(dim, dim), dtype=theano.config.floatX)
    cov = (cov + cov.T) / 2.
    cov[np.arange(dim), np.arange(dim)] = 1.0
    cov_inv = linalg.inv(cov)

    _gaussian=lambda x,mu,cov_inv:(T.dot((x - mu), cov_inv) * (x - mu)).sum(axis=1)/2
    mixgaussianfunc=lambda x: _gaussian(x,mu0,cov_inv)+_gaussian(x,mu1,cov_inv)

    stepsize=0.01
    betas=T.vector("betas")
    srng = T.shared_randomstreams.RandomStreams(seed)    

    def f(xs,Es):
        return run(xs,srng,
                   mixgaussianfunc,
                   betas,
                   stepsize,
                   n_steps)

#    [xsn,Esn],u1=thc.recursion(f,[xs,Es],n_steps)
#    ff=thc.func_withupdate([betas],[xsn,Esn],{xs:xsn[-1],Es:Esn[-1]})  
    xsn,Esn=f(xs,Es)
    ff=thc.func_withupdate([betas],[xsn,Esn],{xs:xsn,Es:Esn})  
