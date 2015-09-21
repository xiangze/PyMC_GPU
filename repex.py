# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 16:30:02 2015

@author: xiangze
"""

import HMC
import theano.tensor as T
import theano
import exchange
import theano_common as thc
import numpy as np
from theano import shared

theano.config.exception_verbosity="high"
seed=121
rng = np.random.RandomState(seed)

sharedX = lambda X, name: shared(np.asarray(X, dtype=theano.config.floatX), name=name)
shared_list=lambda a: shared(np.array(a).T.astype(theano.config.floatX))

class Rep(object):
    
    def __init__(self,logp,dim=3, batchsize=6, n_steps=4, seed=120):
        self.dim=dim
        self.batchsize=batchsize
        self.n_steps=n_steps
        
        self.xs=shared_list([range(batchsize)]*dim)
        self.Es=shared_list(range(batchsize))
        self.logP=logp
            
        self.betas=T.vector("betas")
        self.ii=T.ivector("ii")
        self.jj=T.ivector("jj")

    def setAdaptiveStep(self):
        self.initial_stepsize=0.01
        self.target_acceptance_rate=.9
        self.step_dec=0.98
        self.step_min=0.001
        self.step_max=0.25
        self.step_inc=1.02
        self.avg_acceptance_slowness=0.9
                                      
    def gene(self):
        xout,u=exchange.exchange_even(self.xs,self.Es,self.betas,self.ii)
        u.update({self.xs:xout})
        return thc.func_withupdate([self.betas,self.ii],[],u)

    def geno(self):
        xout,u=exchange.exchange_odd(self.xs,self.Es,self.betas,self.jj)
        u.update({self.xs:xout})
        return thc.func_withupdate([self.betas,self.jj],[],u)

    def genr(n_steps,self,orgf=HMC.run):        
        def f(self,xs,Es):
            return orgf(xs,rng,
                           self.logP,
                           self.betas,
                           self.stepsize,
                           self.n_steps)

        [xsn,Esn],u1=thc.recursion(f,[self.xs,self.Es],n_steps)
        return thc.func_withupdate([self.betas],[xsn,Esn],{self.xs:xsn[-1],self.Es:Esn[-1]})  

    def genr_adp(n_steps,self,orgf=HMC.adaptiverun):        
        def f(xs,Es,betas):
            return orgf(xs,rng,
                self.logP,
                betas,
                self.stepsize,
                self.n_steps,
                self.avg_acceptance_rate,
                self.step_min,
                self.step_max,
                self.step_inc,
                self.step_dec,
                self.target_acceptance_rate,
                self.avg_acceptance_slowness)
        [xsn,Esn],u1=thc.recursion(f,[self.xs,self.Es],n_steps)
        return thc.func_withupdate([self.betas],[xsn,Esn],{self.xs:xsn[-1],self.Es:Esn[-1]})  
        
    def gen(self,n_steps,adaptation=False):
        self.fe=self.gene()
        self.fo=self.geno()
        if(adaptation):
            self.fr=self.genr_adp(n_steps)
        else:
            self.fr=self.genr(n_steps)
 
    def run(self,ii,jj,betas,num=1):
        for i in xrange(num):
            self.fe(betas,ii)
            self.fr()
            self.fo(betas,jj)
            self.fr()
    
    def sample(self,ii,jj,betas,burnin):
        [self.run(ii,jj,betas) for r in xrange(burnin)]
        _samples = np.asarray([self.run(ii,jj,betas) for r in xrange(n_samples)])
        return _samples.T.reshape(self.dim, -1).T
    
    def showsteps(self):
        print 'final stepsize', self.stepsize.get_value()
        print 'final acceptance_rate', self.avg_acceptance_rate.get_value()

    def test0(self,n_steps=100):
        self.gen(n_steps)
        vbetas=range(1,self.batchsize+1)
        ii=range(0,self.batchsize-1,2)
        jj=range(0,self.batchsize-2,2)
        print "ii,jj",ii,jj

        print self.xs.get_value()
        print self.Es.get_value()        

        self.fe(vbetas,ii)
        print "x",self.xs.get_value()
        self.fo(vbetas,jj)
        print "x",self.xs.get_value()
                        
    def test(self,n_steps=100):
        pass
    
if __name__ == "__main__":
    dim=3
    batchsize=6
    num_steps=100
    
    from scipy import linalg

    mu  = np.array(rng.rand(dim) * 10, dtype=theano.config.floatX)
    cov = np.array(rng.rand(dim, dim), dtype=theano.config.floatX)
    cov = (cov + cov.T) / 2.
    cov[np.arange(dim), np.arange(dim)] = 1.0
    cov_inv = linalg.inv(cov)

    gaussian_energy=lambda x:(T.dot((x - mu), cov_inv) * (x - mu)).sum(axis=1)/2
    _gaussian=lambda x,mu,cov_inv:(T.dot((x - mu), cov_inv) * (x - mu)).sum(axis=1)/2

    mixgaussianfunc=lambda x,mu,covinv: _gaussian(mu[0],covinv[0])+_gaussian(mu[1],covinv[1])

    r=Rep(mixgaussianfunc,dim,batchsize,num_steps)
    r.gen(1000)
    r.runex()
    samples=r.sample(100)
            
    print 'target mean:', mu
    print 'target cov:\n', cov

    print 'empirical mean: ', samples.mean(axis=0)
    print 'empirical_cov:\n', np.cov(samples.T)


