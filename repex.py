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
    
    def __init__(self,logp,dim=3, batchsize=6,  seed=120):
        self.dim=dim
        self.batchsize=batchsize
        
        self.xs=shared_list([range(batchsize)]*dim)
        self.Es=shared_list(range(batchsize))
        self.logP=logp
            
        self.initial_stepsize=0.01
        self.stepsize = sharedX(self.initial_stepsize, 'hmc_stepsize')

        self.betas=T.vector("betas")
        self.ii=T.ivector("ii")
        self.jj=T.ivector("jj")
        
        self.rng = T.shared_randomstreams.RandomStreams(seed)
        
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

    def _genr(self,n_steps,orgf=HMC.run):    
        def f(xs,Es):
            return orgf(xs,self.rng,
                           self.logP,
                           self.betas,
                           self.stepsize,
                           n_steps)
        xsn,Esn=f(self.xs,self.Es)
        return thc.func_withupdate([self.betas],[xsn,Esn],{self.xs:xsn,self.Es:Esn})  
        
    def genr(self,n_steps,orgf=HMC.run):    
        def f(xs,Es):
            return orgf(xs,self.rng,
                           self.logP,
                           self.betas,
                           self.stepsize,
                           1)

        [xsn,Esn],u=thc.recursion(f,[self.xs,self.Es],n_steps)
        u.update({self.xs:xsn[-1],self.Es:Esn[-1]})
        return thc.func_withupdate([self.betas],[xsn,Esn],u)  

    def genr_adp(self,n_steps,orgf=HMC.adaptiverun):        
        def f(xs,Es,betas):
            return orgf(xs,self.rng,
                self.logP,
                betas,
                self.stepsize,
                n_steps,
                self.avg_acceptance_rate,
                self.step_min,
                self.step_max,
                self.step_inc,
                self.step_dec,
                self.target_acceptance_rate,
                self.avg_acceptance_slowness)
        [xsn,Esn],u1=thc.recursion(f,[self.xs,self.Es],n_steps)
        u1.update({self.xs:xsn[-1],self.Es:Esn[-1]})
        return thc.func_withupdate([self.betas],[xsn,Esn],u1)  
#        return thc.func_withupdate([self.betas],[xsn,Esn],{self.xs:xsn[-1],self.Es:Esn[-1]})  
        
    def gen(self,n_steps,adaptation=False):
        self.fe=self.gene()
        self.fo=self.geno()
        if(adaptation):
            self.fr=self.genr_adp(n_steps)
        else:
            self.fr=self.genr(n_steps)
 
    def run(self,ii,jj,betas,exchangenum=10):
        xss=[]
        Ess=[]
        for i in xrange(exchangenum):
            self.fe(betas,ii)
            xsn,Esn=self.fr(betas)
            xss.append(xsn)
            Ess.append(Esn)
            self.fo(betas,jj)
            xsn,Esn=self.fr(betas)
            xss.append(xsn)
            Ess.append(Esn)
        return xss,Ess
    
    def sample(self,ii,jj,betas,burnin,n_samples):
        [self.run(ii,jj,betas) for r in xrange(burnin)]
        _samples=[self.run(ii,jj,betas)[0] for r in xrange(n_samples)]
        xss,Ess=zip(*_samples)
        xss=np.asarray(xss)
        print xss.shape
#        Ess=np.asarray(Ess)
#        print Ess.shape
        return xss.T.reshape(self.dim, -1).T
                
    def showsteps(self):
        print 'final stepsize', self.stepsize.get_value()
        print 'final acceptance_rate', self.avg_acceptance_rate.get_value()

                       
    def test(self,exchange_steps=100):
        vbetas=range(1,self.batchsize+1)
        ii=range(0,self.batchsize-1,2)
        jj=range(0,self.batchsize-2,2)        
        self.gen(1000)
        return self.sample(ii,jj,vbetas,exchange_steps/10,exchange_steps)
        
if __name__ == "__main__":
    dim=3
    batchsize=6
    num_steps=100
    
    from scipy import linalg

    mu0  = np.array(rng.rand(dim) * 10, dtype=theano.config.floatX)
    mu1  = np.array(rng.rand(dim) * 10, dtype=theano.config.floatX)

    cov = np.array(rng.rand(dim, dim), dtype=theano.config.floatX)
    cov = (cov + cov.T) / 2.
    cov[np.arange(dim), np.arange(dim)] = 1.0
    cov_inv = linalg.inv(cov)

    _gaussian=lambda x,mu,cov_inv:(T.dot((x - mu), cov_inv) * (x - mu)).sum(axis=1)/2
    mixgaussianfunc=lambda x: _gaussian(x,mu0,cov_inv)+_gaussian(x,mu1,cov_inv)

    vbetas=range(1,batchsize+1)
    ii=range(0,batchsize-1,2)
    jj=range(0,batchsize-2,2)
    
    r=Rep(mixgaussianfunc,dim,batchsize,num_steps)
    r.gen(100)
    samples=r.sample(ii,jj,vbetas,num_steps/10,num_steps)
            
    print 'target mean:', mu0,mu1
    print 'target cov:\n', cov

    print 'empirical mean: ', samples.mean(axis=0)
    print 'empirical_cov:\n', np.cov(samples.T)


