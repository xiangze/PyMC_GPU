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

sharedX = lambda X, name: shared(np.asarray(X, dtype=theano.config.floatX), name=name)
shared_list=lambda a: shared(np.array(a).T.astype(theano.config.floatX))

"""
References
basic HMC implementation
http://deeplearning.net/tutorial/hmc.html
http://deeplearning.net/tutorial/code/hmc/hmc.py

"""

class PyMC_GPU(object):
    
    def __init__(self,logp,dim=3, batchsize=6,  seed=120,debug=False):
        assert batchsize>3 and batchsize%2==0 
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
        self.debug=debug
        
        
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

    def gen_eo(self,n_steps,orgf=HMC.run):    
        def f(xs,Es):
            return orgf(xs,self.rng,
                           self.logP,
                           self.betas,
                           self.stepsize,
                           1)
        xout_e,u=exchange.exchange_even(self.xs,self.Es,self.betas,self.ii)                           
        [xsn0,Esn0],u0=thc.recursion(f,[xout_e,self.Es],n_steps)
        u.update(u0)
        xout_o,uo=exchange.exchange_odd(xsn0[-1],Esn0[-1],self.betas,self.jj)
        u.update(uo)
        [xsn1,Esn1],u1=thc.recursion(f,[xout_o,Esn0[-1]],n_steps)
        u.update(u1)
        u.update({self.xs:xsn1[-1],self.Es:Esn1[-1]})
        self.f=thc.func_withupdate([self.betas,self.ii,self.jj],[xsn0,Esn0,xsn1,Esn1],u)  
                         
    def gen(self,n_steps,adaptation=False):
        self.fe=self.gene()
        self.fo=self.geno()
        if(adaptation):
            self.fr=self.genr_adp(n_steps)
        else:
            self.fr=self.genr(n_steps)
 
    def run(self,ii,jj,betas,exchangenum=10):
        xss=np.zeros([1,self.batchsize,self.dim])
        Ess=np.zeros([1,self.batchsize])
        
        for i in xrange(exchangenum):
            self.fe(betas,ii)
            xsn,Esn=self.fr(betas)
            print ".",
            xss=np.r_[xss,xsn]
            Ess=np.r_[Ess,Esn]
            self.fo(betas,jj)
            xsn,Esn=self.fr(betas)
            xss=np.r_[xss,xsn]
            Ess=np.r_[Ess,Esn]
        return xss,Ess
    
    def sample(self,betas,burnin,n_samples):
        ii=range(0,self.batchsize-1,2)
        jj=range(0,self.batchsize-2,2)
        xss=np.zeros([1,self.batchsize,self.dim])
        Ess=np.zeros([1,self.batchsize])
        [self.run(ii,jj,betas) for r in xrange(burnin)]
        for r in xrange(n_samples):
            xsn,Esn=self.run(ii,jj,betas)
            xss=np.r_[xss,xsn]
            Ess=np.r_[Ess,Esn]
        return xss,Ess
#        return xss.T.reshape(self.dim, -1).T

    def sample3(self,betas,burnin,n_samples):
        ii=range(0,self.batchsize-1,2)
        jj=range(0,self.batchsize-2,2)
        xss=np.zeros([1,self.batchsize,self.dim])
        Ess=np.zeros([1,self.batchsize])
        [self.f(ii,jj,betas) for r in xrange(burnin)]
        for r in xrange(n_samples):
            xsn0,Esn0,xsn1,Esn1=self.run(betas,ii,jj)
            xss=np.r_[xss,xsn0]
            xss=np.r_[xss,xsn1]
            Ess=np.r_[Ess,Esn0]
            Ess=np.r_[Ess,Esn1]
        return xss,Ess
        
    def showsteps(self):
        print 'final stepsize', self.stepsize.get_value()
        print 'final acceptance_rate', self.avg_acceptance_rate.get_value()

                       
    def test(self,num_step=1000,exchange_steps=10):
        vbetas=range(1,self.batchsize+1)
        ii=range(0,self.batchsize-1,2)
        jj=range(0,self.batchsize-2,2)        
        self.gen(num_step)
        return self.sample(ii,jj,vbetas,exchange_steps/10,exchange_steps)

    def test3(self,num_step=1000,exchange_steps=10):
        vbetas=range(1,self.batchsize+1)
        ii=range(0,self.batchsize-1,2)
        jj=range(0,self.batchsize-2,2)        
        self.gen_eo(num_step)
        return self.sample(ii,jj,vbetas,exchange_steps/10,exchange_steps)
        
if __name__ == "__main__":
    seed=121
    
    dim=2
    batchsize=4
    num_onestep=1000
    num_exchange=1
    import mixgaussian as mg

    vbetas=range(1,batchsize+1)
#    func=mg.mixgaussian(dim,size=2,seed=123,rand=True,mu0=2,mu1=-2)
#    func=mg.gaussian(dim,size=2,rand=False,mu=2)
    mu=2
    rng = np.random.RandomState(seed)
    func=lambda x:(T.dot((x - mu), mg.covinv(dim,rng)) * (x - mu)).sum(axis=1)/2
   
    r=Rep(func,dim,batchsize,num_onestep)
    #r.gen_eo(num_onestep)
    r.gen(num_onestep)
    samples=r.sample(vbetas,num_exchange/10,num_exchange)
    samplex=samples[0]
    print samplex.shape
    print samplex[:10]
    print "last"
    print samplex[-10:]
    np.savetxt("%d_%d_%d.csv"%(batchsize,num_onestep,num_exchange),samplex)
    print 'target mean:', mu
#    print 'target cov:\n', cov
    
    print 'empirical mean: ', samplex.mean(axis=0)
    print 'empirical_cov:\n', np.cov(samplex.T)


