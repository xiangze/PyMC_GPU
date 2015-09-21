# -*- coding: utf-8 -*-
"""
Created on Wed Sep 02 23:04:58 2015

@author: xiangze
"""

import theano
from theano import shared
import theano.tensor as T
#from theano import function, shared
import numpy as np

#from scipy import linalg
import exchange

import theano_common as thc
theano.config.exception_verbosity="high"

def shared_list(a):
    return shared(np.array(a).T.astype(theano.config.floatX))

seed=120
dim=3
batchsize=6
n_steps=4

betas=T.vector("betas")
ii=T.ivector("ii")
jj=T.ivector("jj")

xs=shared_list([range(batchsize)]*dim)
Es=shared_list(range(batchsize))
#[1]*batchsize

arun=lambda xs,Es:[xs+xs,Es+2]
run2=lambda xs,Es:[xs+xs,xs[0]]
run3=lambda xs,Es:[xs+xs,xs[0]*betas]

ins=[betas,ii]

def _gene():#compile OK no update
    xout,u=exchange.exchange_even(xs,Es,betas,ii)
    return thc.func_withupdate(ins,[xout],u)

def gene():#OK
    xout,u=exchange.exchange_even(xs,Es,betas,ii)
    u.update({xs:xout})
    return thc.func_withupdate(ins,[],u)

def geno():#OK
    xout,u=exchange.exchange_odd(xs,Es,betas,jj)
    u.update({xs:xout})
    return thc.func_withupdate([betas,jj],[],u)

def genr(f=arun):#OK
    [xsn,Esn],u1=thc.recursion(f,[xs,Es],n_steps)
    return thc.func_withupdate([],[xsn,Esn],{xs:xsn[-1],Es:Esn[-1]})    

def genf0():#OK
    [xsn,Esn],u1=thc.recursion(arun,[xs,Es],n_steps)
    xout_even,u_e=exchange.exchange_even(xsn[-1],Esn[-1],betas,ii)
    return thc.func_withupdate(ins,[xout_even,xsn,Esn], u_e)
    
def genf01():#OK
    xout_even,u_e=exchange.exchange_even(xs,Es,betas,ii)
    [xsn,Esn],u1=thc.recursion(arun,[xout_even,Es],n_steps)
    u_e.update({xs:xsn[-1],Es:Esn[-1]})
    return thc.func_withupdate(ins,[xsn,Esn],u_e)    

def genf001():#OK?
    xout_even,u_e=exchange.exchange_even(xs,Es,betas,ii)
    [xsn,Esn],u1=thc.recursion(arun,[xout_even,Es],n_steps)
    return thc.func_withupdate(ins,[xout_even,xsn,Esn], u_e)

def genf001o():#OK?
    xout_even,u_o=exchange.exchange_odd(xs,Es,betas,ii)
    [xsn,Esn],u1=thc.recursion(arun,[xout_even,Es],n_steps)
    return thc.func_withupdate(ins,[xout_even,xsn,Esn], u_o)
    
def genf1():    
    [xsn,Esn],u1=thc.recursion(arun,[xs,Es],n_steps)
    xout_even,u_e=exchange.exchange_even(xsn[-1],Esn[-1],betas,ii)
    [xsn_eo,Esn_eo],u2=thc.recursion(arun, [xout_even,Esn],n_steps)
    return thc.func_withupdate(ins,[xout_even,xsn,Esn], u2)
    
def genf2():
    [xsn,Esn],u1=thc.recursion(arun,[xs,Es],n_steps)
    xout_even,u_e=exchange.exchange_even(xsn[-1],Esn[-1],betas,ii)
    [xsn_eo,Esn_eo],u2=thc.recursion(arun, [xout_even,Es],n_steps)
    xout_eo,u_eo=exchange.exchange_odd(xsn_eo[-1],Esn_eo[-1],betas,ii)    

    return thc.func_withupdate(ins,[xout_eo,xsn_eo,Esn_eo], u_eo)

def genf002():
    xout_even,u_e=exchange.exchange_even(xs,Es,betas,ii)
    [xsn,Esn],u1=thc.recursion(arun,[xout_even,Es],n_steps)
    xout_eo,u_eo=exchange.exchange_odd(xsn[-1],Esn[-1],betas,jj)    
    [xsn_eo,Esn_eo],u2=thc.recursion(arun, [xout_eo,Esn[-1]],n_steps)
    return thc.func_withupdate(ins,[xout_eo,xsn_eo,Esn_eo], u_e)

def test0():
    fe=gene()
    fo=geno()
    vbetas=range(1,batchsize+1)
    ii=range(0,batchsize-1,2)
    jj=range(0,batchsize-2,2)
    print "ii,jj",ii,jj

    print xs.get_value()
    print Es.get_value()        

    fe(vbetas,ii)
    print "x",xs.get_value()
    fo(vbetas,jj)
    print "x",xs.get_value()

def test2():
    fe=gene()
    fo=geno()
    fr=genr(run2)
    vbetas=range(1,batchsize+1)
    ii=range(0,batchsize-1,2)
    jj=range(0,batchsize-2,2)
    print "ii,jj",ii,jj

    print xs.get_value()
    print Es.get_value()        

    fe(vbetas,ii)
    print "x",xs.get_value()
    fr()
    print "x",xs.get_value()
    fo(vbetas,jj)
    print "x",xs.get_value()
    
def test3():
    fe=gene()
    fo=geno()
    fr=genr(run3)

    vbetas=range(1,batchsize+1)
    ii=range(0,batchsize-1,2)
    jj=range(0,batchsize-2,2)
    print "ii,jj",ii,jj

    print xs.get_value()
    print Es.get_value()        

    fe(vbetas,ii)
    print "x",xs.get_value()
    fo(vbetas,jj)
    print "x",xs.get_value()
    
def test():
    fe=gene()
    fo=geno()
    fr=genr()
 
    def runex(ii,jj,betas,num=1):
        for i in xrange(num):
            fe(betas,ii)
            fr()
            fo(betas,jj)
            fr()
        return fr()
   
    vbetas=range(1,batchsize+1)
    ii=range(0,batchsize-1,2)
    jj=range(1,batchsize,2)
    print "ii,jj",ii,jj

    print xs.get_value()
    print Es.get_value()
    xso,Eso=runex(ii,jj,vbetas)
    runex(ii,jj,vbetas)    
#    print "xo",xo
#    print "xs,Es chain", xso,Eso

if __name__ == "__main__":
    #f=genf001()#OK?
    #f=genf001o()
    #f=genf0()
    #f=genf01()
    #f=genf1()

#    test0()
    test2()
