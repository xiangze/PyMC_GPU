# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 03:23:01 2015

@author: xiangze
"""

import theano
import theano.tensor as T
import numpy as np

seed=12
rng = T.shared_randomstreams.RandomStreams(seed)

def reshape3_1(a):
    s=T.shape(a)
    return T.reshape(a,(s[0]*s[1],s[2]))
    
def ex0(beta0,beta1,E0,E1):
   return (T.exp((beta0-beta1)*(E0 - E1)) - rng.uniform(size=E0.shape)) >= 0

def ex(i,j,betas,Es):
    return ex0(betas[i],betas[j],Es[i],Es[j])
    
def exf(i,j,betas,Es,xs):
    return T.switch(ex(i,j,betas,Es),xs[i],xs[j])

def exchange2(xs,Es,betas,ii,jj):
    x,updates=theano.map(exf,[ii,jj],non_sequences=[betas,Es,xs])
    return x

def _ex_even(i,betas,Es,xs):
    return T.switch(ex(i,i+1,betas,Es),[xs[i+1],xs[i]],[xs[i],xs[i+1]])

def _ex_odd(i,betas,Es,xs):
    return T.switch(ex(i-1,i,betas,Es),[xs[i],xs[i-1]],[xs[i-1],xs[i]])
   
def exchange_even(xs,Es,betas,ii):
    x,updates=theano.map(_ex_even,[ii],non_sequences=[betas,Es,xs])
    return reshape3_1(x),updates

#def exchange_odd(xs,Es,betas,ii):
#    x,updates=theano.map(_ex_odd,[ii],non_sequences=[betas,Es,xs])
#    return reshape3_1(x),updates

def exchange_odd(xs,Es,betas,ii):    
    n=T.shape(xs)[0]
    x,updates=theano.map(_ex_even,[ii],non_sequences=[betas[1:n-1],Es[1:n-1],xs[1:n-1]])
    return T.set_subtensor(xs[1:n-1], reshape3_1(x)),updates
   
def exchange(xs,Es,betas,ii,parity):
    if(parity):
        x,updates=theano.map(_ex_even,[ii],non_sequences=[betas,Es,xs])
    else:
        x,updates=theano.map(_ex_odd ,[ii],non_sequences=[betas,Es,xs])

    return reshape3_1(x),updates

def _exchange(parity,xs,Es,betas,ii):
    return exchange(xs,Es,betas,ii,parity)
    
def exchange_run(xs,Es,betas,ii,ps):
    def _exchange1(xs,Es,betas,ii,parity):
        xn,u=_exchange(xs,Es,betas,ii,parity)
        Es=Es+1
        return xn,Es
    res,updates=theano.scan(_exchange1,[ps],outputs_info=[xs,Es],non_sequences=[betas,ii])
    return res,updates
    
def exchange_rep(xs,Es,betas,ii,ps):
#    res,updates=theano.map(_exchange,non_sequences=[xs,Es,betas,ii])
    res,updates=theano.scan(_exchange,[ps],outputs_info=[xs],non_sequences=[Es,betas,ii])
    return res,updates

#cannot compile
def exchange_rep2(xs,Es,betas,ii,ps):
    res,updates=theano.scan(_exchange,[ps],outputs_info=[xs,Es],non_sequences=[betas,ii])
    return res,updates
    

    
if __name__ == "__main__":
    test=1
    repnum=8    
    num=10
    dim=2
    xs=T.matrix("x")
    Es=T.vector("E")
    betas=T.vector("betas")
    ii=T.ivector("ii")
    
    
    if(test==0):
        parity=T.iscalar("parity")
        res,u=exchange(xs,Es,betas,ii,parity)
        fex=theano.function([xs,Es,betas,ii,parity],
                            res, updates=u,
                            on_unused_input='warn',
                            allow_input_downcast=True)

        x=np.random.uniform(size=repnum*dim).reshape(repnum,dim)
        E=np.random.uniform(size=repnum)
        beta= range(repnum)#np.array(range(num),dtype=np.float32)
        i=range(0,repnum,2)#irange
    
        print "x",x
        print "E",E
        print "result",fex(x,E,beta,i,0)
        
    else:
        import random
        ps=T.ivector("parities")
        res,u=exchange_run(xs,Es,betas,ii,ps)
        fex=theano.function([xs,Es,betas,ii,ps],
                            res, updates=u,
                            on_unused_input='warn',
                            allow_input_downcast=True)

        x=np.random.uniform(size=repnum*dim).reshape(repnum,dim)
        E=np.random.uniform(size=repnum)#x.T[0]
        beta=range(repnum)#np.array(range(num),dtype=np.float32)
        i=range(0,repnum,2)#irange
        
        ps=[j%2 for j in xrange(num)]
        print "x",x
        print "E",E
        print "result",fex(x,E,beta,i,ps)
