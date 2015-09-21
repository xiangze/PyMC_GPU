# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 02:16:48 2015

@author: xiangze
"""

import theano
from theano import shared
import theano.tensor as T
from theano import function, shared
import numpy as np

from scipy import linalg
import exchange

#theano.config.exception_verbosity="high"

def func_withupdate(ins,outs,updates):
    return theano.function(ins,outs,updates=updates,
                           on_unused_input='warn',
                           allow_input_downcast=True)

def recursion(f,vars,n):
    """return values r,updates"""
    return theano.scan(f,
                       outputs_info=vars,
                       n_steps=n)
    
def reshape3_1(x):
    xx=T.stack(x).dimshuffle(0,2,1,3)
    s=T.shape(xx)
    return T.reshape(xx,(s[0]*s[1]*s[2],s[3]))

def simpleex(i,betas,Es,xs):
    return xs[i+1],xs[i]
#    return T.switch(ex(i,i+1,betas,Es),[xs[i+1],xs[i]],[xs[i],xs[i+1]])

def simpleexchange(xs,Es,betas,ii,parity):
    x,updates=theano.map(simpleex,[ii],non_sequences=[betas,Es,xs])
    return reshape3_1(x),updates
    
def p(X,E): 
    print X.get_value()
    print E.get_value()
