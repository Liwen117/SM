#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 11:17:42 2017

@author: Liwen
"""
import numpy as np
from commpy.utilities import dec2bitarray

def BER(yi,yd,Ni,Nd,ibits,dbits):
    xi=np.zeros((yi.size,Ni))
    xd=np.zeros((yd.size,Nd))
    for i in range(0,yi.size):
        xi[i]=dec2bitarray(int(yi[i]),Ni)
    beri=np.sum(np.not_equal(xi.reshape((1,-1)),np.matrix(ibits).H.reshape((1,-1))))/xi.size 
    for i in range(0,yd.size):
        xd[i]=dec2bitarray(int(yd[i]),Nd)
    berd=np.sum(np.not_equal(xd.reshape((1,-1)),np.matrix(dbits).H.reshape((1,-1))))/xd.size 
    return beri, berd



