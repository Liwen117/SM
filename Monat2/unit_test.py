#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 16:07:53 2017

@author: lena
"""
from sm_class import sm
import numpy as np
r=50
SNR=np.arange(0,50,5)
N=100
N_known=50
threshold=0.1

c=np.zeros(SNR.size)
for j in range(0,SNR.size):
    for i in range(0,r):
        c[j]+=sm(SNR[j],N,N_known,threshold)
c=1-c/r
#print(c,"times of ",r,"successes")

#SNR=30
#c=np.zeros(19)
#for j in range(1,20):
#    for i in range(0,r):
#        c[j-1]+=sm(SNR,j)