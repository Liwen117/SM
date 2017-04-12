#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 16:07:53 2017

@author: lena
"""
from sm_class import sm
r=80
SNR=30
N=100
N_known=30
threshold=0.01

c=0
for i in range(0,r):
    c+=sm(SNR,N,N_known,threshold)
    
print(c,"times of ",r,"successes")