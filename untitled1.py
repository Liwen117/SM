#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 21:28:14 2017

@author: lena
"""

import sm 
import numpy as np
#SNR_N=np.arange(0,30)
#SNR_R=0

#BERi=np.zeros(SNR_N.size)
#BERd=np.zeros(SNR_N.size)

n_up=np.arange(1,10)
BERi=np.zeros(n_up.size)
BERd=np.zeros(n_up.size)
for i in range(0,n_up.size):
    BERi[i],BERd[i]=sm.sm(n_up[i])
print(BERi, BERd)