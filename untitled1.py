#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 21:28:14 2017

@author: lena
"""

import sm 
import numpy as np
SNR_N=np.arange(0,30)
SNR_R=20
BERi=np.zeros(SNR_N.size)
BERd=np.zeros(SNR_N.size)
for i in range(0,30):
    BERi[i],BERd[i]=sm.sm(SNR_N[i],SNR_R)
print(BERi, BERd)