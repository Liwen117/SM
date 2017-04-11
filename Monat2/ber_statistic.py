#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 10:34:20 2017

@author: lena
"""

import ber1
import numpy as np
r=50
n_min=1
n_max=9
step=2
BERi=np.zeros((n_max-n_min)//step+1)
BERd=np.zeros((n_max-n_min)//step+1)
for n in range(0,r):
    BERi_,BERd_=ber1.test_upsampling(n_min,n_max+1,step)
    BERi+=BERi_
    BERd+=BERd_
    n=n+1
BERi=BERi/r
BERd=BERd/r