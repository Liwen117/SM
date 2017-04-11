#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 21:32:50 2017

@author: Liwen
"""
import numpy as np
from sm_class import sm
import matplotlib.pyplot as plt
     #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
     #BER-SNR
def BER_SNR(SNR_range):
    BERi=np.zeros(SNR_range)
    BERd=np.zeros(SNR_range)
    for i in range(0,SNR_range):
        BERi[i],BERd[i]=sm(i)
    return BERi,BERd

SNR_range=30
BERI,BERD=BER_SNR(SNR_range)
SNR = np.arange(SNR_range)
plt.plot(SNR, BERI)
plt.title("BER_SNR"); plt.ylabel("BER for Indexbits"); plt.xlabel("SNR/dB"); 
plt.yscale("log")
plt.show()

plt.plot(SNR, BERD)
plt.title("BER_SNR"); plt.ylabel("BER for Databits"); plt.xlabel("SNR/dB"); 
plt.yscale("log")
plt.show()
     #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    