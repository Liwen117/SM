#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 11:17:55 2017

@author: Liwen
"""


#optimal detector
import numpy as np
from commpy.utilities import dec2bitarray
#dmixer
def dmixer(r_BP,fc,fs):
    t = np.arange(len(r_BP)) / fs
    cos = np.sqrt(2) * np.cos(2 * np.pi * fc * t)
    sin = np.sqrt(2) * np.sin(2 * np.pi * fc * t)
    r_BBr=np.zeros(r_BP.shape)
    r_BBi=np.zeros(r_BP.shape)
    for i in range(0,r_BP.shape[1]):
        r_BBr[:,i]=r_BP[:,i] * cos 
        r_BBi[:,i]=r_BP[:,i] * (-sin)
    return r_BBr,r_BBi

def Matched_Filter(r_BB,h,sps):
    group_delay = (h.size - 1) // 2
    r=np.zeros((len(r_BB)-h.size+1,r_BB.shape[1]))*(1+1j)
    for i in range(0,r_BB.shape[1]):
        a= np.convolve(h, r_BB[:,i])
        r[:,i] = a[ 2*group_delay: - 2*group_delay]
    r_down = r[::sps]
    return r_down
    

def detector(SNR_dB,H,mapp,r):
    n=H.shape[1]
    g=np.zeros((n,mapp.size,r.shape[0]))*(1+1j)
    yi=np.zeros(r.shape[0])
    yd=np.zeros(r.shape[0])
    for i in range(0,r.shape[0]):
        #per symbol
        for j in range(0,n):
            #which sender
            for q in range(0,mapp.size):
                #which datasymbol
                g[j,q,i]=np.sqrt(10**(SNR_dB / 10))*np.linalg.norm(H[:,j]*mapp[q])**2-2*np.real(r[i]@H[:,j]*mapp[q])
        yi[i],yd[i]=np.unravel_index(np.argmin(g[:,:,i]), (n,mapp.size))
    return yi,yd


def BER(y,bits,N):    
    x=np.zeros((y.size,N))
    for i in range(0,y.size):
        x[i]=dec2bitarray(int(y[i]),N)
    return np.sum(np.not_equal(x.reshape((1,x.size)),np.matrix(bits).H.reshape((1,bits.size))))/x.size   