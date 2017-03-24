#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 11:17:55 2017

@author: Liwen
"""


#optimal detector
import numpy as np
from commpy.utilities import dec2bitarray,bitarray2dec
#dmixer
def channel(H,ibits,s,RA,SNR_dB,SNR_RA_dB,g,sps):
    noise_variance_linear = 10**(-SNR_dB / 10)
    s_a_index=bitarray2dec(ibits)
    #turn index bits to the Antenne index
 #！！sps!!   

    group_delay = (g.size - 1) // 2
    c=s_a_index[0:group_delay]
    s_a_index=np.concatenate((c,s_a_index,c))
    r=np.zeros((s.size,RA))*(1+1j)
    #initiate received signal in Bandpass
    for j in range(0,RA):
        for i in range(0,s_a_index.size):
            n = np.sqrt(noise_variance_linear / 2) * (np.random.randn(s.size)+1j*np.random.randn(s.size) )
            r[i,j]=np.sqrt(10**(SNR_RA_dB / 10))*s[i]*H[j,s_a_index[i]]
            r[:,j]=r[:,j]+n
    return r

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
    r=np.zeros((len(r_BB)-h.size+1,r_BB.shape[1]),complex)
    for i in range(0,r_BB.shape[1]):
        a= np.convolve(h, r_BB[:,i])
        r[:,i] = a[ 2*group_delay: - 2*group_delay]
    r_down = r[::sps]
    return r_down
    

def detector(SNR_dB,H,mapp,r):
    n=H.shape[1]
    g=np.zeros((n,mapp.size,r.shape[0]),complex)
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


#r_BB=r.channel(H,sender_.ibits,s_BB_off,SNR_noise_dB,SNR_RA_dB,filter_)
#
##r_BB=f.ML_FLL(r_BB,g)
#
#r_BBr_MF=r.Matched_Filter(r_BB.real,filter_.ir() ,filter_.n_up)  
#r_BBi_MF=r.Matched_Filter(r_BB.imag,filter_.ir() ,filter_.n_up)
#r_=r_BBr_MF+1j*r_BBi_MF
#
##r_=s.channel(H,ibits,symbols,RA,SNR_noise_dB,SNR_RA_dB)
#
#yi,yd=r.detector(SNR_RA_dB,H,bpsk_map,r_) 
#beri=r.BER(yi,sender_.ibits,Ni)
#berd=r.BER(yd,sender_.dbits,Nd)
#print(beri,berd)
##return beri,berd