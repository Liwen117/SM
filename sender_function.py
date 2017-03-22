#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 17:56:21 2017

@author: Liwen
"""
import numpy as np  

from commpy.utilities import bitarray2dec

def generate_training_bits(L):
    tx_bits =np.random.choice([0,1],L)
    #tx_bits=np.ones(L)
    return  tx_bits

def divide_index_data_bits(idbits,Ni,Nd):
    #if (idbits.size % (Ni+Nd) !=0):
    #   idbits=idbit[:idbits.size-idbits.size % (Ni+Nd)]
    divided_bits=idbits.reshape((Nd+Ni,idbits.size/(Nd+Ni)))
    ibits=divided_bits[0:Ni,:]
    dbits=divided_bits[Ni:Ni+Nd,:]
    return ibits,dbits

def databits_mapping(mapp,dbits):
    indices=bitarray2dec(dbits) 
    return  mapp[indices]

def databits_pulseforming(symbols,g,sps):
    symbols_up = np.zeros(symbols.size * sps)
    symbols_up[::sps] = symbols
    return np.convolve(g, symbols_up)
    
def mixer(s_BBr,s_BBi,fc,fs):
    t = np.arange(s_BBr.size) / fs
    cosine = np.sqrt(2) * np.cos(2 * np.pi * fc * t)
    sine = np.sqrt(2) * np.sin(2 * np.pi * fc * t)
    return s_BBr * cosine - s_BBi * sine



def channel(H,ibits,s,RA,SNR_dB,SNR_RA_dB,g,sps):
    noise_variance_linear = 10**(-SNR_dB / 10)
    s_a_index=bitarray2dec(ibits)
 #！！sps!!   
#    s=(1+0j)*np.zeros(len(H))
    group_delay = (g.size - 1) // 2
    c=s_a_index[0:group_delay]
    s_a_index=np.concatenate((c,s_a_index,c))
#    s_BP = s_BP[ group_delay: - group_delay:sps]
    r_BP=np.zeros((s.size,RA))+1j*np.zeros((s.size,RA))
#    for i in range(1,np.size(s_BP)):
#        s[s_a_index]=s_BP[i]   
    for j in range(0,RA):
        for i in range(0,s_a_index.size):
            n = np.sqrt(noise_variance_linear / 2) * (np.random.randn(s.size)+1j*np.random.randn(s.size) )
            r_BP[i,j]=np.sqrt(10**(SNR_RA_dB / 10))*s[i]*H[j,s_a_index[i]]
            r_BP[:,j]=r_BP[:,j]+n
    return r_BP