#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 17:56:21 2017

@author: Liwen
"""
import numpy as np  
from commpy.utilities import bitarray2dec
class sender():
    def __init__(self,N,Ni,Nd,mapp,filter_):
        self.N=N
        self.Ni=Ni
        self.Nd=Nd
        self.mapp=mapp
        self.ir=filter_.ir()
        self.sps=filter_.n_up
    
    def generate_training_bits(self):
        return  np.random.choice([0,1],self.N*(self.Ni+self.Nd))
    
    def divide_index_data_bits(self):
        #if (idbits.size % (Ni+Nd) !=0):
        #   idbits=idbit[:idbits.size-idbits.size % (Ni+Nd)]
        divided_bits=self.generate_training_bits().reshape((self.Nd+self.Ni,-1))
        ibits=divided_bits[0:self.Ni,:]
        dbits=divided_bits[self.Ni:self.Ni+self.Nd,:]
        self.dbits= dbits
        self.ibits= ibits
    
    def databits_mapping(self):
        indices=bitarray2dec(self.dbits) 
        self.symbols=self.mapp[indices]
    
    def databits_pulseforming(self,symbs):
        symbols_up = np.zeros(symbs.size * self.sps)
        symbols_up[:: self.sps] = symbs
        return np.convolve(self.ir, symbols_up)

        
#    def mixer(s_BBr,s_BBi,fc,fs):
#        t = np.arange(s_BBr.size) / fs
#        cos = np.sqrt(2) * np.cos(2 * np.pi * fc * t)
#        sin = np.sqrt(2) * np.sin(2 * np.pi * fc * t)
#        return s_BBr * cos - s_BBi * sin

    def bbsignal(self):
        self.divide_index_data_bits()
        self.databits_mapping()
        s_BBr=self.databits_pulseforming(np.real(self.symbols))
        s_BBi=self.databits_pulseforming(np.imag(self.symbols))
        
        return s_BBr+1j*s_BBi

#def channel(H,ibits,s,RA,SNR_dB,SNR_RA_dB):
#    noise_variance_linear = 10**(-SNR_dB / 10)
#    s_a_index=bitarray2dec(ibits)
#    #turn index bits to the Antenne index
# 
#    r=np.zeros((s.size,RA),complex)
#    #initiate received signal in Bandpass
#    for j in range(0,RA):
#        for i in range(0,s_a_index.size):
#            n = np.sqrt(noise_variance_linear / 2) * (np.random.randn(s.size)+1j*np.random.randn(s.size) )
#            r[i,j]=np.sqrt(10**(SNR_RA_dB / 10))*s[i]*H[j,s_a_index[i]]
#            r[:,j]=r[:,j]+n
#    return r


