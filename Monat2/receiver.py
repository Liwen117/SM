#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 11:17:55 2017

@author: Liwen
"""


#optimal detector
import numpy as np
from commpy.utilities import bitarray2dec
import matplotlib.pyplot as plt
class receiver():
    def __init__(self,H,sender_,SNR_dB,filter_,mapp):
        self.H=H
        #self.H_est=H_est
        self.ibits=sender_.ibits
#        self.dbits=sender_.dbits
        self.Ni=sender_.Ni
        self.Nd=sender_.Nd
        self.s=sender_.bbsignal()
        self.symbols=sender_.only_upsampling()
        self.SNR_dB=SNR_dB
        self.MF_ir=filter_.ir()
        self.sps=filter_.n_up
        self.mapp=mapp
        self.r=self.channel()
        self.r_mf=self.Matched_Filter(self.r.real)+1j*self.Matched_Filter(self.r.imag)
        self.r_down = self.r_mf[::self.sps]
        #self.detector(self.r_mf)
     #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%     
#index fuer group delay waehlen, zusammen nach MF loeschen         
    def channel(self):
        noise_variance_linear = 10**(-self.SNR_dB/self.Nd / 10)
        s_a_index=np.repeat(bitarray2dec(self.ibits),self.sps)
        #turn index bits to the Antenne index 
         
#        group_delay = (self.MF_ir.size - 1) // 2
#        #print("delay=",group_delay)
#        c=s_a_index[0:group_delay]
#        d=s_a_index[-group_delay:]
##?      c=np.zeros(group_delay)
#
#        s_a_index=np.concatenate((c,s_a_index,d))
#        self.index=s_a_index
        r=np.zeros((self.s.size,self.H.shape[0]),complex)
        #initiate received signal in Bandpass
        for j in range(0,self.H.shape[0]):
            for i in range(0,s_a_index.size):
                r[i,j]=self.s[i]*self.H[j,s_a_index[i]]

        n = np.sqrt(noise_variance_linear / 2) * (np.random.randn(r.shape[0],r.shape[1])+1j*np.random.randn(r.shape[0],r.shape[1]))
        r=r+n
        return r
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    def Matched_Filter(self,r_BB):
        #group_delay = (self.MF_ir.size - 1) // 2
        #r=np.zeros((r_BB.shape[0]+self.MF_ir.size-1-4*group_delay,r_BB.shape[1]),complex)
        r=np.zeros((r_BB.shape[0]+self.MF_ir.size-1,r_BB.shape[1]),complex)
#        #for each receiver-antenne
        for i in range(0,r_BB.shape[1]):
            #a= np.convolve(self.MF_ir, r_BB[:,i])
            #r[:,i] = a[ 2*group_delay: -2* group_delay]
            r[:,i]=np.convolve(self.MF_ir, r_BB[:,i])
#        #without downsampling
#            r[:,i]=np.convolve(self.MF_ir, r_BB[:,i])
        self.r_down = r[::self.sps]

        return r
#        #with downsampling

#    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 
    def detector(self,r,H_est):
        n=H_est.shape[1]
        g=np.zeros((n,self.mapp.size,r.shape[0]),complex)
        yi=np.zeros(r.shape[0])
        yd=np.zeros(r.shape[0])
        for i in range(0,r.shape[0]):
            #i.th symbol
            for j in range(0,n):
                #which sender
                for q in range(0,self.mapp.size):
                    #which datasymbol
                    g[j,q,i]=np.linalg.norm(H_est[:,j]*self.mapp[q])**2-2*np.real(np.dot(np.conj(r[i]),H_est[:,j]*self.mapp[q]))
            yi[i],yd[i]=np.unravel_index(np.argmin(g[:,:,i]), (n,self.mapp.size))
        return yi,yd

   