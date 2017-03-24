#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 14:25:31 2017

@author: Liwen 
"""

import numpy as np  
import math
import rrc
import sender_function as s
import receiver_function as r
#def sm(SNR_noise_dB,SNR_RA_dB,f_off):
SNR_noise_dB=20
SNR_RA_dB=0
SA=4
#number of sender antennas
RA=8
#number of receiver antennas
M=2
#data bits modulation order (BPSK)

#qpsk_map =1/np.sqrt(2) * np.array([1+1j, -1+1j, 1-1j, -1-1j], dtype=complex)
bpsk_map=np.array([1,-1])
N=200
T=1
#symbol duration
f_off=np.pi/10
#number of training symbols
Ni=int(math.log2(SA))
#number of Index bits per symbol
Nd=int(math.log2(M))
#number of Data bits per symbol


# RRC impulse response
sps = 1  # samples per symbol(!=>ueberabtastung)
K = 16  # length of the impulse response in symbols (!8*4 =32 index in Zeitbereich)
rho = 0  # RRC rolloff factor (!bandbreite*rolloff factor)
g = rrc.get_rrc_ir(K * sps + 1, sps, 1, rho)  

H=1/np.sqrt(2)*((np.random.randn(RA,SA))+1/np.sqrt(2)*(np.random.randn(RA,SA)))
#H=np.ones([RA,SA])  


idbits=s.generate_training_bits(N*(Ni+Nd))
ibits,dbits=s.divide_index_data_bits(idbits,Ni,Nd)
symbols=s.databits_mapping(bpsk_map,dbits)
s_BBr=s.databits_pulseforming(symbols.real,g,sps)
s_BBi=s.databits_pulseforming(symbols.imag,g,sps)
s_BB=s_BBr+1j*s_BBi

s_BB=s_BB*np.exp(1j*np.pi*f_off*np.arange(s_BB.size)/T)
#with frequency offset


r_BB=s.channel(H,ibits,s_BB,RA,SNR_noise_dB,SNR_RA_dB,g,sps)
r_BBr_MF=r.Matched_Filter(r_BB.real,g,sps)  
r_BBi_MF=r.Matched_Filter(r_BB.imag,g,sps)
r_=r_BBr_MF+1j*r_BBi_MF

#r_=s.channel(H,ibits,symbols,RA,SNR_noise_dB,SNR_RA_dB)

yi,yd=r.detector(SNR_RA_dB,H,bpsk_map,r_) 
beri=r.BER(yi,ibits,Ni)
berd=r.BER(yd,dbits,Nd)
print(beri,berd)
#return beri,berd
