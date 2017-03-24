#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 14:25:31 2017

@author: Liwen 
"""

import numpy as np  
from rrc import rrcfilter
from sender_class import sender
from receiver_class import receiver
import f_sync as f
#def sm(SNR_noise_dB,SNR_RA_dB,f_off):
SNR_noise_dB=20
SNR_RA_dB=0
SA=4
#number of sender antennas
RA=8
#number of receiver antennas
M=2
#data bits modulation order (BPSK)
bpsk_map=np.array([1,-1])
#qpsk_map =1/np.sqrt(2) * np.array([1+1j, -1+1j, 1-1j, -1-1j], dtype=complex)
N=200
#number of symbols
T=1
#symbol duration
f_off=0.1
#number of training symbols
Ni=int(np.log2(SA))
#number of Index bits per symbol
Nd=int(np.log2(M))
#number of Data bits per symbol
filter_=rrcfilter(16*1+1, 1, 1, 0)
# RRC Filter (L=K * sps + 1, sps, t_symbol, rho)
H=1/np.sqrt(2)*((np.random.randn(RA,SA))+1/np.sqrt(2)*(np.random.randn(RA,SA)))
# Channel matrix
#H=np.ones([RA,SA])  

sender_=sender(N,Ni,Nd,bpsk_map,filter_)
#tx

s_BB_off=sender_.bbsignal()
#*np.exp(1j*np.pi*f_off*np.arange(sender_.bbsignal().size)/T)
#with frequency offset

receiver_=receiver(H,sender_,s_BB_off,SNR_noise_dB,SNR_RA_dB,filter_,bpsk_map)
#rx

BERi,BERd=receiver_.BER()
#Bit Error Rate

