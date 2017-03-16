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
SA=64
#number of sender antennas
RA=4
#number of receiver antennas
M=4
#data bits modulation order (PSK)
mpsk_map = np.exp(1j * 2 * np.pi * np.arange(M) / M)

Ni=int(math.log2(SA))
#number of Index bits
Nd=int(math.log2(M))
#number of Data bits
fs = 10000  
# sampling rate (samples per second)
fc=2*10e8
#Carrier Frequency

# RRC impulse response
sps = 2  # samples per symbol(!=>ueberabtastung)
K = 8  # length of the impulse response in symbols (!8*4 =32 index in Zeitbereich)
rho = 0.5  # RRC rolloff factor (!bandbreite*rolloff factor)
g = rrc.get_rrc_ir(K * sps + 1, sps, 1, rho)  

H=np.ones((SA,RA))
##Channel matrix
SNR_noise_dB=20
SNR_RA_dB=20

r_index=1

idbits=s.generate_training_bits(Ni+Nd)
ibits,dbits=s.divide_index_data_bits(idbits,Ni)
symbols=s.databits_mapping(mpsk_map,dbits)
s_BB=s.databits_pulseforming(symbols,g)
s_BP=s.mixer(s_BB,fc,fs)
#Rauschen
noise_variance_linear = 10**(-SNR_noise_dB / 10)
n = np.sqrt(noise_variance_linear / 2) * (np.random.randn(s_BP.size) + 1j*np.random.randn(s_BP.size))

r_BP=s.antenna_choicer(H,ibits,s_BB)+n

#symbols_up = np.zeros(N * sps)
#symbols_up[::sps] = symbols
#tx_signal = np.convolve(g, symbols_up)
#rx_signal = np.convolve(g, tx_signal)
#group_delay = (g.size - 1) // 2
#rx_signal = rx_signal[2 * group_delay: -2 * group_delay]

r_BB=r.dmixer(r_BP)
r_BB_MF=r.Matched_Filter(r_BB,g)  
y=r.detector(SNR_RA_dB,H,mpsk_map,r_BB,r_index) 

