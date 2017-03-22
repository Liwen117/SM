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
def sm(SNR_noise_dB,SNR_RA_dB):
    SA=4
    #number of sender antennas
    RA=8
    #number of receiver antennas
    M=4
    #data bits modulation order (PSK)
    
    mpsk_map =1/np.sqrt(2) * np.array([1+1j, -1+1j, 1-1j, -1-1j], dtype=complex)
    #mpsk_map=np.array([1,-1])
    N=5000
    #number of training symbols
    Ni=int(math.log2(SA))
    #number of Index bits per symbol
    Nd=int(math.log2(M))
    #number of Data bits per symbol
    fs = 10000  
    # sampling rate (samples per second)
    fc=2*10e8
    #Carrier Frequency
    
    # RRC impulse response
    sps = 1  # samples per symbol(!=>ueberabtastung)
    K = 8  # length of the impulse response in symbols (!8*4 =32 index in Zeitbereich)
    rho = 0.5  # RRC rolloff factor (!bandbreite*rolloff factor)
    g = rrc.get_rrc_ir(K * sps + 1, sps, 1, rho)  

    H=1/np.sqrt(2)*(np.random.randn(RA,SA)+1j*np.random.randn(RA,SA))
       
    ##Channel matrix
#    SNR_noise_dB=20
#    SNR_RA_dB=10
#    
    
    
    
    idbits=s.generate_training_bits(N*(Ni+Nd))
    ibits,dbits=s.divide_index_data_bits(idbits,Ni,Nd)
    symbols=s.databits_mapping(mpsk_map,dbits)
    s_BBr=s.databits_pulseforming(symbols.real,g,sps)
    s_BBi=s.databits_pulseforming(symbols.imag,g,sps)
    
    s_BP=s.mixer(s_BBr,s_BBi,fc,fs)
    
    
    #
    r_BP=s.channel(H,ibits,s_BP,RA,SNR_noise_dB,SNR_RA_dB,g,sps)
    r_BBr,r_BBi=r.dmixer(r_BP,fc,fs)
    r_BBr_MF=r.Matched_Filter(r_BBr,g,sps)  
    r_BBi_MF=r.Matched_Filter(r_BBi,g,sps) 
    r_BB_MF=r_BBr_MF+1j*r_BBi_MF
    
    
    
    yi,yd=r.detector(SNR_RA_dB,H,mpsk_map,r_BB_MF) 
    beri=r.BER(yi,ibits,Ni)
    berd=r.BER(yd,dbits,Nd)
    #print(beri,berd)
    return beri,berd
