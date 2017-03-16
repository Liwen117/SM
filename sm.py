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
    SA=8
    #number of sender antennas
    RA=4
    #number of receiver antennas
    M=2
    #data bits modulation order (PSK)
    mpsk_map = np.exp(1j * 2 * np.pi * np.arange(M) / M)
    N=10
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
    #!attibute?
    #H=np.ones((4,8))+1j*np.ones((4,8))
    H=np.random.rand(4,8)+1j*np.random.rand(4,8)
    ##Channel matrix

    
    
    
    
    idbits=s.generate_training_bits(N*(Ni+Nd))
    ibits,dbits=s.divide_index_data_bits(idbits,Ni,Nd)
    symbols=s.databits_mapping(mpsk_map,dbits)
    s_BBr=s.databits_pulseforming(symbols.real,g,sps)
    s_BBi=s.databits_pulseforming(symbols.imag,g,sps)
    
    #s_BP=s.mixer(s_BBr,s_BBi,fc,fs)
    
    
    #
    #r_BP=s.channel(H,ibits,s_BB,RA,SNR_noise_dB,g,sps)
    r_BBr=s.channel(H,ibits,s_BBr,RA,SNR_noise_dB,SNR_RA_dB,g,sps)
    r_BBi=s.channel(H,ibits,s_BBi,RA,SNR_noise_dB,SNR_RA_dB,g,sps)
    #r_BBr,r_BBi=r.dmixer(r_BP,fc,fs)
    r_BBr_MF=r.Matched_Filter(r_BBr,g,sps)  
    r_BBi_MF=r.Matched_Filter(r_BBi,g,sps) 
    r_BB_MF=r_BBr_MF+1j*r_BBi_MF
    
    
    #r=r_BB_MF
    #n=H.shape[1]
    #g=np.zeros((n,mpsk_map.size,r.shape[0])) 
    #yi=np.zeros(r.shape[0])
    #yd=np.zeros(r.shape[0])
    #for i in range(0,r.shape[0]):
    #        #per symbol
    #        for j in range(0,n):
    #            #which sender
    #            for q in range(0,mpsk_map.size):
    #                #which datasymbol
    #                g[j,q,i]=np.sqrt(10**(SNR_RA_dB / 10))*np.linalg.norm(H[:,j]*mpsk_map[q])**2-2*np.real(r[i]@H[:,j]*mpsk_map[q])
    #                yi[i],yd[i]=np.unravel_index(np.argmin(g[:,:,i]), (n,mpsk_map.size))
    
    yi,yd=r.detector(SNR_RA_dB,H,mpsk_map,r_BB_MF) 
    #from commpy.utilities import dec2bitarray
    #x=np.zeros((yi.size,Ni))
    #for i in range(0,yi.size):
    #    x[i]=dec2bitarray(int(yi[i]),Ni)
    #ber=np.sum(np.not_equal(x.reshape((1,x.size)),np.matrix(ibits).H.reshape((1,ibits.size))))/x.size  
    #print(ber) 
    beri=r.BER(yi,ibits,Ni)
    berd=r.BER(yd,dbits,Nd)
    return beri,berd
    