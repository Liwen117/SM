#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 17:56:21 2017

@author: lena
"""
import numpy as np  
import rrc
from commpy.utilities import bitarray2dec

def generate_training_bits(L):
    tx_bits =np.random.choice([0,1],L)
    return  tx_bits

def divide_index_data_bits(idbits,Ni):
    #if (idbits.size % (Ni+Nd) !=0):
    #   idbits=idbit[:idbits.size-idbits.size % (Ni+Nd)]
        return idbits[:Ni],idbits[Ni:]
   
def databits_mapping(M,dbits):
    mpsk_map = np.exp(1j * 2 * np.pi * np.arange(M) / M)
    indices=bitarray2dec(dbits) 
    return  mpsk_map[indices]

def databits_pulseforming(symbols):
    sps = 2  # samples per symbol(!=>ueberabtastung)
    K = 8  # length of the impulse response in symbols (!8*4 =32 index in Zeitbereich)
    rho = 0.5  # RRC rolloff factor (!bandbreite*rolloff factor)
    g = rrc.get_rrc_ir(K * sps + 1, sps, 1, rho)  # RRC impulse response
    return np.convolve(g, symbols, mode='same')
    
def mixer(s_BB,fc,fs):
    x = s_BB.real
    y = s_BB.imag
    t = np.arange(s_BB.size) / fs
    cosine = np.sqrt(2) * np.cos(2 * np.pi * fc * t)
    sine = np.sqrt(2) * np.sin(2 * np.pi * fc * t)
    return x * cosine - y * sine

#def antenna_choicer(H,ibits,s_BB):
#    return r_BP