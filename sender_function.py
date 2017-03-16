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
    return  tx_bits

def divide_index_data_bits(idbits,Ni):
    #if (idbits.size % (Ni+Nd) !=0):
    #   idbits=idbit[:idbits.size-idbits.size % (Ni+Nd)]
        return idbits[:Ni],idbits[Ni:]
   

def databits_mapping(mapp,dbits):
    indices=bitarray2dec(dbits) 
    return  mapp[indices]

def databits_pulseforming(symbols,g):

    return np.convolve(g, symbols, mode='same')
    
def mixer(s_BB,fc,fs):
    x = s_BB.real
    y = s_BB.imag
    t = np.arange(s_BB.size) / fs
    cosine = np.sqrt(2) * np.cos(2 * np.pi * fc * t)
    sine = np.sqrt(2) * np.sin(2 * np.pi * fc * t)
    return x * cosine - y * sine

def antenna_choicer(H,ibits,s_BB):
    s_a_index=bitarray2dec(ibits)
    s=np.zeros(len(H))
    s[s_a_index,]=s_BB
    r_BP=s@H
    return r_BP