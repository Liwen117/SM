#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 11:17:55 2017

@author: Liwen
"""


#optimal detector
import numpy as np

#dmixer
def dmixer(r_BP,fc,fs):
    t = np.arange(r_BP.size) / fs
    cosine = np.sqrt(2) * np.cos(2 * np.pi * fc * t)
    sine = np.sqrt(2) * np.sin(2 * np.pi * fc * t)
    return r_BP * cosine - r_BP * sine

def Matched_Filter(r_BB,h):
    return np.convolve(h, r_BB, mode='same')
    

def detector(SNR,H,mapp,r_BB_MF,r_index):
    g=np.zeros(len(H)*mapp.size)
    for j in range(1,len(H)):
        for q in range(1,mapp.size):
            g[j][q]=np.sqrt(SNR)*np.linalg.norm(H[j][r_index]*mapp[q])-2*np.real(np.matrix(r_BB_MF).H*H[j][r_index]*mapp[q])
    return np.unravel_index(np.argmax(g), (len(H),mapp.size))