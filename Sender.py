#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 14:25:31 2017

@author: Liwen 
"""

import numpy as np  
import math
import Sender as s
A=64
#number of antennas
M=4
#data bits modulation order (PSK)
Ni=int(math.log2(A))
#number of Index bits
Nd=int(math.log2(M))
#number of Data bits
fs = 10000  
# sampling rate (samples per second)
fc=2*10e8
#Carrier Frequency
#H=
##Channel matrix

idbits=s.generate_training_bits(Ni+Nd)
ibits,dbits=s.divide_index_data_bits(idbits,Ni)
symbols=s.databits_mapping(M,dbits)
s_BB=s.databits_pulseforming(symbols)
s_BP=s.mixer(s_BB,fc,fs)
#r_BP=antenna_choicer(H,ibits,s_BB)

#symbols_up = np.zeros(N * sps)
#symbols_up[::sps] = symbols
#tx_signal = np.convolve(g, symbols_up)
#rx_signal = np.convolve(g, tx_signal)
#group_delay = (g.size - 1) // 2
#rx_signal = rx_signal[2 * group_delay: -2 * group_delay]
  
    

