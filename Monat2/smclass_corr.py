#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 16:37:23 2017

@author: lena
"""
import numpy as np  
from rrc import rrcfilter
from sender import sender
from receiver import receiver
import test
from f_sync import ML_approx_known, ML_unknown, ML_approx_unknown,FLL,NDA,DC
from takt_synchro import gardner_timing_recovery
import time
from commpy.utilities import bitarray2dec 
import matplotlib.pyplot as plt  
from joint_estimation import joint_estimation
import scipy.linalg as lin
import scipy.signal as sig

import commpy



def sm(SNR_dB):
    fc=1*1e9  # LTE
    offset_range=40*1e-6
    #print("f_max=",fc*offset_range)
    #number of sender antennas
    SA=2
    #number of receiver antennas
    RA=1
    #data bits modulation order (BPSK)
    M=2
    mpsk_map=np.array([1,-1])
    #mpsk_map =1/np.sqrt(2) * np.array([1+1j, -1+1j, 1-1j, -1-1j], dtype=complex)
    #mpsk_map =1/np.sqrt(2) * np.array([1, 1j, -1j, -1], dtype=complex)
    #number of symbols per Frames
    Ns=2
    #number of Frames
    Nf=100
    #number of symbols
    N=Ns*Nf
    #number of training symbols
    N_known=64*4
    N=N_known*2
    k=8
    #symbol duration
    T=1*1e-6
    #print("f_vernachlaessigbar=",0.01/N/T)
    #T=1
    #Frequency offset
    f_off=np.random.randint(-fc*offset_range,fc*offset_range)*0.1
    #f_off=np.random.randint(-0.01/T,0.01/T)
    #N_known=int(1//T//f_off/4)
    #N=10*N_known
    print("f_off=",f_off)
    #symbol offset 
    #n_off=2
    #phase offset
    phi_off=np.random.random()*2*np.pi
    phi_off=0
    #number of Index bits per symbol
    Ni=int(np.log2(SA))
    #number of Data bits per symbol
    Nd=int(np.log2(M))
    #Upsampling rate
    n_up=2
    # RRC Filter (L=K * sps + 1, sps, t_symbol, rho)
    K=20
    filter_=rrcfilter(K*n_up+1,n_up , 1,0.5)
    g=filter_.ir()
    #Plot.spectrum(g,"g")
    #Channel matrix
    H=1/np.sqrt(2)*((np.random.randn(RA,SA))+1j/np.sqrt(2)*(np.random.randn(RA,SA)))
    #H=np.array([[0.5,0.1]])
    H=np.ones([RA,SA])
    MSE_n=[]
    MSE_f=[]
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    tries=1
    for i in range(0,tries):
        #sender
        sender_=sender(N,N_known,Ni,Nd,mpsk_map,filter_,k)
        print("n_start=",sender_.n_start,sender_.n_start*n_up,(sender_.n_start+N_known)*n_up)
        #training symbols(/bits) which may be shared with receiver, when a data-aided method is used
        symbols_known=sender_.symbols_known
        #symbols_known=ss
        symbols=sender_.symbols
        ibits=sender_.ibits
    #    dbits=sender_.dbits
        ibits_known=sender_.ibits_known
        index=bitarray2dec(ibits_known)
       # dbits_known=sender_.dbits_known
        
        
        s_BB=sender_.bbsignal()
        group_delay = (g.size - 1) // 2

            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        #with Filter
        receiver_=receiver(H,sender_,SNR_dB,filter_,mpsk_map)
        
        r=receiver_.r


            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        #with offsets
        #Frequency offset before MF(+filter length) 
        #!!!T anpassen!!!
        off=np.exp(1j*2*np.pi*f_off*np.arange(sender_.bbsignal().size)*T/n_up)
        r=receiver_.r*np.repeat(off,RA).reshape([-1,RA])
        r_mf=receiver_.Matched_Filter(r.real)+1j*receiver_.Matched_Filter(r.imag)
        r_mf=r_mf[2*group_delay:-2*group_delay]
    

    
    #%%%%%%%%%%%%%%%%%
    #Modified Delay Correlation
        f_est,n_est,M=DC(r_mf,T,symbols_known,n_up,N_known,k)

        print("n_est=",n_est)
# 
        print("f_est=",f_est)

            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    #Frequency synchronisation
#        y=r_mf*np.exp(-1j*2*np.pi*f_off*(np.arange(r_mf.shape[0])+2*group_delay)*T/n_up).reshape([-1,RA])
        
                #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#MSE
        MSE_n.append((sender_.n_start-n_est)**2)
        MSE_f.append((f_off-f_est)**2)
        
        return np.average(MSE_n), np.average(MSE_f)