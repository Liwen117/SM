
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 14:25:31 2017

@author: Liwen 
"""

import numpy as np  
from rrc import rrcfilter
from sender import sender
from receiver import receiver
import test
from f_sync import ML_approx_known, ML_unknown, ML_approx_unknown
from joint_estimation import joint_estimation
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def sm(SNR_dB,N,N_known,threshold):
    #number of sender antennas
    SA=4
    #number of receiver antennas
    RA=4
    #data bits modulation order (BPSK)
    M=2
    mpsk_map=np.array([1,-1])
    #mpsk_map =1/np.sqrt(2) * np.array([1+1j, -1+1j, 1-1j, -1-1j], dtype=complex)
    #symbol duration
    T=1*1e-6
    #Frequency offset
    f_off=np.random.randint(0,1*1e-2/T)
    #phase offset
    phi_off=np.random.random()*2*np.pi
    #number of Index bits per symbol
    Ni=int(np.log2(SA))
    #number of Data bits per symbol
    Nd=int(np.log2(M))
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #Upsampling rate
    n_up=1
    # RRC Filter (L=K * sps + 1, sps, t_symbol, rho)
    filter_=rrcfilter(8*n_up+1,n_up , 1,0)
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #Channel matrix
    H=1/np.sqrt(2)*((np.random.randn(RA,SA))+1j/np.sqrt(2)*(np.random.randn(RA,SA)))
    #H=np.ones([RA,SA])
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #sender
    sender_=sender(N,N_known,Ni,Nd,mpsk_map,filter_)
    #print("n_start=",sender_.n_start)
    #training symbols(/bits) which may be shared with receiver, when a data-aided method is used
    symbols_known=sender_.symbols_known
    ibits=sender_.ibits
    dbits=sender_.dbits
    n_start=sender_.n_start
    ibits_known=sender_.ibits_known
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #with Filter(noch zu bearbeiten,überabtastung!)
    receiver_=receiver(H,sender_,SNR_dB,filter_,mpsk_map)
    receiver_.channel()
    r_mf=receiver_.r_mf
    
    #BER for perfect sync
    yi,yd=receiver_.detector(r_mf,H)
    BERi_0,BERd_0=test.BER(yi,yd,Ni,Nd,ibits,dbits)
    
    
    #with offsets
    off=np.exp(1j*2*np.pi*f_off*np.arange(r_mf.shape[0])*T/filter_.n_up)
    r_off_ft=r_mf*np.repeat(off,RA).reshape([-1,RA])*np.exp(1j*phi_off)

 
    
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #coarse Estimation for f_off (draft)
    f_off_coarse=f_off*0.7
    

        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #coarse synchronisation 
    off_syc=np.exp(-1j*2*np.pi*f_off_coarse*np.arange(r_mf.shape[0])*T/filter_.n_up)
    r_syc_coarse=r_off_ft*np.repeat(off_syc,RA).reshape([-1,RA])
    
      #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #Sampling Clock Synchronization
    
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #Joint Estimation for f_off,n_start and CSI 
    j=joint_estimation()
    j.function(r_syc_coarse,N,N_known,T,ibits_known,symbols_known,SA,RA)
    f_est=j.f_est
    n_est=j.n_est
    H_est=j.H_est

    #    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #Frequency synchronisation
    off_syc=np.exp(-1j*2*np.pi*f_est*(np.arange(r_mf.shape[0]))*T)
    r_f_syc=r_syc_coarse*np.repeat(off_syc,RA).reshape([-1,RA])
    
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ##Test for the Joint estimation
    yi,yd=receiver_.detector(r_f_syc,H_est)
    #yi,yd=rr.detector(H_est,SNR_dB,mpsk_map,r_ft_syc)
    BERi,BERd=test.BER(yi,yd,Ni,Nd,ibits,dbits)
    
    
    #
    #    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #
    if BERi<=threshold and BERd<=threshold and n_start==n_est:
        count=1
    else:
        count=0
        
    
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    return count    








