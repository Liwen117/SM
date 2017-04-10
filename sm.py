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
import receiver_class as rr
import test
import f_sync as fr
from f_sync import ML_approx_known, NDA, ML_unknown, ML_approx_unknown
import time
from commpy.utilities import bitarray2dec 
import matplotlib.pyplot as plt  # plotting library
#def sm(n_up):
SNR_noise_dB=30
SNR_RA_dB=0
SA=16
#number of sender antennas
RA=16
#number of receiver antennas
M=2
#data bits modulation order (BPSK)
mpsk_map=np.array([1,-1])
#mpsk_map =1/np.sqrt(2) * np.array([1+1j, 1+1j, 1-1j, -1-1j], dtype=complex)
N=50
#number of symbols
T=1*1e-6
#symbol duration
f_off=50
n_off=5
phi_off=0
#number of training symbols
Ni=int(np.log2(SA))
#number of Index bits per symbol
Nd=int(np.log2(M))
#number of Data bits per symbol
n_up=8
filter_=rrcfilter(8*n_up+1,n_up , 1,0)
# RRC Filter (L=K * sps + 1, sps, t_symbol, rho)
#???? BER fuer Index verschlechtet sich bei Uebungabtastung
# besser mit rho=1
H=1/np.sqrt(2)*((np.random.randn(RA,SA))+1j/np.sqrt(2)*(np.random.randn(RA,SA)))
#H=np.abs(H)
#Channel matrix
#H=np.ones([RA,SA])
#H_est=H.real*0.9+1j*H.imag*1.1
sender_=sender(N,Ni,Nd,mpsk_map,filter_)
#tx
s=sender_.bbsignal()
#r_off=s

#training symbols(/bits) which may be shared with receiver, when a data-aided method is used
symbols=sender_.symbols
ibits=sender_.ibits
dbits=sender_.dbits
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##with Filter(noch zu bearbeiten,überabtastung!)
#receiver_=receiver(H,sender_,SNR_noise_dB,SNR_RA_dB,filter_,mpsk_map)
#r=receiver_.channel()
#r_mf=receiver_.r_mf
##yi=receiver_.yi
##yd=receiver_.yd
#yi,yd=receiver_.detector(r_mf,H)
#BERi_0,BERd_0=test.BER(yi,yd,Ni,Nd,ibits,dbits)
#
#
##with offsets
##Frequency offset before MF(+filter length)
##off=np.exp(1j*2*np.pi*f_off*np.arange(sender_.bbsignal().size)*T/filter_.n_up)
##r=receiver_.channel()*np.repeat(off,RA).reshape([-1,RA])
##r_mf=receiver_.Matched_Filter(r.real)+1j*receiver_.Matched_Filter(r.imag)
##offset after MF
#
#r=receiver_.channel()
#r_mf=receiver_.Matched_Filter(r.real)+1j*receiver_.Matched_Filter(r.imag)
#off=np.exp(1j*2*np.pi*f_off*np.arange(r_mf.shape[0])*T/filter_.n_up)
#
##test
#rrr=np.concatenate((r_mf[n_off:],r_mf[:n_off]))
##
#r_m=r_mf*np.exp(1j*2*np.pi*phi_off)*np.repeat(off,RA).reshape([-1,RA])
#r_off_ft=np.concatenate((r_m[n_off:],r_m[:n_off]))


    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#without Filter

#noise_variance_linear = 10**(-SNR_noise_dB / 10)
#s_a_index=np.repeat(bitarray2dec(ibits),n_up)
#rx=np.zeros((symbols.size,H.shape[0]),complex)
#for j in range(0,H.shape[0]):
#    for i in range(0,s_a_index.size):
#        n = np.sqrt(noise_variance_linear / 2) * (np.random.randn(symbols.size)+1j*np.random.randn(symbols.size) )
#        rx[i,j]=np.sqrt(10**(SNR_RA_dB / 10))*symbols[i]*H[j,s_a_index[i]]
#        rx[:,j]=rx[:,j]+n
receiver_=receiver(H,sender_,SNR_noise_dB,SNR_RA_dB,filter_,mpsk_map)
rx=receiver_.channel_nf(n_up)
yi,yd=receiver_.detector(rx,H)      
BERi_0,BERd_0=test.BER(yi,yd,Ni,Nd,ibits,dbits)        
#return BERi_0,BERd_0
#        
##with offsets  
#off=np.exp(1j*2*np.pi*f_off*np.arange(symbols.size)*T)      
#r_off_f=rx*np.repeat(off,RA).reshape([-1,RA])
#r_off_ft=np.concatenate((r_off_f[n_off:],r_off_f[:n_off]))
#
#
#   # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##Joint Estimation for f_off,n_off and CSI 
#n_range=8
#L=np.zeros(n_range-1)
##when n=0 function doesn't work,this situation should be tested by a if command
#for n in range(1,n_range):
#    r=r_off_ft[:-n,:] 
#    symbs=symbols[n:]
#    index=bitarray2dec(ibits[:,n:])
#    f_estt=ML_approx_unknown(r,T,symbs,ibits[:,n:])
#    H_est=np.zeros([RA,SA],complex)  
#    i=np.zeros(SA)
#    #Channel estimation
#
#    for k in range(0,symbs.size):
#        H_est[:,index[k]]+= r[k,:]/symbs[k]*np.exp(-1j*2*np.pi*T*f_estt*(k+n))
#        i[index[k]]=i[index[k]]+1
#    #Anzahl soll auf Anzahl der Benutzung von jeder Sendeantenne angepasst werden
#    H_est=H_est/np.repeat(i,RA).reshape(-1,RA).transpose()
#   
#    #Likelihood function for Timing estimation       
#    for m in range(0,symbs.size):
#        L[n-1]+=np.linalg.norm(r[m,:]-H_est[:,index[m]]*np.exp(1j*2*np.pi*T*m)*symbs[m])**2
#    n_est=np.argmin(L)+1   
#
#r_n_syc=r_off_ft[:-n_est,:]
#
##save Estimation results for each n or one more estimation after estimation for n?(2)
#f_est=ML_approx_unknown(r_n_syc,T,symbols[n_est:],ibits[:,n_est:]) 
#
#symbs=symbols[n_est:]
#index=bitarray2dec(ibits[:,n_est:])
#H_est=np.zeros([RA,SA],complex) 
#i=np.zeros(SA)
#for k in range(0,symbs.size):
#    H_est[:,index[k]]+= r_n_syc[k,:]/symbs[k]*np.exp(-1j*2*np.pi*T*f_est*(k+n_est))
#    i[index[k]]=i[index[k]]+1
#H_est=H_est/np.repeat(i,RA).reshape(-1,RA).transpose()
#H_diff=H-H_est
##f sychr
#off_syc=np.exp(-1j*2*np.pi*f_est*(np.arange(r_n_syc.shape[0])+n_est)*T)
#r_ft_syc=r_n_syc*np.repeat(off_syc,RA).reshape([-1,RA])
#
#
#yi,yd=receiver_.detector(r_ft_syc,H)
##yi,yd=rr.detector(H_est,SNR_RA_dB,mpsk_map,r_ft_syc)
#BERi,BERd=test.BER(yi,yd,Ni,Nd,ibits[:,n_est:],dbits[:,n_est:])
#
##(test OK)
#
#
#    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
###Test only! for CSI after the Estimation
##sender_=sender(N,Ni,Nd,mpsk_map,filter_)
###tx
##s=sender_.bbsignal()
###r_off=s
##
###training symbols(/bits) which may be shared with receiver, when a data-aided method is used
##symbols=sender_.symbols
##ibits=sender_.ibits
##receiver_=receiver(H,H_est,sender_,s,SNR_noise_dB,SNR_RA_dB,filter_,mpsk_map)
##r=receiver_.channel()
##r_mf=receiver_.Matched_Filter(r.real)+1j*receiver_.Matched_Filter(r.imag)
##BERi,BERd=receiver_.BER()
#
#    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##Test for the Joint estimation
#
#
#
#
#
#    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##Frequency offset estimation with ML-Approximation(data-aided) with known channel
##f_est=ML_approx1(r_mf,T,symbols,ibits,H)
#    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##ML with channel unknown
##f_est=ML_unknown(r,T,symbols,ibits)
#    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##ML_approx with channel unknown
##f_est=ML_approx_unknown(r,T,symbols,ibits)
#    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
###Channel Estimation
##H_est=np.zeros([RA,SA],complex)  
##index=bitarray2dec(ibits)
##i=np.zeros(SA)
##for k in range(0,symbols.size):
##    H_est[:,index[k]]+= r[k,:]/symbols[k]*np.exp(-1j*2*np.pi*T*f_est*k)
##    i[index[k]]=i[index[k]]+1
###Anzahl soll auf Anzahl der Benutzung von jeder Sendeantenne angepasst werden
##H_est=H_est/np.repeat(i,RA).reshape(-1,RA).transpose()
##H_diff=H-H_est
###(tested,ok for all situations)
#
###Frequency offset estimation with Non-Data-Aided method based on MPSK
##f_of=NDA(r_mf,M,T,H)
##r_of= np.repeat(r_mf,H.shape[1]).reshape([N,RA,SA])
#    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##duration test
##t1=time.clock()
##t=time.clock()-t1
#
#
#    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##
#print(f_est,n_est)    
#
#
#
#




