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
import f_sync as fr
from f_sync import ML_approx_known, NDA, ML_unknown, ML_approx_unknown
import time
from commpy.utilities import bitarray2dec 
import matplotlib.pyplot as plt  # plotting library
#def sm(SNR_noise_dB,SNR_RA_dB,f_off):
SNR_noise_dB=50
SNR_RA_dB=0
SA=4
#number of sender antennas
RA=16
#number of receiver antennas
M=2
#data bits modulation order (BPSK)
mpsk_map=np.array([1,-1])
#mpsk_map =1/np.sqrt(2) * np.array([1+1j, -1+1j, 1-1j, -1-1j], dtype=complex)
N=5
#number of symbols
T=1*1e-3
#symbol duration
f_off=24
n_off=0
phi_off=0
#number of training symbols
Ni=int(np.log2(SA))
#number of Index bits per symbol
Nd=int(np.log2(M))
#number of Data bits per symbol
n_up=1
filter_=rrcfilter(8*n_up+1,n_up , 1,0)
# RRC Filter (L=K * sps + 1, sps, t_symbol, rho)
#???? BER fuer Index verschlechtet sich bei Uebungabtastung
# besser mit rho=1
#H=1/np.sqrt(2)*((np.random.randn(RA,SA))+1j/np.sqrt(2)*(np.random.randn(RA,SA)))
#H=np.abs(H)
#Channel matrix
H=np.ones([RA,SA])/4

sender_=sender(N,Ni,Nd,mpsk_map,filter_)
#tx
s=sender_.bbsignal()
#r_off=s

#training symbols(/bits) which may be shared with receiver, when a data-aided method is used
symbols=sender_.symbols
ibits=sender_.ibits


#receiver_=receiver(H,sender_,s,SNR_noise_dB,SNR_RA_dB,filter_,mpsk_map)
#r=receiver_.channel()
#r_mf=receiver_.Matched_Filter(r.real)+1j*receiver_.Matched_Filter(r.imag)
#BERi,BERd=receiver_.BER()

#with frequency offset
#off=np.exp(1j*2*np.pi*f_off*np.arange(sender_.bbsignal().size)*T/filter_.n_up)
#r=receiver_.channel()*np.repeat(off,RA).reshape([-1,RA])
#r_mf=receiver_.Matched_Filter(r.real)+1j*receiver_.Matched_Filter(r.imag)
#r_mf=r_mf*np.exp(1j*2*np.pi*phi_off)
#r_mf=np.concatenate((r_m[n_off:],r_m[:n_off]))



#without Filter
off=np.exp(1j*2*np.pi*f_off*np.arange(symbols.size)*T)
noise_variance_linear = 10**(-SNR_noise_dB / 10)
s_a_index=np.repeat(bitarray2dec(ibits),n_up)
r=np.zeros((symbols.size,H.shape[0]),complex)
for j in range(0,H.shape[0]):
    for i in range(0,s_a_index.size):
        n = np.sqrt(noise_variance_linear / 2) * (np.random.randn(symbols.size)+1j*np.random.randn(symbols.size) )
        r[i,j]=np.sqrt(10**(SNR_RA_dB / 10))*symbols[i]*H[j,s_a_index[i]]
        r[:,j]=r[:,j]
        
r=r*np.repeat(off,RA).reshape([-1,RA])
#n_range=10
#L=np.zeros(n_range)
##when n=0 function doesn't work
#for n in range(1,n_range):
#    r=r_mf[:-n]
#    symbs=symbols[n:]
#    index=bitarray2dec(ibits[:,n:])
#    f_est=ML_approx_unknown(r,T,symbs,ibits[:,n:])
#    H_est=np.zeros([RA,SA],complex)  
#    for k in range(n,symbs.size):
#        H_est[:,index[k]]+= r_mf[k,:]*symbs[k]*np.exp(-1j*2*np.pi*T*k*f_est)/np.sum(symbs**2)    
#    #Likelihood function for Timing estimation
#        L[n]+=np.linalg.norm(r[k,:]-H_est[:,index[k]]*np.exp(1j*2*np.pi*T*(k-n)*symbs[k-n]))**2
#n_est=np.argmax(L)
##    

#n=n_off
#f_est=ML_approx_unknown(r_mf[:-n],T,symbols[n:],ibits[:,n:])  
#sy=symbols[n:]







#t1=time.clock()
#Frequency offset estimation with ML-Approximation(data-aided) with known channel
#f_est=ML_approx1(r_mf,T,symbols,ibits,H)

#ML with channel unknown
#f_est=ML_unknown(r_mf,T,symbols,ibits)

#ML_approx with channel unknown
f_est=ML_approx_unknown(r,T,symbols,ibits)
#TEST for CS
H_est=np.zeros([RA,SA],complex)  
index=bitarray2dec(ibits)
for k in range(0,symbols.size):
    H_est[:,index[k]]+= r[k,:]*symbols[k]*np.exp(-1j*2*np.pi*T*f_est*k)
H_est=H_est/np.sum(symbols**2)    
Hd=H-H_est
#Anzahl soll auf Anzahl der Benutzung von jeder Sendeantenne angepasst werden!!!
##

##Frequency offset estimation with Non-Data-Aided method based on MPSK
#f_of=NDA(r_mf,M,T,H)
#r_of= np.repeat(r_mf,H.shape[1]).reshape([N,RA,SA])
#t=time.clock()-t1




print(f_est)    








