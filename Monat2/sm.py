
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
from f_sync import ML_approx_known, ML_unknown, ML_approx_unknown,FLL,NDA
import time
from commpy.utilities import bitarray2dec 
import matplotlib.pyplot as plt  
from joint_estimation import joint_estimation
import scipy.linalg as lin
import scipy.signal as sig
import Plot
#carrier Frequency
fc=5.2*1e9  # IEEE 802.11 WLAN
offset_range=40*1e-6
SNR_dB=30
#=Eb/N0
#number of sender antennas
SA=2
#number of receiver antennas
RA=1
#data bits modulation order (BPSK)
M=2
mpsk_map=np.array([1,-1])
#mpsk_map =1/np.sqrt(2) * np.array([1+1j, -1+1j, 1-1j, -1-1j], dtype=complex)
#number of symbols per Frames
Ns=100
#number of Frames
Nf=20
#number of symbols
N=Ns*Nf
N=250
#number of training symbols
N_known=0
#symbol duration
T=1*1e-3
#T=1
#Frequency offset
f_off=np.random.randint(-fc*offset_range,fc*offset_range)*0.01
f_off=np.random.randint(-0.1/T,0.1/T)
f_off=320
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
n_up=8
# RRC Filter (L=K * sps + 1, sps, t_symbol, rho)
filter_=rrcfilter(6*n_up+1,n_up , 1,0.35)
g=filter_.ir()
#Plot.spectrum(g,"g")
#Channel matrix
H=1/np.sqrt(2)*((np.random.randn(RA,SA))+1j/np.sqrt(2)*(np.random.randn(RA,SA)))
#H=np.ones([RA,SA])
f_est=[]
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fl=FLL(g,n_up)
for i in range(0,250):
    #sender
    sender_=sender(N,N_known,Ni,Nd,mpsk_map,filter_)
    #print("n_start=",sender_.n_start)
    #training symbols(/bits) which may be shared with receiver, when a data-aided method is used
    symbols_known=sender_.symbols_known
    symbols=sender_.symbols
    ibits=sender_.ibits
    dbits=sender_.dbits
    ibits_known=sender_.ibits_known
    dbits_known=sender_.dbits_known
    
    
    s_BB=sender_.bbsignal()
    
    #spec=sender_.anti_image(s_BB)
    
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #Plot.konstellation(s_BB,'Signal from sender')
    ###Kommentar: Nullpunkt wegen Filterdelay
    #Plot.timesignal(s_BB,"Baseband signal")
    #Plot.spectrum(s_BB,"Baseband spectrum")
    
    #!!Image filterung
    
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #with Filter
    receiver_=receiver(H,sender_,SNR_dB,filter_,mpsk_map)
    group_delay = (g.size - 1) // 2
    r=receiver_.channel()[group_delay:-group_delay,:]
    
    #Plot.timesignal(rr,"nach Kanal")
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #Plot.konstellation(r,'Signal after channel')
    #Plot.timesignal(r[:,0],'Signal after channel')
    #Plot.spectrum(r,'Signal after channel')
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    off=np.exp(1j*2*np.pi*f_off*np.arange(r.shape[0])*T/filter_.n_up)
    r_off_ft=r*np.repeat(off,RA).reshape([-1,RA])*np.exp(1j*phi_off)
    
    
    
    #sp=np.fft.fft(r)
    #yi,yd=receiver_.detector(r_mf,H)
    #BERi_0,BERd_0=test.BER(yi,yd,Ni,Nd,ibits,dbits)
    
    
    
    #with offsets
    #Frequency offset before MF(+filter length) 
    #!!!T anpassen!!!
    #f_off=0
    #off=np.exp(1j*2*np.pi*f_off*np.arange(sender_.bbsignal().size)*T/filter_.n_up)
    #r=receiver_.channel()*np.repeat(off,RA).reshape([-1,RA])
    #r_mf=receiver_.Matched_Filter(r.real)+1j*receiver_.Matched_Filter(r.imag)
    
    
    #Plot.timesignal(r_mf[:,0],"nach MF")
    #Plot.timesignal(r_mf[:n_up],"1. Symbol nach MF")
    #Plot.spectrum(r_mf[:,0],"nach MF")
    #Plot.spectrum(r_mf[:n_up],"1. Symbol nach MF")
    #Plot.timesignal(receiver_.r_down[:,0],"downsampling")
    
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #Plot.konstellation(r_mf,'Signal after MF')
    #Plot.timesignal(r_mf,'Signal after MF')
    #Plot.spectrum(r_mf,'Signal after MF')
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #offset after MF
    #r=receiver_.channel()
    #r_mf=receiver_.Matched_Filter(r.real)+1j*receiver_.Matched_Filter(r.imag)
    #off=np.exp(1j*2*np.pi*f_off*np.arange(r_mf.shape[0])*T/filter_.n_up)
    #
    ##
    #r_off_ft=r_mf*np.repeat(off,RA).reshape([-1,RA])*np.exp(1j*phi_off)
    #r_off_ft=np.concatenate((r_off_f[n_off:],r_off_f[:n_off]))*np.exp(1j*2*np.pi*phi_off)
    
    # %%%%%%%%%%%%%%%%%%%%%%
    #f_NDA=NDA(r_mf,M,T,H,n_up)
    #print("NDA:",f_NDA)
    
    
    
    
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #FLL
    #CSI=true
    
    g_mf=g
    
    r=r_off_ft[:,0]
    
    #for i in range(0,500):
    #    r=r*np.exp(-1j*2*np.pi*f*np.arange(sender_.bbsignal().size)*T/filter_.n_up)    
        #print("phi=",fl.phi)
    x_out,f=fl.recovery(r)
    f_est.append(f)


    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#Plot.konstellation(r_mf,'Signal after FLL')
#Plot.timesignal(r_mf,'Signal after FLL')
#Plot.spectrum(r_mf,'Signal after FLL')


    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##### Frequency estimation without n_offset
#    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##
###Frequency offset estimation with ML-Approximation(data-aided) with known channel for each antenna
#f_est=ML_approx_known(r_off_ft[sender_.n_start:sender_.n_start+N_known],T,symbols_known,ibits_known,H)[0]
#H_est=H
#n_est=sender_.n_start
#    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
###ML with channel unknown
###f_est1=ML_unknown(r_off_f,T,symbols,ibits)
#    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
###ML_approx with channel unknown
##f_est1=ML_approx_unknown(r_off_ft[sender_.n_start:sender_.n_start+N_known],T,symbols_known,ibits_known)
#    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
####Channel Estimation
###H_est=np.zeros([RA,SA],complex)  
###index=bitarray2dec(ibits)
###i=np.zeros(SA)
###for k in range(0,symbols.size):
###    H_est[:,index[k]]+= r[k,:]/symbols[k]*np.exp(-1j*2*np.pi*T*f_est*k)
###    i[index[k]]=i[index[k]]+1
####Anzahl soll auf Anzahl der Benutzung von jeder Sendeantenne angepasst werden
###H_est=H_est/np.repeat(i,RA).reshape(-1,RA).transpose()
###H_diff=H-H_est
##
#
#    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
###speed test
###t1=time.clock()
###t=time.clock()-t1
##
#
#
#    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##coarse Estimation for f_off
#f_off_coarse=0
#    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##coarse synchronisation for f_off
#off_syc=np.exp(-1j*2*np.pi*f_off_coarse*np.arange(r_mf.shape[0])*T/filter_.n_up)
#r_syc_coarse=r_off_ft*np.repeat(off_syc,RA).reshape([-1,RA])
#print(f_off-f_off_coarse)
#
#    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
###Joint Estimation for f_off,n_start and CSI 
##j=joint_estimation()
##j.function(r_syc_coarse,N,N_known,T,ibits_known,symbols_known,SA,RA)
##f_est=j.f_est
##n_est=j.n_est
##H_est=j.H_est
   #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##Frequency synchronisation
#off_syc=np.exp(-1j*2*np.pi*f_est*(np.arange(r_mf.shape[0]))*T)
#r_f_syc=r_syc_coarse*np.repeat(off_syc,RA).reshape([-1,RA])

   #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
###Test for the Joint estimation
#if CSI==true:
#    H_est=H  #with unknown CSI
#yi,yd=receiver_.detector(r_f_syc,H_est)
##yi,yd=rr.detector(H_est,SNR_dB,mpsk_map,r_ft_syc)
#BERi,BERd=test.BER(yi,yd,Ni,Nd,ibits,dbits)

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##
#print("f_est=",f_est,", n_est=",n_est," , H_diff_max=", np.max(H-H_est))  
#print("BER for index bits=",BERi,", BER for data bits=",BERd)  
###







