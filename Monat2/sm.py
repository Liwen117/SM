
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
from f_sync import ML_approx_known, ML_unknown, ML_approx_unknown,FLL
import time
from commpy.utilities import bitarray2dec 
import matplotlib.pyplot as plt  
from joint_estimation import joint_estimation
import scipy.linalg as lin
import scipy.signal as sig
#carrier Frequency
fc=5.2*1e9  # IEEE 802.11 WLAN
offset_range=40*1e-6
SNR_dB=100
#number of sender antennas
SA=4
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
N=4
#number of training symbols
N_known=0
#symbol duration
T=1*1e-3
#Frequency offset
f_off=np.random.randint(-fc*offset_range,fc*offset_range)*0.01
f_off=100
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
n_up=1
# RRC Filter (L=K * sps + 1, sps, t_symbol, rho)
filter_=rrcfilter(6*n_up+1,n_up , 1,0)
g=filter_.ir()


#Channel matrix
H=1/np.sqrt(2)*((np.random.randn(RA,SA))+1j/np.sqrt(2)*(np.random.randn(RA,SA)))
#H=np.ones([RA,SA])

#sender
sender_=sender(N,N_known,Ni,Nd,mpsk_map,filter_)
print("n_start=",sender_.n_start)
#training symbols(/bits) which may be shared with receiver, when a data-aided method is used
symbols_known=sender_.symbols_known
symbols=sender_.symbols
ibits=sender_.ibits
dbits=sender_.dbits
ibits_known=sender_.ibits_known
dbits_known=sender_.dbits_known


s_BB=sender_.bbsignal()
plt.figure()
plt.scatter(s_BB.real, s_BB.imag)
plt.xlabel('I'); plt.ylabel('Q')
plt.title('Signal from sender');
#Kommentar: Nullpunkt wegen Filterdelay

#plt.figure()
#plt.plot(s_BB)
#plt.title("Baseband signal")
#plt.xlabel("Index")

#plt.figure()
#f = np.linspace(-0.5, 0.5, s_BB.size)
#S_BB = np.abs(np.fft.fftshift(np.fft.fft(s_BB)))**2/s_BB.size
#plt.semilogy(f, S_BB)
#plt.xlim(-0.5, 0.5)
#plt.xlabel("f/B")
#plt.title("Baseband spectrum");

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#with Filter(noch zu bearbeiten,überabtastung!)
receiver_=receiver(H,sender_,SNR_dB,filter_,mpsk_map)
r=receiver_.channel()

plt.figure()
plt.scatter(r.real, r.imag)
plt.xlabel('I'); plt.ylabel('Q')
plt.title('Signal after channel');

off=np.exp(1j*2*np.pi*f_off*np.arange(r.shape[0])*T/filter_.n_up)
r_off_ft=r*np.repeat(off,RA).reshape([-1,RA])*np.exp(1j*phi_off)

#r_mf=receiver_.r_mf
#sp=np.fft.fft(r)
#yi,yd=receiver_.detector(r_mf,H)
#BERi_0,BERd_0=test.BER(yi,yd,Ni,Nd,ibits,dbits)
#plt.figure()
#plt.scatter(r_mf.real, r_mf.imag)
#plt.xlabel('I'); plt.ylabel('Q')
#plt.title('Signal after MF');

#plt.figure()
#plt.plot(r_mf)
#plt.title("Signal after MF")
#plt.xlabel("Index")
#
#plt.figure()
#i=1
#f = np.linspace(-0.5, 0.5, r_mf[:,i].size)
#R_MF = np.abs(np.fft.fftshift(np.fft.fft(r_mf[:,i])))**2/r_mf[:,i].size
#plt.semilogy(f, R_MF)
#plt.xlim(-0.5, 0.5)
#plt.xlabel("f/B")
#plt.title("Spectrum for Signal after MF");

#with offsets
#Frequency offset before MF(+filter length)
#off=np.exp(1j*2*np.pi*f_off*np.arange(sender_.bbsignal().size)*T/filter_.n_up)
#r=receiver_.channel()*np.repeat(off,RA).reshape([-1,RA])
#r_mf=receiver_.Matched_Filter(r.real)+1j*receiver_.Matched_Filter(r.imag)
#offset after MF

#r=receiver_.channel()
#r_mf=receiver_.Matched_Filter(r.real)+1j*receiver_.Matched_Filter(r.imag)
#off=np.exp(1j*2*np.pi*f_off*np.arange(r_mf.shape[0])*T/filter_.n_up)
#
##
#r_off_ft=r_mf*np.repeat(off,RA).reshape([-1,RA])*np.exp(1j*phi_off)
#r_off_ft=np.concatenate((r_off_f[n_off:],r_off_f[:n_off]))*np.exp(1j*2*np.pi*phi_off)

#
plt.figure()
plt.scatter(r_off_ft.real, r_off_ft.imag)
plt.xlabel('I'); plt.ylabel('Q')
plt.title('Signal with offset');


plt.figure()
f = np.linspace(-0.5, 0.5, s_BB.size)
S_BB = np.abs(np.fft.fftshift(np.fft.fft(s_BB)))**2/s_BB.size
plt.semilogy(f, S_BB)
plt.xlim(-0.5, 0.5)
plt.xlabel("f/B")

R_off_ft = np.abs(np.fft.fftshift(np.fft.fft(r_off_ft)))**2/r_off_ft.size
plt.semilogy(f, R_off_ft)
plt.xlim(-0.5, 0.5)
plt.xlabel("f/B")
plt.title("Baseband spectrum without and with offset");

#    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##FLL
g_mf=g
r=r_off_ft[:,0]
x_out=FLL(r,g_mf,n_up)

plt.figure()
f = np.linspace(-0.5, 0.5, x_out.size)
S_BB = np.abs(np.fft.fftshift(np.fft.fft(x_out)))**2/x_out.size
plt.semilogy(f, S_BB)
plt.xlim(-0.5, 0.5)
plt.xlabel("f/B")
#    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#g_mf=g
#r=r_off_ft[:,0]


#k = np.arange(np.ceil(-len(g_mf)/2),np.floor(len(g_mf)/2)+1) 
#T=1 
#g_dmf= 2*np.pi*T*k[:]*g_mf[:] 
## Ausgabe initialisieren 
#x_out = np.zeros([len(r)/n_up*2,1],complex) 
#f = 0 
## 
##Schrittweite 
#gamma = 0.01 
#x_buf = np.zeros([len(g_mf)-1],complex) 
#x_1 = complex(0)
#y_buf = np.zeros([len(g_dmf)-1],complex)
#y_1 = complex(0)
#nu = 0 
#phi = 0 
#cnt=0
## 
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
##% Phasenkorrektur 
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
## 
#for ii in range(0,len(r)): 
##     
##    % Korrektur des Eingangswertes 
#    r[ii] = r[ii] * np.exp(1j*(-phi)) 
##     
##    % neues Phaseninkrement berechnen 
#    phi = phi + 2*np.pi*nu/8000
##     
##    % Filteroperationen, MF+DMF 
##    x=np.convolve(g_mf,r[ii])[0] 
##    y=np.convolve(g_dmf,r[ii])[0] 
#    (x,x_buf)=sig.lfilter(g_mf, np.ones(len(g_mf)),[r[ii]],0,x_buf)
#    
#    (y,y_buf)=sig.lfilter(g_dmf, np.ones(len(g_dmf)),[r[ii]],0,y_buf)
#     
##     
##       
##    % Downsample by 8 
#    if (np.mod(ii,n_up) == 1): 
##         
##         Fehler berechnen 
#            e = 0.5*np.imag(x_1*np.conj(y_1)) + 0.5*np.imag(x*np.conj(y))
#            
#            nu= nu + gamma*e[0];      
##        Frequenzoffset berechnen 
#            f=f+nu; 
#
#    if (np.mod(ii,n_up/2) == 1): 
##         
#        x_1 = x
#        y_1 = y
##         
##        % Ausgabe, Faktor 2 ueberabgetastet! 
#        x_out[cnt] = x
#        cnt=cnt+1
#
#
#plt.figure()
#plt.scatter(x_out.real, x_out.imag)
#plt.xlabel('I'); plt.ylabel('Q')
#plt.title('Signal after FLL')
#
#f=f/len(r)*n_up
#print(f)


r_f_syc=x_out[6:-6:2]

#    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
##    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##Frequency synchronisation
#off_syc=np.exp(-1j*2*np.pi*f_est*(np.arange(r_mf.shape[0]))*T)
#r_f_syc=r_syc_coarse*np.repeat(off_syc,RA).reshape([-1,RA])
#
#    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
###Test for the Joint estimation
yi,yd=receiver_.detector(r_f_syc,H)
#yi,yd=rr.detector(H_est,SNR_dB,mpsk_map,r_ft_syc)
BERi,BERd=test.BER(yi,yd,Ni,Nd,ibits,dbits)
#
#
##
#    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##
#print("f_est=",f_est,", n_est=",n_est," , H_diff_max=", np.max(H-H_est))  
#print("BER for index bits=",BERi,", BER for data bits=",BERd)  
###







