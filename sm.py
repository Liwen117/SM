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
from f_sync import ML_approx
#def sm(SNR_noise_dB,SNR_RA_dB,f_off):
SNR_noise_dB=30
SNR_RA_dB=0
SA=4
#number of sender antennas
RA=8
#number of receiver antennas
M=2
#data bits modulation order (BPSK)
bpsk_map=np.array([1,-1])
#qpsk_map =1/np.sqrt(2) * np.array([1+1j, -1+1j, 1-1j, -1-1j], dtype=complex)
N=500
#number of symbols
T=0.01
#symbol duration
f_off=0.51
#number of training symbols
Ni=int(np.log2(SA))
#number of Index bits per symbol
Nd=int(np.log2(M))
#number of Data bits per symbol
filter_=rrcfilter(8*1+1,1 , 1,0)
# RRC Filter (L=K * sps + 1, sps, t_symbol, rho)
#???? BER fuer Index verschlechtet sich bei Uebungabtastung
# besser mit rho=1
H=1/np.sqrt(2)*((np.random.randn(RA,SA))+1j/np.sqrt(2)*(np.random.randn(RA,SA)))
H=np.abs(H)
# Channel matrix
#H=np.ones([RA,SA])  

sender_=sender(N,Ni,Nd,bpsk_map,filter_)
#tx
s=sender_.bbsignal()
#r_off=s

#training symbols(/bits) which may be shared with receiver, when a data-aided method is used
symbols=sender_.symbols
ibits=sender_.ibits


receiver_=receiver(H,sender_,s,SNR_noise_dB,SNR_RA_dB,filter_,bpsk_map)
#BERi,BERd=receiver_.BER()

#with frequency offset
off=np.exp(1j*2*np.pi*f_off*np.arange(sender_.bbsignal().size)*T/filter_.n_up)
r=receiver_.channel()*np.repeat(off,8).reshape([-1,8])
#Frequency offset estimation with ML-Approximation(data-aided)
f_of=ML_approx(filter_,r,T,symbols,ibits,H)




print(f_of)


#g_mf=filter_.ir()
#k = np.arange(np.ceil(-len(g_mf)/2),np.floor(len(g_mf)/2)+1)
#T=1
#g_dmf= 2*np.pi*T*k[:]*g_mf[:]
## Ausgabe initialisieren
##x_out = np.zeros([len(r)/4,1],complex)
#f = 0
##
##Schrittweite
#gamma = 0.5
#
#x_1 = complex(0);
#
#y_1 = complex(0);
#
#nu = 0;
#
#phi = 0;
#
#t=0
#for ii in range(0,r.size):
##    
##    % Korrektur des Eingangswertes
#    r[ii] = r[ii] * np.exp(1j*(-phi))
##    
##    % neues Phaseninkrement berechnen
#    phi = phi + 2*np.pi*nu*t
##    
##    % Filteroperationen, MF+DMF
#
#    x=sum(np.convolve(g_mf,r[ii]))
#    y=sum(np.convolve(g_dmf,r[ii]))
#    
##    
##      
##    % Downsample by 4
#    if (np.mod(ii,4) == 1):
##        
##        % Fehler berechnen
#        e = 0.5*np.imag(x_1*np.conj(y_1)) + 0.5*np.imag(x*np.conj(y)); 
##        
##        % Frequenzoffset berechnen
#        nu= nu + gamma*e;     
##        
##        % nur zur Visualisierung
#        f=f+nu;
##    end
##    
##    % Downsample by 2
#    if (np.mod(ii,2) == 1):
##        
#        x_1 = x;
#        y_1 = y;
#    t=t+T
##        % Ausgabe, Faktor 2 ueberabgetastet!
#








#for ii in range(0,r.size):
##    
##    % Korrektur des Eingangswertes
#    r[ii] = r[ii] * np.exp(1j*(-phi))
#    
###    % neues Phaseninkrement berechnen
#
###    
###    % Filteroperationen, MF+DMF
#    x=np.convolve(g_mf,r[ii])[0]
#    y=np.convolve(g_dmf,r[ii])[0]
#
### Fehler berechnen
#    e = 0.5*np.imag(x_1*np.conj(y_1)) + 0.5*np.imag(x*np.conj(y))
#    if (np.mod(ii,4) == 1):
##        
#        x_1 = x;
#        y_1 = y;
#    nu += gamma*e     
#    f += nu
#    t += T
#    phi = 2*np.pi*f*t



##rx
#
#


