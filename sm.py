#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 14:25:31 2017

@author: Liwen 
"""
from commpy.utilities import dec2bitarray,bitarray2dec 
import numpy as np  
from rrc import rrcfilter
from sender_class import sender
from receiver_class import receiver
import f_sync as fr
from f_sync import ML_approx
#def sm(SNR_noise_dB,SNR_RA_dB,f_off):
SNR_noise_dB=100
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
f_off=0.46
#number of training symbols
Ni=int(np.log2(SA))
#number of Index bits per symbol
Nd=int(np.log2(M))
#number of Data bits per symbol
filter_=rrcfilter(8*2+1,2 , 1, 0)
# RRC Filter (L=K * sps + 1, sps, t_symbol, rho)
# besser mit rho=1
H=1/np.sqrt(2)*((np.random.randn(RA,SA))+1j/np.sqrt(2)*(np.random.randn(RA,SA)))
H=np.abs(H)
# Channel matrix
#H=np.ones([RA,SA])  

sender_=sender(N,Ni,Nd,bpsk_map,filter_)
#tx
s=sender_.bbsignal()
#r_off=s

#with frequency offset
symbols=sender_.symbols
ibits=sender_.ibits
receiver_=receiver(H,sender_,s,SNR_noise_dB,SNR_RA_dB,filter_,bpsk_map)
off=np.exp(1j*2*np.pi*f_off*np.arange(sender_.bbsignal().size)*T/filter_.n_up)
r=receiver_.channel()*np.repeat(off,8).reshape([-1,8])
f_of=ML_approx(filter_,r,T,symbols,ibits,H)
print(f_of)

#f_delta=100
#group_delay = (filter_.ir().size - 1) // 2
#p=np.zeros([f_delta,H.shape[0]],complex)
#f_o=np.zeros(H.shape[0])
#r_up=np.zeros([r.shape[0]-filter_.ir().size+1,H.shape[0]],complex)
#r_=np.zeros([symbols.size,H.shape[0]])
#for j in range(0,H.shape[0]):
#    a= np.convolve(filter_.ir(), r[:,j])
#    r_up[:,j]= a[ 2*group_delay: - 2*group_delay]
#    r_[:,j] = r_up[::filter_.n_up,j]
#    s_a_index=bitarray2dec(ibits)
#    for f in range(0,f_delta):
#        for i in range(0,r_.shape[0]):
#            off_=np.exp(-1j*2*np.pi*f/f_delta*T*i)
#            p[f,j]+=symbols[i]*r_[i,j]*H[j,s_a_index[i]]*off_
#    f_o[j]=np.argmax(np.abs(p[:,j]))/f_delta
#print(f_o)
#r_of=receiver_.channel()
#f_est=np.zeros([RA,SA])
#for j in range(0,SA):
#    for i in range(0,RA):
#        f_est[i,j]=ML_approx(H[i,:],filter_,r_of[:,i],T,sender_.symbols)
#print(f_est)
##nicht korrekt




#group_delay = (filter_.ir().size - 1) // 2
#r_=np.zeros((len(r)-filter_.ir().size+1),complex)
#a= np.convolve(filter_.ir(), r)
#r_= a[ 2*group_delay: - 2*group_delay]
#r_mf = r_[::filter_.n_up]
#p=np.zeros(1000)
#for f in range(0,1000):
#    for i in range(0,r_mf.size):
#        p[f]=sender_.symbols[i]*r_mf[i]*np.exp(-1j*2*np.pi*f/1000*i*T/filter_.n_up)
#q=np.abs(p)
#f_o=np.argmax(q)/1000




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
BERi,BERd=receiver_.BER()

