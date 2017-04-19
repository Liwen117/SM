#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 17:18:38 2017

@author: lena
"""
import numpy as np
from commpy.utilities import bitarray2dec 
import scipy.linalg as lin
import scipy.signal as sig
import matplotlib.pyplot as plt 
#data-aided ML Approximation with known channel(Voraussetzung:f_off<<1/T) for each antenna
def ML_approx_known(r_,T,symbols,ibits,H):
    f_delta=1
    f_range=100
    interp_fact=10
    p=np.zeros([int(f_range/f_delta),H.shape[0]],complex)
    f_o=np.zeros(H.shape[0])
    xvals = np.linspace(0, f_range, f_range/f_delta*interp_fact)
    x = np.linspace(0, int(f_range/f_delta),int(f_range/f_delta))
    for j in range(0,H.shape[0]):
        s_a_index=bitarray2dec(ibits)
        for f in range(0,int(f_range/f_delta)):
            for i in range(0,r_.shape[0]):
                off_=np.exp(-1j*2*np.pi*f*f_delta*T*i)
                p[f,j]+=symbols[i]*r_[i,j]*H[j,s_a_index[i]]*off_
        #with Interpolation            
        f_o[j]=np.argmax(np.interp(xvals,x,np.abs(p[:,j])))*f_delta**2/interp_fact
    return f_o
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#data-aided ML with channel unknown
def ML_unknown(y,T,symbols,ibits):
    f_delta=1
    f_range=100
    interp_fact=1
    R=np.zeros(y.shape[0],complex)
    X=np.zeros(int(f_range/f_delta),complex)
    
    N=1/np.sum(symbols**2)
    index=bitarray2dec(ibits)
    xvals = np.linspace(0, f_range, f_range/f_delta*interp_fact)
    x = np.linspace(0, int(f_range/f_delta),int(f_range/f_delta))
    for f in range(0,int(f_range/f_delta)):
        for m in range(1,y.shape[0]):
            for k in range(m,y.shape[0]):
                if index[k]==index[k-m]:
                    R[m]+= np.dot(np.conj(y[k-m,:]),y[k,:])*symbols[k]*symbols[k-m]*N
            X[f] +=R[m]*np.exp(-1j*2*np.pi*m*f*f_delta*T)
        R[m]=0
    f_est=np.argmax(np.interp(xvals,x,np.real(X)))*f_delta**2/interp_fact
    return f_est
            

            


    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#data-aided ML Approximation with channel unknown
def ML_approx_unknown(y,T,symbols,ibits):

    R=np.zeros(y.shape[0],complex)

    X_1=0
    X_2=0
    N=1/np.sum(symbols**2)
    index=bitarray2dec(ibits)
    
    for m in range(0,y.shape[0]):
        for k in range(m,y.shape[0]):
            if index[k]==index[k-m]:
                R[m]+=np.dot(np.conj(y[k-m,:]),y[k,:])*symbols[k]*symbols[k-m]*N
        X_1 +=m*np.abs(R[m])*np.angle(R[m])
        X_2 +=m**2*np.abs(R[m])
    
    f_est=1/(2*np.pi*T)*X_1/X_2

    return f_est
            
#non data-aided, MPSK 
     #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def NDA(r,M,T,H,n_up):    
    summ=np.zeros([r.shape[1],H.shape[1]],complex) 
    f_off=np.zeros([r.shape[1],H.shape[1]]) 
    for j in range(0,H.shape[1]): 
        for i in range(0,r.shape[1]): 
            for n in range(1,r.shape[0]): 
                summ[i,j] += (r[n,i]/H[i,j]*np.conj(r[n-1,i]/H[i,j]))**M 
                f_off[i,j]=1/(2*np.pi*T*M)*(np.angle(summ[i,j])) 
    return f_off*n_up
#kommentar: funktioniert nur bei sehr hohem SNR

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def FLL(r,g_mf,n_up): 
    k = np.arange(np.ceil(-len(g_mf)/2),np.floor(len(g_mf)/2)+1) 
    T=1 
    g_dmf= 2*np.pi*T*k[:]*g_mf[:] 
    # Ausgabe initialisieren 
    x_out = np.zeros([len(r)/n_up*2,1],complex) 
    f = 0 
    # 
    #Schrittweite 
    gamma = 0.01 
    x_buf = np.zeros([len(g_mf)-1],complex) 
    x_1 = complex(0)
    y_buf = np.zeros([len(g_dmf)-1],complex)
    y_1 = complex(0)
    nu = 0 
    phi = 0 
    cnt=0
    # Phasenkorrektur 
    for ii in range(0,len(r)): 
    #Korrektur des Eingangswertes 
        r[ii] = r[ii] * np.exp(1j*(-phi)) 
 
    #neues Phaseninkrement berechnen 
        phi = phi + 2*np.pi*nu/8000

    #Filteroperationen, MF+DMF 
        (x,x_buf)=sig.lfilter(g_mf, np.ones(len(g_mf)),[r[ii]],0,x_buf)        
        (y,y_buf)=sig.lfilter(g_dmf, np.ones(len(g_dmf)),[r[ii]],0,y_buf)
    #Downsample by 8 
        if (np.mod(ii,n_up) == 1): 
    #Fehler berechnen 
                e = 0.5*np.imag(x_1*np.conj(y_1)) + 0.5*np.imag(x*np.conj(y))
                nu= nu + gamma*e[0];      
    #Frequenzoffset berechnen 
                f=f+nu; 
   
        if (np.mod(ii,n_up/2) == 1):     
            x_1 = x
            y_1 = y    
    #Ausgabe, Faktor 2 ueberabgetastet! 
            x_out[cnt] = x
            cnt=cnt+1

    print(f/len(r)*n_up) 
    return x_out 

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

##% Phasenkorrektur 
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



