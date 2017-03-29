#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 17:18:38 2017

@author: lena
"""
import numpy as np
from commpy.utilities import bitarray2dec 

#data-aided ML Approximation (Voraussetzung:f_off<<1/T)
def ML_approx(filter_,r_,T,symbols,ibits,H):
    f_delta=2
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
2
#non data-aided, MPSK

def NDA(r,M,T,H):
    
    summ=np.zeros([r.shape[1],H.shape[1]],complex)
    f_off=np.zeros([r.shape[1],H.shape[1]])
    for j in range(0,H.shape[1]):
        for i in range(0,r.shape[1]):
            for n in range(1,r.shape[0]):
                summ[i,j] += (r[n,i]*H[i,j]*np.conj(r[n-1,i]*H[i,j]))**M
                f_off[i,j]=1/(2*np.pi*T*M)*(np.angle(summ[i,j]))
    return f_off

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def FLL(r,g_mf):
    k = np.arange(np.ceil(-len(g_mf)/2),np.floor(len(g_mf)/2)+1)
    T=1
    g_dmf= 2*np.pi*T*k[:]*g_mf[:]
    # Ausgabe initialisieren
    x_out = np.zeros([len(r)/4,1],complex)
    f = 0
    #
    #Schrittweite
    gamma = 0.01
    
    # temp. Variable: Speicherinhalt x-Zweig
    #persistent x_buf
    #if isempty(x_buf)
    x_buf = np.zeros([len(g_mf)-1,1],complex);
    #end
    #
    #persistent x_1
    #if isempty(x_1)
    x_1 = complex(0);
    #end
    #
    #% temp. Variable: Speicherinhalt y-Zweig
    #persistent y_buf
    #if isempty(y_buf)
    y_buf = np.zeros([len(g_dmf)-1,1],complex);
    #end
    #
    #persistent y_1
    #if isempty(y_1)
    y_1 = complex(0);
    #end
    #
    #% Frequenz
    #persistent nu
    #if isempty(nu)
    nu = 0;
    #end
    #
    #% Phaseninkrement
    #persistent phi
    #if isempty(phi)
    phi = 0;
    #end
    #
    #% temp. Zaehlervariable
    cnt=1;
    #
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #% Phasenkorrektur
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #
    for ii in range(0,len(r)+1):
    #    
    #    % Korrektur des Eingangswertes
        r[ii] = r[ii] * np.exp(1j*(-phi))
    #    
    #    % neues Phaseninkrement berechnen
        phi = phi + 2*np.pi*nu/8000
    #    
    #    % Filteroperationen, MF+DMF
        x=np.convolve(g_mf,r[ii])[:r[ii].size]
        y=np.convolve(g_dmf,r[ii])[:r[ii].size]
        
    #    
    #      
    #    % Downsample by 8
        if (np.mod(ii,8) == 1):
    #        
    #        % Fehler berechnen
            e = 0.5*np.imag(x_1*np.conj(y_1)) + 0.5*np.imag(x*np.conj(y)); 
    #        
    #        % Frequenzoffset berechnen
            nu= nu + gamma*e;     
    #        
    #        % nur zur Visualisierung
            f=f+nu;
    #    end
    #    
    #    % Downsample by 4
        if (np.mod(ii,4) == 1):
    #        
            x_1 = x;
            y_1 = y;
    #        
    #        % Ausgabe, Faktor 2 ueberabgetastet!
            x_out[cnt] = x;
            cnt=cnt+1;
    #    end    
    #   
    #end
    #
    #% nur zur Visualisierung
    print(f/len(r)*8)
    return x_out
