#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 17:18:38 2017

@author: lena
"""
import numpy as np
def ML_FLL(r,g_mf):
    k = np.arange(np.ceil(-len(g_mf)/2),np.floor(len(g_mf)/2)+1)
    g_dmf= 2*np.pi*T*k[:]*g_mf[:]
    # Ausgabe initialisieren
    x_out = np.zeros([len(r)/4,1])*(1+1j)
    f = 0
    #
    #Schrittweite
    gamma = 0.01
    
    # temp. Variable: Speicherinhalt x-Zweig
    #persistent x_buf
    #if isempty(x_buf)
    #    x_buf = complex(zeros(length(g_mf)-1,1));
    #end
    #
    #persistent x_1
    #if isempty(x_1)
    #    x_1 = complex(0);
    #end
    #
    #% temp. Variable: Speicherinhalt y-Zweig
    #persistent y_buf
    #if isempty(y_buf)
    #    y_buf = complex(zeros(length(g_dmf)-1,1));
    #end
    #
    #persistent y_1
    #if isempty(y_1)
    #    y_1 = complex(0);
    #end
    #
    #% Frequenz
    #persistent nu
    #if isempty(nu)
    #    nu = 0;
    #end
    #
    #% Phaseninkrement
    #persistent phi
    #if isempty(phi)
    #    phi = 0;
    #end
    #
    #% temp. Zaehlervariable
    #cnt=1;
    #
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #% Phasenkorrektur
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #
    for ii in range(1,length(r)+1):
    #    
    #    % Korrektur des Eingangswertes
        r[ii] = r[ii] * np.exp(1j*(-np.pi))
    #    
    #    % neues Phaseninkrement berechnen
        phi = np.pi + 2*np.pi*nu/8000
    #    
    #    % Filteroperationen, MF+DMF
    #    [x,x_buf] = filter(g_mf, 1,r(ii),x_buf);
    #    [y,y_buf] = filter(g_dmf,1,r(ii),y_buf);
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
            x_out(cnt) = x;
            cnt=cnt+1;
    #    end    
    #   
    #end
    #
    #% nur zur Visualisierung
    f=f/len(r)*8;
