#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 16:00:06 2017

@author: lena
"""
import numpy as np
import matplotlib.pyplot as plt 

L = 128   # Anzahl Punkte in der DFT
pn1 = np.random.choice([-np.sqrt(2), np.sqrt(2)], L//2) # erste PN Sequenz -sqrt(2) oder +sqrt(2)
pn2 = np.random.choice([-1, 1], L)  # zweite PN Sequenz -1 oder +1
s1=np.concatenate((pn1,pn1))
s2=pn2
s = np.concatenate((s1, s2))


#f = np.linspace(-0.5, 0.5, L, endpoint=False)
#S1 = np.abs(np.fft.fftshift(np.fft.fft(s[:L])))**2
#plt.plot(f, np.abs(S1))
#plt.title("|S1|"); plt.xlabel("Normierte Frequenz"); plt.xlim([-0.5, 0.5])
#plt.show()
#
#S2 = np.abs(np.fft.fftshift(np.fft.fft(s[L+2*n_CP:])))**2
#plt.plot(f, np.abs(S2))
#plt.title("|S2|"); plt.xlabel("Normierte Frequenz"); plt.xlim([-0.5, 0.5])
#plt.show()
f_off=23.7
t_off=66



r =s

# Frequenzoffset
r_off= r*np.exp(1j*2*np.pi*f_off*np.arange(len(r))/L)

# Zeitoffset
r = np.concatenate((np.zeros((t_off,), dtype=complex), r_off))


Pd = np.asarray([np.sum(np.conj(r[i:i+L//2])*r[i+L//2:i+L]) for i in range(len(r)-L)])
Rd = np.asarray([np.sum(np.abs(r[i+L//2:i+L])**2) for i in range(len(r) - L)])
M = np.abs(Pd/Rd)**2
print(np.argmax(M[60:]))



plt.plot(np.abs(M)); plt.ylabel("M"); plt.xlabel("d [Samples]"); plt.xlim([0, len(M)]); plt.ylim([0,1.1]); plt.title("M(d)"); plt.show()
plt.plot(np.angle(Pd)); plt.title("arg(P(d))"); plt.xlabel("Verschiebung"); plt.xlim([0, len(M)]); plt.show()
#OFDM symbole erzeugen