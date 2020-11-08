#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 22:39:50 2020

@author: wayenvan
"""

import numpy as np
import fir_filter as fir
import matplotlib.pyplot as plt

"""define functions"""
def generateXf(sampleRate, N):
    """generateXf for frequeny domain"""
    return np.linspace(0.0, (N-1)*sampleRate/N, N)

def generateXt(sampleRate, N):
    """generateXt for time domain"""
    return np.linspace(0.0, (N-1)*1/sampleRate, N)

"""main function"""
ecgData = np.genfromtxt('ECG_msc_matric_4.dat',
                     dtype=None)
N = len(ecgData)
Fs = 250
M = 250

k0 = int(5/Fs * M)
k1 = int(45/Fs * M)
k2 = int(55/Fs * M)

#generate filter coefficiencies
H = np.ones(M)

H[0:k0+1] = 0
H[M-k0-1:M] = 0
H[k1:k2+1] = 0
H[M-k2-1:M-k1] = 0

htemp = np.fft.ifft(H)
htemp = np.real(htemp)

h = np.zeros(M)

h[0:int(M/2)] = htemp[int(M/2):M]
h[int(M/2):M] = htemp[0:int(M/2)]
h = h*np.hamming(M)

#feed values into filter
fir_filter = fir.FIR_filter(h)
ecgDataFiltered = np.empty(0)
for value in ecgData:
    ecgDataFiltered = np.append(ecgDataFiltered, fir_filter.dofilter(value))


"""plot figures"""
plt.figure(figsize=(20,10))
plt.plot(generateXt(Fs, N), ecgData)
plt.title("original signal")
plt.xlabel("time(s)")
plt.ylabel("amplitude")
plt.savefig("./Figures/ecgDataOriginal.pdf")

# plt.figure(figsize=(20,10))
# plt.plot(generateXf(Fs, M),H)
# plt.xlabel("frequency(Hz)")
# plt.ylabel("amplitude")
# plt.savefig("./Figures/H.pdf")

# plt.figure(figsize=(20,10))
# plt.plot(h)
# plt.xlabel("n")
# plt.ylabel("amplitude")
# plt.savefig("./Figures/h.pdf")

plt.figure(figsize=(20,10))
plt.plot(generateXt(Fs, N), ecgDataFiltered)
plt.title("filtered signal")
plt.xlabel("time(s)")
plt.ylabel("amplitude")
plt.savefig("./Figures/ecgDataFiltered.pdf")