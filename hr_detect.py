#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 18:23:05 2020

@author: wayenvan
"""

import numpy as np
import matplotlib.pyplot as plt
import fir_filter as fir
from ecg_gudb_database import GUDb

def heartRateCalculate(peaks, FS):
    ret = np.empty(0)
    for index, value in np.ndenumerate(peaks):
        if(index[0]==0):
            continue
        N = value-peaks[index[0]-1]
        ret=np.append(ret, 60//(N*(1/FS)))
    return ret
        
def heartRateDetect(data, preFilter: fir.FIR_filter, matchedFilter: fir.FIR_filter):
    """detect heart rate in by filters"""
    Fs = 250
    threshold = 2e-11
    Nlimited = 100
    
    template = np.empty(0)
    
    #reference variable
    preFilterResult = np.empty(0)
    matchedResult = np.empty(0)
    heartRateResult = np.empty(0)
    R_peakPoint = np.zeros(len(data))
    
    for index, value in np.ndenumerate(data):
        preFltValue = preFilter.dofilter(value)
        matchedValue = matchedFilter.dofilter(preFltValue)
        matchedValue *= matchedValue
        
        template = np.append(template, matchedValue)
        preFilterResult = np.append(preFilterResult, preFltValue)
        matchedResult = np.append(matchedResult, matchedValue)
        
        N = len(template)
        if(N>Nlimited):
            if((template[N-1]>threshold)&(template[N-2]<threshold)):
                heartRate =60//(N*(1/Fs))
                print(heartRate)
                heartRateResult = np.append(heartRateResult, heartRate)
                R_peakPoint[index] = 1
                
                #clear template
                template = np.empty(0)
    
    return (preFilterResult,
            matchedResult,
            heartRateResult,
            R_peakPoint)
        
"""create a matched filter"""
#pre filtering
ecgData = np.genfromtxt('ECG_msc_matric_4.dat',
                     dtype=None)
N = len(ecgData)
Fs = 250
M = 250

k0 = int(5/Fs * M)
k1 = int(45/Fs * M)
k2 = int(55/Fs * M)

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

fir_filter = fir.FIR_filter(h)
ecgDataFiltered = np.empty(0)
for value in ecgData:
    ecgDataFiltered = np.append(ecgDataFiltered, fir_filter.dofilter(value))
    
#create a matched filter
matchedCore = ecgDataFiltered[906:1012]
matchedCore = matchedCore[::-1]

#test matchedFilter 
# matchedData = np.empty(0)
# for value in ecgDataFiltered:
#     matchedData = np.append(matchedData, matchedFilter.dofilter(value))
# matchedData = matchedData * matchedData

"""detect heart rate"""
ecg_class = GUDb(14, "walking")
matchedFilter = fir.FIR_filter(matchedCore)
preFilter = fir.FIR_filter(h)

result = heartRateDetect(ecg_class.einthoven_III, preFilter, matchedFilter)
tresult = heartRateCalculate(ecg_class.anno_cs, 250)


"""plot figures"""