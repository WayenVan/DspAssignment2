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
    """this function calculate the heart rate when we get peak sequencies, just for  verifying"""
    ret = np.empty(0)
    for index, value in np.ndenumerate(peaks):
        if(index[0]==0):
            continue
        N = value-peaks[index[0]-1]
        ret=np.append(ret, 60//(N*(1/FS)))
    return ret
        
def heartRateDetect(data, preFilter: fir.FIR_filter, matchedFilter: fir.FIR_filter):
    """detect heart rate in by filters
    
    """
    Fs = 250
    threshold = 2e-11
    Nlimited = 100
    
    template = np.empty(0)
    
    #reference variable to recorld some information
    preFilterResult = np.empty(0)
    matchedResult = np.empty(0)
    heartRateResult = np.empty(0)
    R_peakPoint = np.empty(0)
    
    for index, value in np.ndenumerate(data):
        
        #do prefilter
        preFltValue = preFilter.dofilter(value)
        #do match filter
        matchedValue = matchedFilter.dofilter(preFltValue)
        #increase S/N ratio
        matchedValue *= matchedValue
        
        #recordding gap
        template = np.append(template, matchedValue)
        
        #record other data for verifying
        preFilterResult = np.append(preFilterResult, preFltValue)
        matchedResult = np.append(matchedResult, matchedValue)
        
        N = len(template)
        if(N>Nlimited):
            if((template[N-1]>threshold)&(template[N-2]<threshold)):
                #calculate the heartRate
                heartRate =60//(N*(1/Fs))
                #calculate the time when beat occur
                time = index[0]*(1/Fs)
                print("in",round(time, 2),"s:", heartRate, "Bpm")
                heartRateResult = np.append(heartRateResult, heartRate)
                R_peakPoint = np.append(R_peakPoint, index[0])
                
                #clear template
                template = np.empty(0)
    
    return (preFilterResult,
            matchedResult,
            heartRateResult,
            R_peakPoint)
        
"""2.1 create a matched filter"""

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
    
#create a matched filter coefficients
matchedCore = ecgDataFiltered[906:1012]
matchedCore = matchedCore[::-1]
templateRPosition = 21

#begin create matchedFilter and find peaks
matchedDataResult = np.empty(0)
preFilterResult = np.empty(0)
peakPosition = np.empty(0)
matchedFilter1 = fir.FIR_filter(matchedCore)
preFilter1 = fir.FIR_filter(h) #clear fir_filter

#find peaks
for index, value in np.ndenumerate(ecgData):
    prefilteredData = preFilter1.dofilter(value)
    #record prefilter result
    preFilterResult = np.append(preFilterResult, prefilteredData)
    #record matchfilter result
    matchedData = matchedFilter1.dofilter(prefilteredData)**2 #reduce S/N ratio
    matchedDataResult = np.append(matchedDataResult, matchedData)
    
    #define threshold
    threshold = 0.65e-11
    N = len(matchedDataResult)
    #when the value exceed the threshold while the previous one didn't
    if((matchedDataResult[N-1]>threshold) & (matchedDataResult[N-2] > threshold)):
        peakPosition = np.append(peakPosition, index[0])

print("peakFinding in 2.1:", peakPosition)
    
"""2.2 detect heart rate"""
#create corresponding filters
ecg_class = GUDb(14, "walking")
matchedFilter = fir.FIR_filter(matchedCore)
preFilter = fir.FIR_filter(h)

result = heartRateDetect(ecg_class.einthoven_III, preFilter, matchedFilter)

#compare with the original signal
tresult = heartRateCalculate(ecg_class.anno_cs, 250)


"""plot figures"""

#2.1
# plt.figure(figsize=(20,10))
# plt.title("template of single beat (QRST)")
# plt.plot(matchedCore[::-1])
# plt.xlabel("N")
# plt.ylabel("amplitude")
# plt.savefig("./Figures/matchCore.pdf")

# a = np.zeros(len(ecgData))
# a[peakPosition.astype('int')] = 1
# plt.figure(figsize=(20,10))
# plt.title("match filter result")
# plt.plot(matchedDataResult,label="match filter result")
# plt.plot(a*matchedDataResult, label="peak position")
# plt.legend()
# plt.xlabel("N")
# plt.ylabel("amplitude")
# plt.ylim(0.0e-10, 1.5e-11)
# plt.savefig("./Figures/matchResult.pdf")

# plt.figure(figsize=(20,10))
# plt.plot(result[3]*1/250, result[2])
# plt.title("detected heart rate")
# plt.ylabel("heart rate(BPM)")
# plt.xlabel("time(s)")
# plt.savefig("./Figures/heartRate.pdf")
