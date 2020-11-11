#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 14:17:26 2020

@author: wayenvan
"""
import numpy as np

class FIR_filter:
    
    def __init__(self, _coefficcients):
        """create a filter"""
        self._ntaps = len(_coefficcients)
        self._coefficcients = _coefficcients
        self._buffer = np.zeros(self._ntaps)
        self._offset = 0 #the current place of x(n)
              
    def dofilter(self, v):
        """dofilter for this """
        output = 0
        self._buffer[self._offset] = v

        # loop to calculate the ring buffer
        for indexH in range(self._ntaps):
            indexX = self._offset - indexH
            if(indexX < 0):
                indexX += self._ntaps     
            output +=self._coefficcients[indexH]*self._buffer[indexX]
        
        #when reach the end, turn the offset into the first position
        self._offset+=1
        if self._offset >= self._ntaps:
            self._offset =self._offset - self._ntaps
            
        return output

def unittest():
    x = np.array([4,5,3])
    h = np.array([5,2])
    output_correct = np.array([20, 33, 25])
    
    output = np.empty(0)
    
    fir_filter = FIR_filter(h)
    
    print("starting unittest")
    print("x value:", x)
    print("h value", h)
    print("correct output", output_correct)
    for value in x:
        output=np.append(output, fir_filter.dofilter(value)) 
    
    print("output from FIR_filter.dofilter:", output)
   
if __name__=="__main__":
    unittest()