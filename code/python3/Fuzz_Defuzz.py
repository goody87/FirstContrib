# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 23:57:41 2018

@author: enggh
"""

import numpy as np
import skfuzzy as fuzz

def getMaxIndices(lst):
    res = []
    maxVal = np.max(lst)
    for i in range(0,len(lst)):
        if lst[i] == maxVal:
            res.append(i)
    return res

def getFuzzyRep(arr):
    fuzzRep = ""
    fuzztot = 0
    x_qual = np.arange(0, 11, 0.1)
    qual_lo = fuzz.trimf(x_qual, [0, 0, 0.5])
    qual_md = fuzz.trimf(x_qual, [0, 0.5, 1.0])
    qual_hi = fuzz.trimf(x_qual, [0.5, 1.0, 1.0])
    FuzzVals=["Low","Medium","High"]
    i =0
    for val in arr:
        
        tmp = FuzzVals[np.argmax([fuzz.interp_membership(x_qual, qual_lo, val),fuzz.interp_membership(x_qual, qual_md, val),fuzz.interp_membership(x_qual, qual_hi, val)])]
        
        if i == 0:
            fuzzRep = tmp
        else:
            fuzzRep = fuzzRep + "," + tmp
        
        if tmp == "Low":
            fuzztot += 1
        elif tmp == "Medium":
            fuzztot += 2
        else:
            fuzztot += 3
                
        i+=1
    return fuzzRep, fuzztot 
    

x_qual = np.arange(0, 11, 0.1)

qual_lo = fuzz.trimf(x_qual, [0, 0, 0.5])
qual_md = fuzz.trimf(x_qual, [0, 0.5, 1.0])
qual_hi = fuzz.trimf(x_qual, [0.5, 1.0, 1.0])

valToFuzz = 0.547
fuzzarr = []
np.random.seed(777)
tst = np.random.uniform(0.0,1.0,10)

qual_level_lo = fuzz.interp_membership(x_qual, qual_lo, valToFuzz)
fuzzarr.append(qual_level_lo)
qual_level_md = fuzz.interp_membership(x_qual, qual_md, valToFuzz)
fuzzarr.append(qual_level_md)
qual_level_hi = fuzz.interp_membership(x_qual, qual_hi, valToFuzz)
fuzzarr.append(qual_level_hi)

print (qual_level_lo)
print (qual_level_md)
print (qual_level_hi)

print (getMaxIndices(fuzzarr))

frep, ftot = getFuzzyRep(tst)
print(frep)
print ("total is: "+ str(ftot));

            