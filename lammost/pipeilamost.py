# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 20:59:48 2021

@author: dingxu
"""
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

cg18data = pd.read_csv('table1.dat', sep = '\s+' ,header=None)
cg18 = pd.read_csv('table1.dat', sep = '\s+' ,header=None, usecols=[1, 2])

npcg18 = np.array(cg18)


lmstcsv = pd.read_csv('testcsv.csv', sep = ',', usecols=[1, 2])

nplast = np.array(lmstcsv)

#cg18对应lamost数据
kdt = cKDTree(lmstcsv)
dist, indices = kdt.query(npcg18)


temp = []
for i in range (len(indices)):
    index = indices[i]
    temp.append(nplast[index])
    
nptemp = np.array(temp)    

allradec = np.column_stack((npcg18, nptemp))

indexdata = np.arange(0,npcg18.shape[0])

allradec = np.column_stack((indexdata, allradec))



