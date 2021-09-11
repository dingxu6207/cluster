# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 23:56:58 2021

@author: dingxu
"""

import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

VSXdata = pd.read_csv('VSXTEST.tsv', sep = ';', encoding='gbk')
datara = VSXdata['RAJ2000']
datadec = VSXdata['DEJ2000']

#print(VSXdata.iloc[1,1])
 
listdatara = datara.tolist()
listdatadec = datadec.tolist()

radectemp = []
for i in range(1,len(listdatara)):
    RA = np.float32(listdatara[i])
    DEC = np.float32(listdatadec[i])
    
    radectemp.append(RA)
    radectemp.append(DEC)
    
VSXradec = np.float32(radectemp).reshape(-1,2)

data1 = np.array([[2,3],[4,5],[7,8]])

print('it is ok1')
kdt = cKDTree(VSXradec)
dist, indices = kdt.query(data1)


#df4 = [VSXdata, VSXdata]
#
#result = pd.concat(df4, axis=1)