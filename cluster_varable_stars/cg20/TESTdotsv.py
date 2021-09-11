# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 23:56:58 2021

@author: dingxu
"""

import pandas as pd
import numpy as np
from scipy.spatial import cKDTree


file = 'CG20star.dat'
cg20data = pd.read_csv(file, sep='\s+', error_bad_lines=False) #设置列名和 默认以多个空格来识别数据
cg20radec = cg20data[['RAdeg','DEdeg']]
clustername = cg20data['Cluster']
npcg20radec = np.array(cg20radec)
radecname = np.array(clustername)
listname = clustername.tolist()
npname = np.array(listname)
radecname = np.column_stack((npcg20radec, npname))

VSXdata = pd.read_csv('VSX.tsv', sep = ';', encoding='gbk')
VSXdata = VSXdata.drop(labels=0)
VSXdata.index = range(len(VSXdata))
datara = VSXdata['RAJ2000']
datadec = VSXdata['DEJ2000']
#print(VSXdata.iloc[1,1])
 
listdatara = datara.tolist()
listdatadec = datadec.tolist()

radectemp = []
for i in range(0,len(listdatara)):
    RA = np.float32(listdatara[i])
    DEC = np.float32(listdatadec[i])
    
    radectemp.append(RA)
    radectemp.append(DEC)
    
VSXradec = np.float32(radectemp).reshape(-1,2)

kdt = cKDTree(radecname[:,0:2])
dist, indices = kdt.query(VSXradec)

temp = []
for i in range (len(indices)):
    index = indices[i] 
    temp.append(radecname[index])
nptemp = np.array(temp)  

VSXradec = np.column_stack((dist, VSXradec))
allradec = np.column_stack((VSXradec, nptemp))

lista = ['distance','VSXRA', 'VSXDEC', 'cg20RA', 'cg20DEC', 'cg20name']
dfallradec = pd.DataFrame(allradec, columns= lista)

df4 = [VSXdata, dfallradec]
resultsort = pd.concat(df4, axis=1)
#resultsort = result.sort_values(by='distance')
#resultsort = result.sort_values('cg20name')
print(str(len(resultsort.iloc[0,1]))+resultsort.iloc[0,1])
DataEW = resultsort[resultsort.iloc[:,1] == 'EW'+' '.join(' 'for i in range(14)) +' '] #14=(30-2)/2

DataEW = DataEW[DataEW.iloc[:,7].astype(np.float)<0.01]
resultsort.to_csv('allvsx.csv', index=0)
DataEW.to_csv('EW.csv', index=0)

#DataEW = resultsort[resultsort.iloc[:,1] == 'EW                            ']