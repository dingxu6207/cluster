# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 09:47:16 2021

@author: dingxu
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

def distancecompute(selectGO, selectBPRPO,ydatao):
    distance = 0
    ydata = np.copy(ydatao)
    yuanBPRPG = [(ydata[:,1], ydata[:,0])]
    npyBPRPG = np.array(yuanBPRPG)[0].T

    selectBPRP = np.copy(selectBPRPO)
    selectG = np.copy(selectGO)
    mBPRPG = [(selectBPRP,selectG)]
    nmBPRPG = np.array(mBPRPG)[0].T
    
    #print (npyBPRPG)
    #print (nmBPRPG)
    
    kdt = cKDTree(nmBPRPG)#nmBPRPG[:,1:]
    dist, indices = kdt.query(npyBPRPG)
    
    temp = []
    for i in range (len(indices)):
        index = indices[i] 
        temp.append(nmBPRPG[index])
    
    nptemp = np.array(temp)    

    pipeidata = np.column_stack((npyBPRPG, nptemp))
    d1 = (pipeidata[:,0]-pipeidata[:,2])**2
    d2 = (pipeidata[:,1]-pipeidata[:,3])**2
    d = np.sqrt((d1+d2))
    d1 = np.copy(d[d<0.1])
    lend1 = len(d1)
    distance = np.sum(d)
    return distance,lend1

temp = []
agedata = np.loadtxt('agefeh09.dat')

lanli = 0.75
ydata = np.loadtxt('BPRPG31.txt')
ydata = ydata.T
ydata = ydata[ydata[:,1]>lanli]

for age in np.arange(8.6,10.145,0.01):#6.6
    print(str(np.round(age,2))+' it is ok')
    ageGBPRP = agedata[:,[2,28,29,30]]
    selectdata = ageGBPRP[np.round(ageGBPRP[:,0],2)== np.round(age,2)]
    selectG = selectdata[:,1]
    selectBPRP = selectdata[:,2]-selectdata[:,3]
    for E in np.arange(0.3,0.8,0.01): #0.8
        for Mod in np.arange(12,16,0.01):#4-18
            selectGO = selectG+Mod
            selectBPRPO = selectBPRP+E
           
            distance,d1len = distancecompute(selectGO, selectBPRPO,ydata)
            temparameter = (age, E, Mod, distance, d1len)
            temp.append(temparameter)
            #print(E,Mod)    
    #print(d1len)
      
arraytemp = np.array(temp)

#datad = arraytemp[-arraytemp[:,4].argsort()]
#datad = np.copy(datad[0:50,:])
#data = datad[datad[:,3].argsort()]
data = arraytemp[arraytemp[:,3].argsort()]
print(data[0:10,:])
np.savetxt('parameter.txt', data[0:10,:])


