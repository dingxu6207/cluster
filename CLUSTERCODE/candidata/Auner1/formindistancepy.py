# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 09:47:16 2021

@author: dingxu
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

def distancecompute(selectG, selectBPRP,ydata):
    yuanBPRPG = [(ydata[1,:], ydata[0,:])]
    npyBPRPG = np.array(yuanBPRPG)[0].T

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
    distance = np.sum(d)
    return distance

temp = []
agedata = np.loadtxt('agefeh09.dat')
ydata = np.loadtxt('BPRPG.txt')
for age in np.arange(6.6,10.15,0.01):
    print(str(np.round(age,2))+' it is ok')
    ageGBPRP = agedata[:,[2,28,29,30]]
    selectdata = ageGBPRP[np.round(ageGBPRP[:,0],2)== np.round(age,2)]
    selectG = selectdata[:,1]
    selectBPRP = selectdata[:,2]-selectdata[:,3]
    for E in np.arange(0.5,1,0.01):
        for Mod in np.arange(10,20,0.01):
            selectG = selectG+Mod
            selectBPRP = selectBPRP+E
            distance = distancecompute(selectG, selectBPRP,ydata)
            temparameter = (age, E, Mod,distance)
            temp.append(temparameter)
            #print(distance)
            
            
arraytemp = np.array(temp)
np.savetxt('parameter.txt', arraytemp)

data = arraytemp[arraytemp[:,3].argsort()]
print(data[0,:])

'''
plt.figure(0)
highdataGmag = npyBPRPG[:,1]
highdataBPRP = npyBPRPG[:,0]
loaddata = np.vstack((highdataGmag,highdataBPRP))
np.savetxt('BPRPG.txt', loaddata)
plt.xlim((-1,3))
plt.ylim((10,22))
plt.scatter(highdataBPRP, highdataGmag, marker='o', color='lightcoral',s=5)

plt.xlabel('BP-RP',fontsize=14)
plt.ylabel('Gmag',fontsize=14)
ax = plt.gca()
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向
'''