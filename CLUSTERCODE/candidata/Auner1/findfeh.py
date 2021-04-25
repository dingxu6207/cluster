# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 14:41:15 2021

@author: dingxu
"""

import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

agedata = np.loadtxt('FeH.dat')
ageGBPRP = agedata[:,[1,28,29,30]]


plt.figure(0)
selectdata = ageGBPRP[np.round(ageGBPRP[:,0],1)==-0.7]
selectG = selectdata[:,1]
selectBPRP = selectdata[:,2]-selectdata[:,3]
plt.scatter(selectBPRP, selectG, marker='o', color='lightcoral',s=5)

selectdata = ageGBPRP[np.round(ageGBPRP[:,0],1)== 0]
selectG = selectdata[:,1]
selectBPRP = selectdata[:,2]-selectdata[:,3]
plt.scatter(selectBPRP, selectG, marker='o', color='green',s=5)

ax = plt.gca()
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向

plt.xlabel('BP-RP',fontsize=14)
plt.ylabel('Gmag',fontsize=14)
