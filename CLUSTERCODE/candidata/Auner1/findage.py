# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 11:23:57 2021

@author: dingxu
"""
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

agedata = np.loadtxt('Age.dat')
ageGBPRP = agedata[:,[2,28,29,30]]


plt.figure(0)
selectdata = ageGBPRP[np.round(ageGBPRP[:,0],1)==8.6]
selectG = selectdata[:,1]
selectBPRP = selectdata[:,2]-selectdata[:,3]
plt.scatter(selectBPRP, selectG, marker='o', color='lightcoral',s=5)

selectdata = ageGBPRP[np.round(ageGBPRP[:,0],1)==10]
selectG = selectdata[:,1]
selectBPRP = selectdata[:,2]-selectdata[:,3]
plt.scatter(selectBPRP, selectG, marker='o', color='green',s=5)

ax = plt.gca()
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向

plt.xlabel('BP-RP',fontsize=14)
plt.ylabel('Gmag',fontsize=14)

#lendata = np.int((np.max(ageGBPRP[:,0])-np.min(ageGBPRP[:,0])))*10+1
#color=cm.rainbow(np.linspace(0,1,lendata))
#index = -2
#for i in np.arange(6.1,10.2,0.1):
#    index = index+1
#    selectdata = ageGBPRP[ageGBPRP[:,0]==np.round(i,1)]
#    selectG = selectdata[:,1]
#    selectBPRP = selectdata[:,2]-selectdata[:,3]
#    plt.scatter(selectBPRP, selectG, marker='o', color=color[index],s=5)
#    ax = plt.gca()
#    ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
#    ax.invert_yaxis() #y轴反向
#    