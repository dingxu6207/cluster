# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 22:00:50 2021

@author: dingxu
"""

import numpy as np
import matplotlib.pyplot as plt

PATH = 'E:\\shunbianyuan\\phometry\\pipelinecode\\cluster\\cluster\\targets\\'
file = 'BH_67BPRPG.dat'
data = np.loadtxt(PATH+file)

BPRP = data[1,:]
G = data[0,:]



agedata = np.loadtxt('kuagefeh.txt')
fehageGBPRP = agedata[:,[1,2,28,29,30]]

Age = 9.45 #9.69
feh = -0.2
E = 0.6 #0.73
mM = 15.3 #15.5
plt.figure(1)
#selectdata = ageGBPRP[np.round(ageGBPRP[:,0],2)== Age]
selectdata = fehageGBPRP[np.round(fehageGBPRP[:,1],2)== np.round(Age,2)]
selectdata = selectdata[np.round(selectdata[:,0],1)== np.round(feh,1)]
        
selectG = selectdata[:,2]
selectBPRP = selectdata[:,3]-selectdata[:,4]



plt.figure(0)

plt.xlim((-1,3))
plt.ylim((10,22))

plt.scatter(BPRP, G, marker='o', color='green',s=5)
plt.scatter(selectBPRP+E, selectG+mM, marker='o', color='lightcoral',s=5)

plt.xlabel('BP-RP',fontsize=14)
plt.ylabel('Gmag',fontsize=14)
ax = plt.gca()
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向