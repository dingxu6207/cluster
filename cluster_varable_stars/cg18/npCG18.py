# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 20:58:08 2021

@author: dingxu
"""

import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('Trumpler 5.txt')

datap = data[data[:,7]>0.5]

BPRP = datap[:,6]
G = datap[:,5]


plt.figure(0)

plt.scatter(BPRP, G, marker='o', color='green',s=5)
plt.title('Auner_1', fontsize=14)
plt.xlim((-1,4))
ax = plt.gca()
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #
plt.xlabel('G(BP-RP)',fontsize=14)
plt.ylabel('Gmag',fontsize=14)
plt.savefig('Auner_1.png')
