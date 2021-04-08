# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 22:45:17 2021

@author: dingxu
"""

import matplotlib.pyplot as plt
import numpy as np

BPRPG = np.loadtxt('BPRPG.txt')
BPRPG = BPRPG.T

Gmag = BPRPG[:,0]
BPRP = BPRPG[:,1]

BPRPG1 = np.loadtxt('BPRPG1.txt')
BPRPG1 = BPRPG1.T

Gmag1 = BPRPG1[:,0]
BPRP1 = BPRPG1[:,1]


plt.scatter(BPRP1, Gmag1, marker='o', color='lightcoral',s=5)
plt.scatter(BPRP, Gmag, marker='o', color='green',s=5)
plt.xlim((-1,4))
ax = plt.gca()
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis()