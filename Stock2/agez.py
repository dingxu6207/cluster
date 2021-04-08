# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 23:08:48 2021

@author: dingxu
"""
#https://webda.physics.muni.cz/cgi-bin/ocl_page.cgi?dirname=st02
import numpy as np
import matplotlib.pyplot as plt

BPRPG = np.loadtxt('BPRPG.txt')

G = BPRPG[0,:]
BPRP = BPRPG[1,:]

plt.figure(0)
plt.scatter(BPRP, G, marker='o', color='lightcoral',s=5)
plt.xlabel('BP-RP',fontsize=14)
plt.ylabel('Gmag',fontsize=14)
plt.xlim((-1,4))
plt.ylim((1,22))
ax = plt.gca()
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向

temdata = np.loadtxt('tmp.txt')
temG = temdata[:,23]
temBPRP = temdata[:,24]-temdata[:,26]

plt.plot(temBPRP+0.53, temG+8.59,'.')

