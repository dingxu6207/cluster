# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 20:58:08 2021

@author: dingxu
"""

import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('CG18_31.txt')

datap = data[data[:,7]>0.5]

BPRP = datap[:,6]
G = datap[:,5]

RAmean = np.mean(datap[:,0])
DECmean = np.mean(datap[:,1])

prallax = 1000/(np.mean(datap[:,2]))

arcspc = ((1/60)*prallax)*3.14/180 #1分视场对于PC

dRA = (datap[:,0]-RAmean)*60*arcspc
dDEC = (datap[:,1]-DECmean)*60*arcspc
dprallex = 1000/datap[:,2]

plt.figure(25)

ax1 = plt.axes(projection='3d')
ax1.scatter3D(dRA, dDEC, dprallex, c = 'b', marker='o', s=5)
#ax1.scatter3D(dlRA, dlDEC, dlprallex, c ='r', marker='o', s=0.01)
ax1.set_xlabel('RA')
#ax1.set_xlim(-6, 4)  #拉开坐标轴范围显示投影
ax1.set_ylabel('DEC')
#ax1.set_ylim(-4, 6)
ax1.set_zlabel('Parallax')
ax1.set_zlim(prallax-3400, prallax+3400)



#dbDATA = np.loadtxt('BPRPG.txt')
#dbDATA = dbDATA.T
#dbG = dbDATA[:,0]
#dbBPRP = dbDATA[:,1]
#
#plt.figure(0)
#
#plt.scatter(BPRP, G, marker='o', color='green',s=5)
#plt.scatter(dbBPRP, dbG, marker='o', color='red',s=1)
#
#plt.xlim((-1,4))
#ax = plt.gca()
#ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
#ax.invert_yaxis() #
#plt.xlabel('BPRP',fontsize=14)
#plt.ylabel('G',fontsize=14)
