# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 23:08:48 2021

@author: dingxu
"""
#https://webda.physics.muni.cz/cgi-bin/ocl_page.cgi?dirname=st02
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

'''
BPRPG = np.loadtxt('BPRPG.txt')
BPRPG = BPRPG.T
BPRPG = BPRPG[BPRPG[:,0]<21]


G = BPRPG[:,0]
BPRP = BPRPG[:,1]

plt.figure(0)
plt.scatter(BPRP, G, marker='o', color='lightcoral',s=5)
plt.xlabel('BP-RP',fontsize=14)
plt.ylabel('Gmag',fontsize=14)
plt.xlim((-1,4))
plt.ylim((1,22))
ax = plt.gca()
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向
'''

temdata = np.loadtxt('temp.dat')
temG = temdata[:,23]
'''
temBPRP = temdata[:,24]-temdata[:,26]

plt.plot(temBPRP+0.53, temG+8.59,'.')
#plt.plot(temBPRP-np.mean(temBPRP), temG-np.mean(temG),'.')


yuanBPRPG = [(BPRP, G)]
npyBPRPG = np.array(yuanBPRPG)[0].T

mBPRPG = [(temdata[:,3],temBPRP+0.53,temG+8.59)]
nmBPRPG = np.array(mBPRPG)[0].T

kdt = cKDTree(nmBPRPG[:,1:])#nmBPRPG[:,1:]
dist, indices = kdt.query(npyBPRPG)

sumtem = 0
for i in range(len(indices)):
    index = indices[i]
    #nmBPRPG[index][0]
    sumtem = sumtem + nmBPRPG[index][0]

print(sumtem)
'''
highdata = np.loadtxt('highdata.txt')
Gmag = highdata[:,5]
Bpmag = highdata[:,6]
Rpmag = highdata[:,7]

temBP = temdata[:,24]
temRP =temdata[:,26]
plt.figure(1)
#plt.plot(Bpmag-Rpmag, Gmag-Bpmag, '.')
plt.plot(temBP-temRP, temG-temBP,'.')


