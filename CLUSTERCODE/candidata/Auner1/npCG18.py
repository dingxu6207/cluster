# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 20:58:08 2021

@author: dingxu
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

cg18 = pd.read_csv('CG18_Auner1.tsv', sep = ';' ,header=None)

cg18data = cg18.iloc[3:,]

df = cg18data.astype('float')

data = df[df.iloc[:,7]>0.9]

print(1000/np.mean(data.iloc[:,2]))
plt.figure(0)

plt.scatter(data.iloc[:,6], data.iloc[:,5], marker='o', color='red',s=5)
ax = plt.gca()
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #
plt.xlabel('BPRP',fontsize=14)
plt.ylabel('G',fontsize=14)
plt.xlim((0,3))

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
