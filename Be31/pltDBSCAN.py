# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 13:53:46 2021

@author: dingxu
"""

import numpy as np
from sklearn.cluster import KMeans,DBSCAN,AgglomerativeClustering
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import mixture
from matplotlib.pyplot import MultipleLocator
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import imageio


data = np.loadtxt('Be31.txt')
print(len(data))
data = data[data[:,2]>0]

data = data[data[:,3]<15]
data = data[data[:,3]>-15]

data = data[data[:,4]<15]
data = data[data[:,4]>-15]

#data = data[data[:,5]<18]
X = np.copy(data[:,0:5])


X = StandardScaler().fit_transform(X)
#X = MinMaxScaler().fit_transform(X)
data_zs = np.copy(X)

clt = DBSCAN(eps = 0.25, min_samples = 14)
datalables = clt.fit_predict(data_zs)

r1 = pd.Series(datalables).value_counts()

print(r1)

datapro = np.column_stack((data ,datalables))

highdata = datapro[datapro[:,8] == 0]
lowdata = datapro[datapro[:,8] == -1]

np.savetxt('highdata.txt', highdata)
np.savetxt('lowdata.txt',  lowdata)

RAmean = np.mean(highdata[:,0])
DECmean = np.mean(highdata[:,1])

plt.figure(5)
plt.scatter(lowdata[:,0], lowdata[:,1], marker='o', color='grey',s=5.0)
plt.scatter(highdata[:,0], highdata[:,1], marker='o', color='lightcoral',s=5.0)
plt.plot(RAmean, DECmean, '.')
plt.xlabel('RA',fontsize=14)
plt.ylabel('DEC',fontsize=14)

prallax = 1000/(np.mean(highdata[:,2]))

arcspc = ((1/60)*prallax)*3.14/180 #1分视场对于PC
print(arcspc)

dRA = (highdata[:,0]-RAmean)*60*arcspc
dDEC = (highdata[:,1]-DECmean)*60*arcspc
dprallex = 1000/highdata[:,2]

dlRA = (lowdata[:,0]-RAmean)*60*arcspc
dlDEC = (lowdata[:,1]-DECmean)*60*arcspc
dlprallex = 1000/lowdata[:,2]

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


