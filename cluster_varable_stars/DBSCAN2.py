# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 21:08:47 2020

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


data = np.loadtxt('Trumpler 5.txt')
print(len(data))
#data = data[data[:,2]>0.25]
#data = data[data[:,2]<0.5]

data = data[data[:,3]<5]
data = data[data[:,3]>-5]
#
data = data[data[:,4]<5]
data = data[data[:,4]>-5]


X = np.copy(data[:,0:5])


X = StandardScaler().fit_transform(X)
#X = MinMaxScaler().fit_transform(X)
data_zs = np.copy(X)

clt = DBSCAN(eps = 0.16, min_samples = 14)
datalables = clt.fit_predict(data_zs)

r1 = pd.Series(datalables).value_counts()

print(r1)

datapro = np.column_stack((data ,datalables))

highdata = datapro[datapro[:,8] == 0]
lowdata = datapro[datapro[:,8] == -1]

np.savetxt('highdata.txt', highdata)
np.savetxt('lowdata.txt',  lowdata)

plt.figure(1)
plt.scatter(lowdata[:,3], lowdata[:,4], marker='o', color='grey',s=5.0)
plt.scatter(highdata[:,3], highdata[:,4], marker='o', color='lightcoral',s=5.0)
pmdata = np.vstack((highdata[:,3], highdata[:,4]))
np.savetxt('PM.txt', pmdata)
plt.xlabel('pmRA',fontsize=14)
plt.ylabel('pmDEC',fontsize=14)
plt.xlim((-15,15))
plt.ylim((-15,15))


plt.figure(2)
hparallax = highdata[:,2]
hGmag = highdata[:,5]

lparallax = lowdata[:,2]
lGmag = lowdata[:,5]

plt.scatter(lGmag, lparallax, marker='o', color='grey',s=5)
plt.scatter(hGmag, hparallax, marker='o', color='lightcoral',s=5)
pallaxdata = np.vstack((lGmag, lparallax))
np.savetxt('parallax.txt', pallaxdata)
plt.xlabel('Gmag',fontsize=14)
plt.ylabel('parallax',fontsize=14)



plt.figure(3)
highdataGmag = highdata[:,5]
highdataBPRP = highdata[:,6]-highdata[:,7]
loaddata = np.vstack((highdataGmag,highdataBPRP))
np.savetxt('BPRPG.txt', loaddata)
plt.xlim((-1,3))
plt.ylim((10,22))
#plt.scatter((lowdata[:,6]-lowdata[:,7]), lowdata[:,5], marker='o', color='grey',s=5)
plt.scatter(highdataBPRP, highdataGmag, marker='o', color='lightcoral',s=5)
x_major_locator = MultipleLocator(1)
plt.xlabel('BP-RP',fontsize=14)
#plt.xlabel('G-RP',fontsize=14)
plt.ylabel('Gmag',fontsize=14)
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_ticks_position('left') #???y???????????????????????????
ax.invert_yaxis() #y?????????


plt.figure(4)
plt.scatter(lowdata[:,0], lowdata[:,1], marker='o', color='grey',s=5)
plt.scatter(highdata[:,0], highdata[:,1], marker='o', color='lightcoral',s=5)
plt.xlabel('RA',fontsize=14)
plt.ylabel('DEC',fontsize=14)



