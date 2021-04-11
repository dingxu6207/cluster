# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 14:07:50 2021

@author: dingxu
"""

import hdbscan
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from matplotlib.pyplot import cm

data = np.loadtxt('Be3140.txt')
print(len(data))

X = np.copy(data[:,0:5])


X = StandardScaler().fit_transform(X)
#X = MinMaxScaler().fit_transform(X)
data_zs = np.copy(X)


clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
cluster_labels = clusterer.fit_predict(data_zs)

r1 = pd.Series(cluster_labels).value_counts()

print(r1)

datapro = np.column_stack((data ,cluster_labels))

highdata = datapro[datapro[:,8] == 22]
lowdata = datapro[datapro[:,8] == -1]
meandata = datapro[datapro[:,8] == 7]

plt.figure(1)
plt.scatter(lowdata[:,3], lowdata[:,4], marker='o', color='grey',s=5.0)
plt.scatter(highdata[:,3], highdata[:,4], marker='o', color='lightcoral',s=5.0)
plt.scatter(meandata[:,3], meandata[:,4], marker='o', color='lightGreen',s=5.0)
pmdata = np.vstack((highdata[:,3], highdata[:,4]))
np.savetxt('PM.txt', pmdata)
plt.xlabel('pmRA',fontsize=14)
plt.ylabel('pmDEC',fontsize=14)
plt.xlim((-25,25))
plt.ylim((-25,25))

plt.figure(2)
hparallax = highdata[:,2]
lparallax = lowdata[:,2]
mparallax = meandata[:,2]

hGmag = highdata[:,5]
lGmag = lowdata[:,5]
mGmag = meandata[:,5]

plt.scatter(lGmag, lparallax, marker='o', color='grey',s=5)
plt.scatter(hGmag, hparallax, marker='o', color='lightcoral',s=5)
plt.scatter(mGmag, mparallax, marker='o', color='lightGreen',s=5)

plt.xlabel('Gmag',fontsize=14)
plt.ylabel('plx',fontsize=14)

plt.figure(3)
highdataGmag = highdata[:,5]
highdataBPRP = highdata[:,6]-highdata[:,7]
loaddata = np.vstack((highdataGmag,highdataBPRP))
np.savetxt('BPRPG1.txt', loaddata)
plt.xlim((-1,4))
#plt.ylim((5,22))
#plt.scatter((lowdata[:,6]-lowdata[:,7]), lowdata[:,5], marker='o', color='grey',s=5)
plt.scatter(highdataBPRP, highdataGmag, marker='o', color='lightcoral',s=5)
plt.scatter(meandata[:,6]-meandata[:,7], meandata[:,5], marker='o', color='lightGreen',s=5)
x_major_locator = MultipleLocator(1)
plt.xlabel('BP-RP',fontsize=14)
#plt.xlabel('G-RP',fontsize=14)
plt.ylabel('Gmag',fontsize=14)
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向


plt.figure(4)
plt.scatter(lowdata[:,0], lowdata[:,1], marker='o', color='grey',s=5.0)
plt.scatter(highdata[:,0], highdata[:,1], marker='o', color='lightcoral',s=5.0)
plt.scatter(meandata[:,0], meandata[:,1], marker='o', color='lightGreen',s=5.0)
plt.xlabel('RA',fontsize=14)
plt.ylabel('DEC',fontsize=14)


lendata = len(r1)
color=cm.rainbow(np.linspace(0,1,lendata))
plt.figure(5)
plt.scatter(lowdata[:,0], lowdata[:,1], marker='o', color='grey',s=1.0)
for index in range(0, lendata):
    #print('it is ok', index)
    pldata = datapro[datapro[:,8] == index]
    plt.scatter(pldata[:,0], pldata[:,1], marker='o', color= color[index],s=5.0)
    
plt.xlabel('RA',fontsize=14)
plt.ylabel('DEC',fontsize=14)


plt.figure(6)
plt.scatter(lGmag, lparallax, marker='o', color='grey',s=5)
for index in range(0, lendata):
    mldata = datapro[datapro[:,8] == index]
    plt.scatter(mldata[:,5], mldata[:,2], marker='o', color = color[index],s=5)
plt.xlabel('Gmag',fontsize=14)
plt.ylabel('plx',fontsize=14)


plt.figure(7)
plt.scatter(lowdata[:,3], lowdata[:,4], marker='o', color='grey',s=5.0)
for index in range(0, lendata):
    zldata = datapro[datapro[:,8] == index]
    plt.scatter(zldata[:,3], zldata[:,4], marker='o', color = color[index],s=5)
plt.xlabel('pmRA',fontsize=14)
plt.ylabel('pmDEC',fontsize=14)
plt.xlim((-25,25))
plt.ylim((-25,25))

plt.figure(8)
plt.scatter((lowdata[:,6]-lowdata[:,7]), lowdata[:,5], marker='o', color='grey',s=5)
for index in range(0, lendata):
    czldata = datapro[datapro[:,8] == index]
    plt.scatter(czldata[:,6]-czldata[:,7], czldata[:,5], marker='o', color = color[index],s=5)
    
plt.xlabel('BP-RP',fontsize=14)
plt.ylabel('Gmag',fontsize=14)
plt.xlim((-1,4))

ax1 = plt.gca()
ax1.xaxis.set_major_locator(x_major_locator)
ax1.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax1.invert_yaxis() #y轴反向

