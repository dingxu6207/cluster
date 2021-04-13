# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 20:59:11 2021

@author: dingxu
"""

import hdbscan
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import seaborn as sns

sns.set()

data = np.loadtxt('Be31.txt')
print(len(data))
data = data[data[:,2]>0]


X = np.copy(data[:,0:5])


X = StandardScaler().fit_transform(X)
#X = MinMaxScaler().fit_transform(X)
data_zs = np.copy(X)


clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
cluster_labels = clusterer.fit_predict(data_zs)

r1 = pd.Series(cluster_labels).value_counts()

print(r1)

datapro = np.column_stack((data ,cluster_labels))

highdata = datapro[datapro[:,8] == 14]
lowdata = datapro[datapro[:,8] == -1]
meandata = datapro[datapro[:,8] == 7]

plt.figure(10)
sns.kdeplot(highdata[:,3],shade=True)
sns.kdeplot(meandata[:,3],shade=True)
plt.xlabel('pmRA',fontsize=14)

plt.figure(11)
sns.kdeplot(highdata[:,2],shade=True)
sns.kdeplot(meandata[:,2],shade=True)
plt.xlabel('PLX',fontsize=14)

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



