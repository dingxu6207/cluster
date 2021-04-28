# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 15:08:49 2021

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
from scipy.spatial import cKDTree

data = np.loadtxt('Auner_1.txt')
print(len(data))
data = data[data[:,2]>0]

X = np.copy(data[:,0:5])


X = StandardScaler().fit_transform(X)
data_zs = np.copy(X)

clt = DBSCAN(eps = 0.17, min_samples = 15)
datalables = clt.fit_predict(data_zs)

r1 = pd.Series(datalables).value_counts()
print(r1)

datapro = np.column_stack((data ,datalables))

highdata = datapro[datapro[:,8] == 0]
lowdata = datapro[datapro[:,8] == -1]

plt.figure(0)
highdataGmag = highdata[:,5]
highdataBPRP = highdata[:,6]-highdata[:,7]
loaddata = np.vstack((highdataGmag,highdataBPRP))
np.savetxt('BPRPG.txt', loaddata)
plt.xlim((-1,3))
plt.ylim((10,22))
plt.scatter((lowdata[:,6]-lowdata[:,7]), lowdata[:,5], marker='o', color='grey',s=5)
plt.scatter(highdataBPRP, highdataGmag, marker='o', color='lightcoral',s=5)
x_major_locator = MultipleLocator(1)
plt.xlabel('BP-RP',fontsize=14)
plt.ylabel('Gmag',fontsize=14)
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向

agedata = np.loadtxt('kuagefeh.txt')
fehageGBPRP = agedata[:,[1,2,28,29,30]]

Age = 9.69 #9.69
feh = -0.9
plt.figure(1)
#selectdata = ageGBPRP[np.round(ageGBPRP[:,0],2)== Age]
selectdata = fehageGBPRP[np.round(fehageGBPRP[:,1],2)== np.round(Age,2)]
selectdata = selectdata[np.round(selectdata[:,0],1)== np.round(feh,1)]
        
selectG = selectdata[:,2]
selectBPRP = selectdata[:,3]-selectdata[:,4]
plt.scatter(selectBPRP, selectG, marker='o', color='lightcoral',s=5)

ax = plt.gca()
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向

plt.xlabel('BP-RP',fontsize=14)
plt.ylabel('Gmag',fontsize=14)

E = 0.69 #0.73
DEL = 0
mM = 15.43 #15.5
#mM = 5*np.log10(1000/np.mean(highdata[:,2]))+5-2.046*E-DEL
plt.figure(2)
plt.scatter(highdataBPRP, highdataGmag, marker='o', color='lightcoral',s=5)
plt.scatter(selectBPRP+E, selectG+mM, marker='o',color='green',s=1)
plt.ylim((10,21))
plt.xlim((0,3))
ax = plt.gca()
ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向
plt.xlabel('BPRP',fontsize=14)
plt.ylabel('G',fontsize=14)

######################################
yuanBPRPG = [(highdataBPRP, highdataGmag)]
npyBPRPG = np.array(yuanBPRPG)[0].T

mBPRPG = [(selectBPRP+E,selectG+mM)]
nmBPRPG = np.array(mBPRPG)[0].T

kdt = cKDTree(nmBPRPG)#nmBPRPG[:,1:]
dist, indices = kdt.query(npyBPRPG)

temp = []
for i in range (len(indices)):
    index = indices[i] 
    temp.append(nmBPRPG[index])
    
nptemp = np.array(temp)    

pipeidata = np.column_stack((npyBPRPG, nptemp))
d1 = (pipeidata[:,0]-pipeidata[:,2])**2
d2 = (pipeidata[:,1]-pipeidata[:,3])**2
d = np.sqrt(d1+d2)
print(np.sum(d))

EBV = 0.396
AV = 3.1*(EBV)
AG = 0.9761-0.1704*E+0.0086*(E**2)+0.0011*(E**3)-0.0438*AV+0.0013*(AV**2)+0.0099*E*AV
#(m-M) = 5logr-5+A
distanr = 10**((mM+5-AG)/5)
print(distanr)