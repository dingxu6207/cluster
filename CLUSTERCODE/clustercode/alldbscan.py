# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 20:59:16 2021

@author: dingxu
"""

import numpy as np
from sklearn.cluster import KMeans,DBSCAN,AgglomerativeClustering
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import mixture
from matplotlib.pyplot import MultipleLocator
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
import os
import imageio

PATH = 'E:\\shunbianyuan\\phometry\\pipelinecode\\cluster\\cluster\\CLUSTERCODE\\clusterdata\\'

def findepsmindata(filemeter):
    df = pd.read_csv(PATH+filemeter)

    stasdata = df[['eps','min_samples','stats']]


    hang,lie = stasdata.shape

    for i in range(hang-1):
        member1 = int(df['stats'][i][-6:-1])
        member2 = int(df['stats'][i+1][-6:-1])
        cha = member2-member1
    
        if cha >500:
            print(i)
            eps = stasdata.iloc[i,0]
            mindata = stasdata.iloc[i,1]
        
            print(np.round(eps,2),mindata)
            break
        else:
            eps = stasdata.iloc[i+1,0]
            mindata = stasdata.iloc[i+1,1]
            print(np.round(eps,2),mindata)
    return np.round(eps,2),mindata


FILE = []
for root, dirs, files in os.walk(PATH):
   for file in files:
       strfile = os.path.join(root, file)
       if (strfile[-4:] == '.txt'):
           print(strfile)
           #filetemp.append(strfile)
           FILE.append(file)

lenfile = len(FILE)
gif_images = []
for i in range (lenfile):
    data = np.loadtxt(PATH+FILE[i])
    print(FILE[i])
    
    data = data[data[:,2]>0]

    X = np.copy(data[:,0:5])
    
    X = StandardScaler().fit_transform(X)
    data_zs = np.copy(X)
    eps,min_samples = findepsmindata(FILE[i][:-4]+'.csv')
    clt = DBSCAN(eps = eps, min_samples = min_samples)
    datalables = clt.fit_predict(data_zs)
    
    datapro = np.column_stack((data ,datalables))
    highdata = datapro[datapro[:,8] == 0]
    lowdata = datapro[datapro[:,8] == -1]
    
    plt.clf()
    plt.figure(1)
    highdataGmag = highdata[:,5]
    highdataBPRP = highdata[:,6]-highdata[:,7]
    loaddata = np.vstack((highdataGmag,highdataBPRP))
    np.savetxt(PATH+FILE[i][:-4]+'BPRPG.dat', loaddata)
    plt.xlim((-1,4))
    plt.ylim((10,22))
    plt.scatter((lowdata[:,6]-lowdata[:,7]), lowdata[:,5], marker='o', color='grey',s=5)
    plt.scatter(highdataBPRP, highdataGmag, marker='o', color='lightcoral',s=5)
    plt.xlabel('BP-RP',fontsize=14)
    plt.ylabel('Gmag',fontsize=14)
    plt.title(FILE[i][:-4])
    ax = plt.gca()
    ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
    ax.invert_yaxis() #y轴反向
    plt.pause(1)
  
    plt.savefig(PATH+FILE[i][:-4]+'.jpg')
    gif_images.append(imageio.imread(PATH+FILE[i][:-4]+'.jpg'))
    imageio.mimsave(PATH+"test.gif",gif_images,fps=0.5)
    