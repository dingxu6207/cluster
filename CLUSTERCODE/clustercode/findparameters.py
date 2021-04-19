# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 23:42:52 2021

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

#PATH = 'E:\\shunbianyuan\\phometry\\pipelinecode\\cluster\\cluster\\CLUSTERCODE\\clusterdata\\'
#FILE = ['ASCC_12','ASCC_108','ASCC_110']
#
PATH = 'E:\\shunbianyuan\\phometry\\pipelinecode\\cluster\\cluster\\CLUSTERCODE\\clusterdata\\'
#
#filetemp = []
FILE = []
for root, dirs, files in os.walk(PATH):
   for file in files:
       strfile = os.path.join(root, file)
       if (strfile[-4:] == '.txt'):
           print(strfile)
           #filetemp.append(strfile)
           FILE.append(file)

lenfile = len(FILE)

for i in range (lenfile):
    data = np.loadtxt(PATH+FILE[i])
    print(FILE[i])
    
    data = data[data[:,2]>0]
    
#    data = data[data[:,3]<10]
#    data = data[data[:,3]>-10]
#
#    data = data[data[:,4]<10]
#    data = data[data[:,4]>-10]

    X = np.copy(data[:,0:5])
    
    X = StandardScaler().fit_transform(X)
    data_zs = np.copy(X)
    
    res = []
    index = 0
    for eps in np.arange(0.1,0.5,0.01):
        for min_samples in range(6,16):
            
            clt = DBSCAN(eps = eps, min_samples = min_samples)
            datalables = clt.fit_predict(data_zs)
            
            n_clusters = len([i for i in set(datalables)])
            stats = str(pd.Series([i for i in datalables]).value_counts().values)
           
            res.append({'eps':eps,'min_samples':min_samples,'n_clusters':n_clusters,'stats':stats})
            
            index = index+1
            print(str(index)+' '+'it is ok!')
            
    df = pd.DataFrame(res)
    df2cluster = df.loc[df.n_clusters == 2, :]
    df2cluster.to_csv(PATH+FILE[i][:-4]+'.csv')
    
