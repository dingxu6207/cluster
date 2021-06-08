# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 21:35:24 2021

@author: dingxu
"""

from astropy.coordinates import SkyCoord 
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
import imageio  
     
#VSX的星表数据           
#VSXdata = pd.read_csv('NGC559.csv')
VSXdata = pd.read_csv('Trumpler 5.csv')
dataradec = VSXdata['Coords']
vsxtype = VSXdata['Type']

listdata = dataradec.tolist()
print(listdata[0])

radectemp = []
for i in range(len(listdata)):
    RA = listdata[i][0:2]+'h'+ listdata[i][3:5] +'m'+listdata[i][6:11]+'s'
    DEC = listdata[i][12:15]+'d'+ listdata[i][16:18] +'m'+listdata[i][19:23]+'s'
    c3 = SkyCoord(RA, DEC, frame='icrs')
    radectemp.append(c3.ra.degree)
    radectemp.append(c3.dec.degree)
    
VSXradec = np.float32(radectemp).reshape(-1,2)

dbscanRADEC = np.loadtxt('highdata.txt')
from scipy.spatial import cKDTree
kdt = cKDTree(dbscanRADEC[:,0:2])
dist, indices = kdt.query(VSXradec)

temp = []
for i in range (len(indices)):
    index = indices[i] 
    temp.append(dbscanRADEC[index])
nptemp = np.array(temp)  

VSXradec = np.column_stack((dist, VSXradec))
allradec = np.column_stack((VSXradec, nptemp))

indexdata = np.arange(0, VSXradec.shape[0])
allradec = np.column_stack((indexdata, allradec))
allradec[:,0] = allradec[:,0]+2

lista = ['INDEX', 'distance','VSXRA', 'VSXDEC', 'GRA', 'GDEC', 'PLX', 'PMRA', 'PMDEC', 'G', 'BP','RP', 'P=0']
dfallradec = pd.DataFrame(allradec, columns= lista)

df4 = [vsxtype, dfallradec]

result = pd.concat(df4, axis=1)

resultsort = result.sort_values(by='distance')

DataEW = resultsort[resultsort.iloc[:,0] == 'EW']

resultsort.to_csv('allvsx.csv', index=0)
DataEW.to_csv('EW.csv', index=0)

