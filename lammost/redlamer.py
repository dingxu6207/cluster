# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 20:56:31 2021

@author: dingxu
"""

import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

df = pd.read_csv('memberla.csv', delimiter='|')

print(df.head(5))

df = df[['combined_ra', 'combined_dec', 'combined_rv', 'combined_z','combined_feh']]


df.to_csv('memberselectlamost.csv',encoding='gbk')

npdf = np.array(df)

bemember = np.loadtxt('highdata.txt')

#cg18对应lamost数据
kdt = cKDTree(npdf[:,0:2])
dist, indices = kdt.query(bemember[:,0:2])


temp = []
for i in range (len(indices)):
    index = indices[i]
    temp.append(npdf[index])
    
nptemp = np.array(temp)    

distbe = np.column_stack((dist, bemember[:,0:2]))
radeclamost = np.column_stack((distbe, nptemp))

data = radeclamost[radeclamost[:,0].argsort()]