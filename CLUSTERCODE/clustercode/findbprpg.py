# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 17:39:32 2021

@author: dingxu
"""

import pandas as pd
import numpy as np

#CSV_FILE_PATH = 'CSCC9_40.csv'
CSV_FILE_PATH = 'M67.csv'
df = pd.read_csv(CSV_FILE_PATH)

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