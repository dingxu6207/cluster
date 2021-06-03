# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 14:25:11 2021

@author: dingxu
"""

import numpy as np
import pandas as pd

FILE = 'table1.dat'
file = FILE

df = pd.read_csv(file, sep = '\s+')

#data = df[df.iloc[:,18]>4500]
#data = data[data.iloc[:,6]>100]
n_number = df.iloc[:,6]

dfdata = df.sort_values(df.columns[6], ascending = False)

dfdata.to_csv('sample.csv', header=0, index = 0)