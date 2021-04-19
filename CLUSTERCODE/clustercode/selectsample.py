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

data = df[df.iloc[:,18]>4500]
#data = data[data.iloc[:,6]>100]

data.to_csv('sample.csv', header=0)