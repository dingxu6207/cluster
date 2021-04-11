# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 17:57:27 2021

@author: dingxu
"""

import pandas as pd

df = pd.read_csv('dr7_v1.2_LRS.csv', delimiter='|')

print(df.head(5))

df = df[['ra', 'dec']]


df.to_csv('testcsv.csv',encoding='gbk')



