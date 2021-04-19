# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 17:37:34 2021

@author: dingxu
"""

import os
import pandas as pd
import numpy as np

PATH = 'E:\\shunbianyuan\\phometry\\pipelinecode\\cluster\\cluster\\CLUSTERCODE\\clusterdata\\'

filetemp = []
namefile = []
for root, dirs, files in os.walk(PATH):
   for file in files:
       strfile = os.path.join(root, file)
       if (strfile[-4:] == '.tsv'):
           print(strfile)
           filetemp.append(strfile)
           namefile.append(file)
           
for i in range(len(filetemp)):
    df = pd.read_csv(filetemp[i], sep = ';', encoding='gbk')
    dataframe = df.dropna()
    
    newdata = dataframe.values

    hang,lie = newdata.shape

    temp = []
    for index in range(0,hang):
        try:
            data = np.float32(newdata[index])
            temp.append(data)
        except:
            print('it is error')
        
    arraydata = np.array(temp)

    np.savetxt(PATH+namefile[i][:-4]+'.txt', arraydata)



