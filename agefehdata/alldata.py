# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 16:38:03 2021

@author: dingxu
"""

import os
import pandas as pd
import numpy as np

PATH = 'E:\\shunbianyuan\\phometry\\pipelinecode\\cluster\\cluster\\agefehdata\\'

datatemp = []
filetemp = []
namefile = []
for root, dirs, files in os.walk(PATH):
   for file in files:
       strfile = os.path.join(root, file)
       if (strfile[-4:] == '.dat'):
           print(strfile)
           filetemp.append(strfile)
           namefile.append(file)
           
           dataparame = np.loadtxt(strfile)
           datatemp.append(dataparame)
           

print(len(datatemp))           
#npallparmdata = np.array(datatemp)
#np.savetxt('kuagefeh.txt', npallparmdata)
ar7 = np.zeros((1,36))           
for i in range(len(datatemp)):
    ar7=np.vstack((ar7,datatemp[i]))
    
np.savetxt('kuagefeh.txt', ar7[1:,:])