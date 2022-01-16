# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 21:32:30 2021

@author: dingxu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ztfquery import lightcurve
from PyAstronomy.pyasl import foldAt
from PyAstronomy.pyTiming import pyPDM
from astropy.timeseries import LombScargle
import os

def computeperiod(npjdmag):
    JDtime = npjdmag[:,0]
    targetflux = npjdmag[:,1]
    ls = LombScargle(JDtime, targetflux, normalization='model')
    frequency, power = ls.autopower(minimum_frequency=0.025,maximum_frequency=20)
    index = np.argmax(power)
    maxpower = np.max(power)
    period = 1/frequency[index]
    wrongP = ls.false_alarm_probability(power.max())
    return period, wrongP, maxpower


def pholdata(npjdmag, P):
    phases = foldAt(npjdmag[:,0], P)
    sortIndi = np.argsort(phases)
    phases = phases[sortIndi]
    resultmag = npjdmag[:,1][sortIndi]
    
    return phases, resultmag

def computePDM(npjdmag,P):
    timedata = npjdmag[:,0]
    magdata = npjdmag[:,1]
    f0 =1/(2*P) 
    S = pyPDM.Scanner(minVal=f0-0.01, maxVal=f0+0.01, dVal=0.001, mode="frequency")
    P = pyPDM.PyPDM(timedata, magdata)
    bindata = int(len(magdata)/4)
    #bindata = 10
    f2, t2 = P.pdmEquiBin(bindata, S)
    delta = np.min(t2)
    pdmp = 1/f2[np.argmin(t2)]
    return pdmp, delta

def showfig(phases, resultmag):
    plt.figure(1)
    plt.plot(phases, resultmag, '.')
    plt.xlabel('phase',fontsize=14)
    plt.ylabel('mag',fontsize=14)
    ax = plt.gca()
    ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
    ax.invert_yaxis() #y轴反向
    plt.pause(1)
    plt.clf()
    
    
def showmjdmag(mjdmag):
    plt.figure(2)
    plt.clf()
    mjd = mjdmag[:,0]
    mag = mjdmag[:,1]
    plt.plot(mjd, mag, '.')
    plt.xlabel('MJD',fontsize=14)
    plt.ylabel('mag',fontsize=14)
    ax = plt.gca()
    ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
    ax.invert_yaxis() #y轴反向
    plt.pause(1)
    
    
def showcmd(highdata,RA,DEC):
    plt.figure(3)
    plt.clf()
    plt.scatter(highdata.iloc[:,0], highdata.iloc[:,1], c ='r', marker='o', s=1)
    plt.plot(RA,DEC,'o',c='g')
    plt.xlabel('RA',fontsize=14)
    plt.ylabel('DEC',fontsize=14)
    plt.pause(1)
    
def computePDMA(npjdmag):
    timedata = npjdmag[:,0]
    magdata = npjdmag[:,1]
    S = pyPDM.Scanner(minVal=0.005, maxVal=20, dVal=0.0001, mode="frequency")
    P = pyPDM.PyPDM(timedata, magdata)
    bindata = int(len(magdata)/4)
    #bindata = 10
    f2, t2 = P.pdmEquiBin(bindata, S)
    delta = np.min(t2)
    pdmp = 1/f2[np.argmin(t2)]
    return pdmp, delta   

def computeperiodbs(JDtime, targetflux):
    from astropy.timeseries import BoxLeastSquares
    model = BoxLeastSquares(JDtime, targetflux)
    results = model.autopower(0.16)
    period = results.period[np.argmax(results.power)]
    return period, 0, 0

def stddata(npjdmag, P):
    yuandata = np.copy(npjdmag[:,1])
    phases, resultmag = pholdata(npmjdmag, P)
    datamag = np.copy(resultmag)
    datanoise = np.diff(datamag,2).std()/np.sqrt(6)
    stdndata = np.std(yuandata)
    return stdndata/datanoise

file = 'CG20star.dat'
cg20data = pd.read_csv(file, sep='\s+', error_bad_lines=False) #设置列名和 默认以多个空格来识别数据
df = cg20data[['RAdeg', 'DEdeg', 'Cluster']]

#temp = df[df["Cluster"]== 'Melotte_25']

#temp.to_csv('Melotte_25.csv', index=0)

#Radecname = pd.read_csv('Melotte_25.csv')
hang,lie = df.shape
ra = df.iloc[0,0]
dec = df.iloc[0,1]
SAVEPATH = 'F:\\CG20variable\\'
for i in range (0,hang):
    RAp = df.iloc[i,0]
    DECp = df.iloc[i,1]
    np.savetxt('i.txt', [i,0])
    #showcmd(df,RAp,DECp)
    print('clustername= '+str(df.iloc[i,2]))
    
    pathfilename = SAVEPATH+str(df.iloc[i,2])+'\\'
    ishere = os.path.exists(pathfilename)
    if not ishere:
        os.mkdir(pathfilename)
        print(pathfilename)
    
    print('**************'+str(i)+'*******************')
    print('RA = '+str(RAp)+' '+'DEC = '+str(DECp))
    try:
        lcq = lightcurve.LCQuery.from_position(RAp, DECp, 1)
    except:
        try:
            lcq = lightcurve.LCQuery.from_position(RAp, DECp, 1)
        except:
            try:
                lcq = lightcurve.LCQuery.from_position(RAp, DECp, 1)
            except:
                print('it is continue 1!')
                continue
            
    print('downdata is ok!')
           
    dfdata = lcq.data
    
    
    try:
        gri = dfdata['filtercode'].value_counts()
        leng = gri['zg']    
    except:
        try:
            lenr = gri['zr']
            leng = 5
        except:
            print('it is continue 2!')
            continue
        
    try:
        lenr = gri['zr']    
    except:
        try:
            leng = gri['zg'] 
            lenr = 5
        except:
            print('it is continue 3!')
            continue
    
    
    try:
        print('r: '+str(lenr)+' g: '+str(leng)+' is ok')
        if leng<lenr:
            dfdata = dfdata[dfdata["filtercode"]=='zr']
        else:
            dfdata = dfdata[dfdata["filtercode"]=='zg']
            
        dfdata = dfdata[dfdata["catflags"]!= 32768]
        mjdmag = dfdata[['mjd', 'mag']]
        npmjdmag = np.array(mjdmag)
        period, wrongP, maxpower = computeperiod(npmjdmag)
        #P,delta = computePDM(npmjdmag, period)
        P1 = period
        P2 = period*2
        stddata1 = stddata(npmjdmag, P1)
        stddata2 = stddata(npmjdmag, P2)
        showmjdmag(npmjdmag)
        #print('delta = '+ str(delta))
        print('stddata1 = '+ str(stddata1))
        print('stddata2 = '+ str(stddata2))
        
        phases, resultmag = pholdata(npmjdmag, P2)
        showfig(phases, resultmag)
        print('wrongP= '+str(wrongP))
        if (stddata1 > 1.15 or stddata2 > 1.15) and (wrongP<1e-10) :
            dfdata.to_csv(SAVEPATH+str(df.iloc[i,2])+'\\'+str(RAp)+'-'+str(DECp)+'.csv')
    except:
        print(len(dfdata))
        print(str(len(dfdata))+'it is error')
        
        