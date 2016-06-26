# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 12:40:57 2016

@author: it-lab412
"""

from os import listdir
from os.path import isfile, join
import numpy as np
import cv2
import pandas as pd
#import Neural2
import re
import cPickle as pickle
import loss

from sklearn.preprocessing import MinMaxScaler

_nsre = re.compile('([0-9]+)')
scaler = MinMaxScaler()

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
           for text in re.split(_nsre, s)]
              
mypath='/home/it-lab412/Desktop/test'
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f))]
onlyfiles.sort(key=natural_sort_key)
onlyfiles = np.array(onlyfiles)

y = pd.read_csv('retinopathy_solution.csv')
y = y['level']

yval = y.values
yval = yval.astype(np.int32)
yval = yval[:5000]

X = np.ndarray(shape=(5000,1,512,512),dtype='float32') 

with open('CNN3.pickle', 'r') as f:
       net =  pickle.load(f)
       
for n in range(5000):
    X[n,0] = cv2.imread(join(mypath,onlyfiles[n]),0) / np.float32(255)
   # X[n,0] = X[n,0] - np.mean(X[n,0])
   # X[n,0] = X[n,0] / np.std(X[n,0])
    X[n,0] = scaler.fit_transform(X[n,0])
    
prediction = net.predict(X)
score = loss.quadratic_weighted_kappa(prediction,yval)
print score


