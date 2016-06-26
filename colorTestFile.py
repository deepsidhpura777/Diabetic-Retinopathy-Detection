# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 13:11:05 2016

@author: it-lab412
"""

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
              
mypath='/home/it-lab412/Desktop/color'
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f))]
onlyfiles.sort(key=natural_sort_key)
onlyfiles = np.array(onlyfiles)
onlyfiles = onlyfiles[:5000]

y = pd.read_csv('retinopathy_solution.csv')
y = y['level']

yval = y.values
yval = yval.astype(np.int32)
yval = yval[:5000]

X = np.ndarray(shape=(5000,3,256,256),dtype='float32') 

with open('CNN_color6.pickle', 'r') as f:
       net =  pickle.load(f)
       
for n in range(5000):
        image = cv2.imread(join(mypath,onlyfiles[n])) / np.float32(255)
        
        r,g,b = cv2.split(image)
        rMean,rStd = r.mean(),r.std()
        gMean,gStd = g.mean(),g.std()
        bMean,bStd = b.mean(),b.std()
        
        image[:,:,0] = image[:,:,0] - rMean
        image[:,:,0] = image[:,:,0] / rStd
        
        image[:,:,1] = image[:,:,1] - gMean
        image[:,:,1] = image[:,:,1] / gStd
        
        image[:,:,2] = image[:,:,2] - bMean
        image[:,:,2] = image[:,:,2] / bStd
        
        image = np.swapaxes(image,0,2)
        image = np.swapaxes(image,1,2)
       
        X[n] = image

    
prediction = net.predict(X)
score = loss.quadratic_weighted_kappa(prediction,yval)
print score


