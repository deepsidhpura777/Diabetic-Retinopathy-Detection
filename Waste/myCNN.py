# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 09:52:11 2016

@author: it-lab412
"""

from os import listdir
from os.path import isfile, join
import numpy as np
import cv2
import pandas as pd
import Split
import Neural
import re
import gc

_nsre = re.compile('([0-9]+)')

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(_nsre, s)]

mypath='/home/it-lab412/Desktop/All512'
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f))]
onlyfiles.sort(key=natural_sort_key)

onlyfiles = np.array(onlyfiles)
onlyfiles = onlyfiles[2:]
#X = np.ndarray(shape=(5018,1,512,512),dtype='float32')  ### Shifted inside the loop 

#random_seq = np.random.permutation(175630)
random_seq = seqArray
random_files = onlyfiles[random_seq]
random_splits = Split.split_seq(random_files,65)


y = pd.read_csv('trainLabels.csv')
y = y['level']


y1 = y
y2 = y

for i in range(4):
    y1 = y1.append(y2)

y1 = y1.values
y1 = y1.astype(np.int32)
   
y1 = y1[random_seq]



y_splits = Split.split_seq(y1,65)




for i in range(29,len(random_splits)-1):
    
    net = Neural.NN()
    X = np.ndarray(shape=(2702,1,512,512),dtype='float32') 
    print "Batch Number = ",i
    files = random_splits[i]
    Y = y_splits[i]    
    
    for n in range(0,2702):
        X[n,0] = cv2.imread(join(mypath,files[n]),0) / np.float32(255)
        
    if i!= 0:
        net.load_params_from('savedWeights.csv')
        print "Loaded parameters into the net"
        
    net.fit(X,Y)
    net.save_params_to('savedWeights.csv')
    
    if i!= 63 :
        X = None
        net = None
        gc.collect()
        print "Deleted Variables X and net"
    
###### Test on random_split[64] and y_split[64]  ==========   net.predict(random_split[64],y_split[64])
        
    
        
        








