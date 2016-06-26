# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 19:28:26 2016

@author: it-lab412
"""

from os import listdir
from os.path import isfile, join
import numpy as np
import cv2
import pandas as pd
#import Split
import Neural
import re
import gc

np.random.seed(6)


_nsre = re.compile('([0-9]+)')

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(_nsre, s)]

mypath='/home/it-lab412/Desktop/All512'
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f))]
onlyfiles.sort(key=natural_sort_key)

onlyfiles = np.array(onlyfiles)
onlyfiles = onlyfiles[2:]


y = pd.read_csv('trainLabels.csv')
y = y['level']

y = y.values
y = y.astype(np.int32)

flop_files = onlyfiles[:35126]
new_files = onlyfiles[35126:70252]
shear_files = onlyfiles[70252:105378]
transpose_files = onlyfiles[105378:140504]
transverse_files = onlyfiles[140504:175630]


for i in range(0,12):             # Total 63 batches. Main outer loop for calling the net
     net = Neural.NN()
     X = np.ndarray(shape=(2702,1,512,512),dtype='float32')
     Y = np.ndarray(shape=(2702,),dtype=np.int32)
     Y[:] = 0
     Y = y[:2702]
     print "Batch Number = ",i 
     
     for n in range(2702):
        X[n,0] = cv2.imread(join(mypath,new_files[n]),0) / np.float32(255)
        X[n,0] = X[n,0] - np.mean(X[n,0])
        X[n,0] = X[n,0] / np.std(X[n,0])
        
     net.fit(X,Y)
     net.save_params_to('new_weights.csv')
     
     if i!= 0:
        net.load_params_from('new_weights.csv')
        print "Loaded parameters into the net"
    
     if i!= 11:
         X = None
         net = None
         gc.collect()
         print "Deleted Variables X and net"
        
    
     
"""  ff = flop_files[i*540:(i+1)*540]
     nf = new_files[i*540:(i+1)*540]
     sf = shear_files[i*540:(i+1)*540]
     tpf = transpose_files[i*540:(i+1)*540]
     tvf = transverse_files[i*540:(i+1)*540]
     
     for n in range(0,2700,5):
        X[n,0] = cv2.imread(join(mypath,ff[n/5]),0) / np.float32(255)
        X[n,0] = X[n,0] - np.mean(X[n,0])
        X[n,0] = X[n,0] / np.std(X[n,0])
        Y[n] = y[n/5]

     for n in range(1,2700,5):
        X[n,0] = cv2.imread(join(mypath,nf[(n-1)/5]),0) / np.float32(255)
        X[n,0] = X[n,0] - np.mean(X[n,0])
        X[n,0] = X[n,0] / np.std(X[n,0])
        Y[n] = y[(n-1)/5]
        
     for n in range(2,2700,5):
        X[n,0] = cv2.imread(join(mypath,sf[(n-2)/5]),0) / np.float32(255)
        X[n,0] = X[n,0] - np.mean(X[n,0])
        X[n,0] = X[n,0] / np.std(X[n,0])
        Y[n] = y[(n-2)/5]
        
     for n in range(3,2700,5):
        X[n,0] = cv2.imread(join(mypath,tpf[(n-3)/5]),0) / np.float32(255)
        X[n,0] = X[n,0] - np.mean(X[n,0])
        X[n,0] = X[n,0] / np.std(X[n,0])
        Y[n] = y[(n-3)/5]
        
     for n in range(4,2700,5):
        X[n,0] = cv2.imread(join(mypath,tvf[(n-4)/5]),0) / np.float32(255)
        X[n,0] = X[n,0] - np.mean(X[n,0])
        X[n,0] = X[n,0] / np.std(X[n,0])
        Y[n] = y[(n-4)/5]"""
     
    
        
     
    
     