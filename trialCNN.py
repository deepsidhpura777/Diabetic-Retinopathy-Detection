# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 11:07:16 2016

@author: it-lab412
"""

#from os import listdir
from os.path import join
import numpy as np
import cv2
#import pandas as pd
import Split
import Neural
#import re
#import gc
import cPickle as pickle
from sklearn.preprocessing import MinMaxScaler


#_nsre = re.compile('([0-9]+)')
np.random.seed(788)
#
#
#def natural_sort_key(s):
#    return [int(text) if text.isdigit() else text.lower()
#            for text in re.split(_nsre, s)]
#
mypath='/home/it-lab412/Desktop/All512'
#onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f))]
#onlyfiles.sort(key=natural_sort_key)
#
#onlyfiles = np.array(onlyfiles)
#onlyfiles = onlyfiles[2:]
onlyfiles = newFilesArray
random_seq = np.random.permutation(72390)
onlyfiles = onlyfiles[random_seq]
#
#
files_splits = Split.split_seq(onlyfiles,30)
#
#y = pd.read_csv('trainLabels.csv')
#y = y['level']
#
#
#y1 = y
#y2 = y
#
#for i in range(4):
#    y1 = y1.append(y2)
#
#
#y1 = y1.values
#y1 = y1.astype(np.int32)
#
#
y1 = newYArray.astype(np.int32)
y1 = y1[random_seq]
y_splits = Split.split_seq(y1,30)

net = Neural.NN()
scaler = MinMaxScaler()
        
for i in range(30):
        
    X = np.ndarray(shape=(2413,1,512,512),dtype='float32') 
    print "Batch Number = ",i
    files = files_splits[i]
    Y = y_splits[i]    
    
    for n in range(0,2413):
        X[n,0] = cv2.imread(join(mypath,files[n]),0) / np.float32(255)
        #X[n,0] = X[n,0] - np.mean(X[n,0])
        #X[n,0] = X[n,0] / np.std(X[n,0])
        X[n,0] = scaler.fit_transform(X[n,0])
        
   # if i!= 0:
   # net.load_params_from('kstew_weights.csv')
   # print "Loaded parameters into the net"
        
    net.fit(X,Y)
    net.save_params_to('CNN6.csv')
    with open('CNN6.pickle', 'wb') as f:
        pickle.dump(net, f, -1)
    
    if i!= 29:
        X = None
       # net = None
        print "Deleted Variable X"