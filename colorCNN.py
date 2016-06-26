# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 10:21:59 2016

@author: it-lab412
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 11:07:16 2016

@author: it-lab412
"""

from os.path import join
import numpy as np
import cv2
import Split
import ColorNeural
import cPickle as pickle
from sklearn.preprocessing import MinMaxScaler
np.random.seed(488)

#def normalize(imgs, std_reg=1e-5):
#    return (imgs - imgs.mean(axis=0, keepdims=True)) / (imgs.std(axis=0, keepdims=True) + std_reg)

scaler=MinMaxScaler()
mypath='/home/it-lab412/Desktop/combined'

onlyfiles = colorFilesArray
random_seq = np.random.permutation(72390)
onlyfiles = onlyfiles[random_seq]
files_splits = Split.split_seq(onlyfiles,30)

y1 = colorYArray.astype(np.int32)
y1 = y1[random_seq]
y_splits = Split.split_seq(y1,30)

net = ColorNeural.NN()
        
for i in range(30):
        
    X = np.ndarray(shape=(2413,3,256,256),dtype='float32') 
    print "Batch Number = ",i
    files = files_splits[i]
    Y = y_splits[i]    
    
    for n in range(0,2413):
        image = cv2.imread(join(mypath,files[n])) / np.float32(255)
        
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
        
    if i == 0:
        net.load_params_from('CNN_color5.csv')
    net.fit(X,Y)
    net.save_params_to('CNN_color6.csv')
    with open('CNN_color6.pickle', 'wb') as f:
        pickle.dump(net, f, -1)
    
    if i!= 29:
        X = None
       
        print "Deleted Variable X"
        
        