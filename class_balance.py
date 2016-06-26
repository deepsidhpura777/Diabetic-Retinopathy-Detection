# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 09:55:05 2016

@author: it-lab412
"""
import numpy as np

def class_balance(X,y): #X and y is a list of files and y labels respectively
   ind = np.where(y==0)
   index = ind[0]
   y_list = y.tolist()
   files_list = X.tolist() 
   rev = index[::-1]
   for i in rev:
       if i < 35126 or i > 70251:
           x = files_list.pop(i)
           a = y_list.pop(i)
           
   filesArray = np.array(files_list)
   yArray = np.array(y_list)
   return(filesArray, yArray)
    