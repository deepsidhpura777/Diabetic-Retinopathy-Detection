# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 11:50:19 2016

@author: it-lab412
"""


import numpy as np


class AdjustVariable(object):
    def __init__(self, name):
        self.name = name
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = {1:0.003,250:0.0003}

        epoch = train_history[-1]['epoch']
      
        
        if epoch in self.ls:
            new_value = np.float32(self.ls[epoch])
            ##print "In new Value= ",new_value
            getattr(nn, self.name).set_value(new_value)
