# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 14:54:53 2016

@author: it-lab412
"""

import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
import updateRate
import theano
import numpy as np


def NN():
    
    net = NeuralNet(
                layers = [
                          ('input',layers.InputLayer),

                          
                          ('conv1',layers.Conv2DLayer), 
                          ('pool1',layers.MaxPool2DLayer),

#                          ('conv2',layers.Conv2DLayer),
#                          ('pool2',layers.MaxPool2DLayer),
#                          
                          
                          ('conv3',layers.Conv2DLayer),  
                          ('pool3',layers.MaxPool2DLayer),

                          ('conv4',layers.Conv2DLayer),  
                          ('pool4',layers.MaxPool2DLayer),

                          ('conv5',layers.Conv2DLayer), 
                          ('pool5',layers.MaxPool2DLayer),

                          ('conv6',layers.Conv2DLayer), 
                          ('pool6',layers.MaxPool2DLayer),

                          ('conv7',layers.Conv2DLayer), 
                          ('pool7',layers.MaxPool2DLayer),

                          ('dropout1',layers.DropoutLayer),
                          ('hidden1',layers.DenseLayer),
                          ('maxout1',layers.pool.FeaturePoolLayer),

                          ('dropout2',layers.DropoutLayer),
                          ('hidden2',layers.DenseLayer),
                          ('maxout2',layers.pool.FeaturePoolLayer),

                          ('output',layers.DenseLayer)],

            input_shape=(None,1,512,512),

            conv1_num_filters =32, conv1_filter_size=(5, 5), conv1_stride=(2,2), conv1_nonlinearity = lasagne.nonlinearities.LeakyRectify(0.01), conv1_pad = 2,     
            pool1_pool_size=(2, 2),
            
#            conv2_num_filters =16, conv2_filter_size=(3, 3), conv2_stride=(1,1), conv2_nonlinearity = lasagne.nonlinearities.LeakyRectify(0.01), conv2_pad = 1,
#            pool2_pool_size=(2, 2),
            
            conv3_num_filters =32, conv3_filter_size=(3, 3), conv3_stride=(1,1), conv3_nonlinearity = lasagne.nonlinearities.LeakyRectify(0.01), conv3_pad = 1, 
            pool3_pool_size=(2, 2),

            conv4_num_filters =64, conv4_filter_size=(3, 3),  conv4_stride=(1,1), conv4_nonlinearity = lasagne.nonlinearities.LeakyRectify(0.01), conv4_pad = 1,
            pool4_pool_size=(2, 2),

            conv5_num_filters =96, conv5_filter_size=(3, 3), conv5_stride=(1,1), conv5_nonlinearity = lasagne.nonlinearities.LeakyRectify(0.01), conv5_pad = 1,
            pool5_pool_size=(2, 2),

            conv6_num_filters =128, conv6_filter_size=(3, 3), conv6_stride=(1,1), conv6_nonlinearity = lasagne.nonlinearities.LeakyRectify(0.01), conv6_pad = 1,
            pool6_pool_size=(2, 2),

            conv7_num_filters =256, conv7_filter_size=(3, 3), conv7_stride=(1,1), conv7_nonlinearity = lasagne.nonlinearities.LeakyRectify(0.01), conv7_pad = 1,
            pool7_pool_size=(2, 2),

            dropout1_p = 0.5,
            hidden1_num_units = 512, hidden1_nonlinearity = lasagne.nonlinearities.LeakyRectify(0.01),
            maxout1_pool_size = 2,

            dropout2_p = 0.5,
            hidden2_num_units = 512, hidden2_nonlinearity = lasagne.nonlinearities.LeakyRectify(0.01),
            maxout2_pool_size = 2,
            
            #regression = True,
           # output_nonlinearity=None,
            output_num_units = 5,
            output_nonlinearity = lasagne.nonlinearities.softmax,
           
            
            update = nesterov_momentum,
            update_learning_rate = theano.shared(np.cast['float32'](0.03)),
            #update_learning_rate = 0.003,
            update_momentum = 0.9,
            
            on_epoch_finished=[
            updateRate.AdjustVariable('update_learning_rate')
            ],
            
           # objective_l2 = 0.0005,
           
            
            max_epochs = 250,
            verbose = 1,

            

            
          
            
                 )
                 
    return net