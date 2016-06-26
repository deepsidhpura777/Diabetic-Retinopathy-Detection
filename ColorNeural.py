# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 10:34:25 2016

@author: it-lab412
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 19:21:23 2016

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
                          ('conv2',layers.Conv2DLayer), 
                          ('conv3',layers.Conv2DLayer),  
                          ('pool1',layers.MaxPool2DLayer),
                          ('cnndrop1',layers.DropoutLayer),


                         
                          ('conv4',layers.Conv2DLayer),
                          ('conv5',layers.Conv2DLayer), 
                          ('conv6',layers.Conv2DLayer),  
                          ('pool2',layers.MaxPool2DLayer),
                          ('cnndrop2',layers.DropoutLayer),

                         

                         
                          ('conv7',layers.Conv2DLayer),
                          ('conv8',layers.Conv2DLayer),    
                          ('pool3',layers.MaxPool2DLayer),
                          ('cnndrop3',layers.DropoutLayer),
                        

                          ('conv9',layers.Conv2DLayer),
                          ('conv10',layers.Conv2DLayer),  
                          ('pool4',layers.MaxPool2DLayer),
                          ('cnndrop4',layers.DropoutLayer),
                          
                          ('conv11',layers.Conv2DLayer),
                          ('conv12',layers.Conv2DLayer),  
                          ('pool5',layers.MaxPool2DLayer),
                         
                          

                          ('dropout1',layers.DropoutLayer),
                          ('hidden1',layers.DenseLayer),
                          ('maxout1',layers.pool.FeaturePoolLayer),

                          ('dropout2',layers.DropoutLayer),
                          ('hidden2',layers.DenseLayer),
                          ('maxout2',layers.pool.FeaturePoolLayer),

                          ('output',layers.DenseLayer)],

            input_shape=(None,3,256,256),

            conv1_num_filters =16, conv1_filter_size=(3, 3), conv1_stride=(2,2), conv1_nonlinearity = lasagne.nonlinearities.LeakyRectify(0.01), conv1_pad = 1,
            conv2_num_filters =16, conv2_filter_size=(3, 3), conv2_stride=(1,1), conv2_nonlinearity = lasagne.nonlinearities.LeakyRectify(0.01), conv2_pad = 1,            
            conv3_num_filters =16, conv3_filter_size=(3, 3), conv3_stride=(1,1), conv3_nonlinearity = lasagne.nonlinearities.LeakyRectify(0.01), conv3_pad = 1,            
            pool1_pool_size=(2, 2),
            cnndrop1_p = 0.2,   
       
           
            conv4_num_filters =32, conv4_filter_size=(3, 3), conv4_stride=(1,1), conv4_nonlinearity = lasagne.nonlinearities.LeakyRectify(0.01), conv4_pad = 1,                         
            conv5_num_filters =32, conv5_filter_size=(3, 3), conv5_stride=(1,1), conv5_nonlinearity = lasagne.nonlinearities.LeakyRectify(0.01), conv5_pad = 1,                        
            conv6_num_filters =32, conv6_filter_size=(3, 3), conv6_stride=(1,1), conv6_nonlinearity = lasagne.nonlinearities.LeakyRectify(0.01), conv6_pad = 1,             
            pool2_pool_size=(2, 2),  
            cnndrop2_p = 0.3,            
           
           
           
            conv7_num_filters =48, conv7_filter_size=(3, 3), conv7_stride=(1,1), conv7_nonlinearity = lasagne.nonlinearities.LeakyRectify(0.01), conv7_pad = 1,             
            conv8_num_filters =48, conv8_filter_size=(3, 3), conv8_stride=(1,1), conv8_nonlinearity = lasagne.nonlinearities.LeakyRectify(0.01), conv8_pad = 1,             
            pool3_pool_size=(2, 2),  
            cnndrop3_p = 0.3,             
           
            conv9_num_filters =64, conv9_filter_size=(3, 3), conv9_stride=(1,1), conv9_nonlinearity = lasagne.nonlinearities.LeakyRectify(0.01), conv9_pad = 1, 
            conv10_num_filters =64, conv10_filter_size=(3, 3), conv10_stride=(1,1), conv10_nonlinearity = lasagne.nonlinearities.LeakyRectify(0.01), conv10_pad = 1, 
            pool4_pool_size=(2, 2),
            cnndrop4_p = 0.4,  
              
            conv11_num_filters =96, conv11_filter_size=(3, 3), conv11_stride=(1,1), conv11_nonlinearity = lasagne.nonlinearities.LeakyRectify(0.01), conv11_pad = 1, 
            conv12_num_filters =96, conv12_filter_size=(3, 3), conv12_stride=(1,1), conv12_nonlinearity = lasagne.nonlinearities.LeakyRectify(0.01), conv12_pad = 1, 
            pool5_pool_size=(2, 2),
            dropout1_p = 0.5,
            
            hidden1_num_units = 256, hidden1_nonlinearity = lasagne.nonlinearities.LeakyRectify(0.01),
            maxout1_pool_size = 2,

            dropout2_p = 0.5,
            hidden2_num_units = 256, hidden2_nonlinearity = lasagne.nonlinearities.LeakyRectify(0.01),
            maxout2_pool_size = 2,
            
           
            output_num_units = 5,
            output_nonlinearity = lasagne.nonlinearities.softmax,
            
          #  objective_l2 = 0.0005,
           
            
            update = nesterov_momentum,
            update_learning_rate = theano.shared(np.cast['float32'](0.003)),
            update_momentum = 0.9,
            
            on_epoch_finished=[
            updateRate.AdjustVariable('update_learning_rate')
            ],
            
            
            max_epochs = 300,
            verbose = 1,

            

            
          
            
                 )
                 
    return net