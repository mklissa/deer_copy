"""
Neural network using Keras (called by q_net_keras)

.. Authors: Vincent Francois-Lavet
"""

import numpy as np
from keras.models import Sequential
from keras.layers import Input, Layer, Dense, Flatten, Merge, Activation, Convolution2D, MaxPooling2D
import theano.tensor as T

class NN():
    """
    Deep Q-learning network using Keras
    
    Parameters
    -----------
    batch_size : int
        Number of tuples taken into account for each iteration of gradient descent
    inputDimensions :
    n_Actions :
    randomState : numpy random number generator
    """
    def __init__(self, batchSize, inputDimensions, n_Actions, randomState):
        self._inputDimensions=inputDimensions
        self._batchSize=batchSize
        self._randomState=randomState
        self._nActions=n_Actions

    def _buildDQN(self, inputs):
        """
        Build a network consistent with each type of inputs
        """
        layers=[]
        models_conv=[]
        
        for i, dim in enumerate(self._inputDimensions):
            nfilter=[]
            print i
            # - observation[i] is a FRAME - FIXME
            if len(dim) == 3: 
                model = Sequential()
                layers.append(model.layers[-1])
                model.add(Flatten())
                
            # - observation[i] is a VECTOR - FIXME
            elif len(dim) == 2 and dim[0] > 3:                                
                model = Sequential()
                layers.append(model.layers[-1])
                model.add(Flatten())
            # - observation[i] is a SCALAR -
            else:
                if dim[0] > 3:
                    print "here"
                    print dim[0]
                    model = Sequential()
                    model.add( Convolution2D(8, 1, 2, border_mode='valid', input_shape=(1, 1, dim[0],)) )
                    layers.append(model.layers[-1])
                    # model.add(Activation('relu'))                    
                    model.add(Convolution2D(8, 1, 2))
                    layers.append(model.layers[-1])
                    # model.add(Activation('relu'))
                    model.add(Flatten())
                    
                else:
                    print "here2"
                    print (dim[0])
                    
                    #lay = Activation('linear')
                    #lay.input = T.matrix()
                    #model = Sequential()
                    #model.add(lay)

                    model = Sequential()
                    if(len(dim) == 2):
                        model.add( Activation('linear', input_shape=(dim[0],dim[1],)) )
                    #    x = Input(shape=(dim[0],dim[1],), name='aux_input')
                    #    model.add( Layer()(x) )
                    if(len(dim) == 1):
                        model.add( Activation('linear', input_shape=(1, dim[0],))  )
                    #    x= Input(shape=(1,dim[0],), name='aux_input')
                    #    model.add( Layer()(x) )
                    
                    layers.append(model.layers[-1])
                    model.add(Flatten())
                #models_conv.append(model)

            models_conv.append(model)
        
        
        ## Merge of layers

        merged = Merge(models_conv, mode='concat')
        
        model = Sequential()
        model.add(merged)
        
        
        model.add(Dense(50, activation='relu'))
        layers.append(model.layers[-1])
        model.add(Dense(20, activation='relu'))
        layers.append(model.layers[-1])
        model.add( Dense(1) )
        layers.append(model.layers[-1])

        
        # Grab all the parameters together.
        params = [ param
                    for layer in layers 
                    for param in layer.trainable_weights ]
#        params.append([ param
#                    for layer in layers 
#                    for param in layer.non_trainable_weights])

        return model.layers[-1].output, params, None

if __name__ == '__main__':
    pass

