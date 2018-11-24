"""
Neural network using Keras (called by q_net_keras)
.. Author: Vincent Francois-Lavet
"""

import numpy as np
from keras.models import Model
from keras.layers import Input, Layer, Dense, Flatten, merge, Activation, Conv2D, MaxPooling2D, Reshape, Permute

np.random.seed(102912)

class NN():
    """
    Deep Q-learning network using Keras
    
    Parameters
    -----------
    batch_size : int
        Number of tuples taken into account for each iteration of gradient descent
    input_dimensions :
    n_actions :
    random_state : numpy random number generator
    action_as_input : Boolean
        Whether the action is given as input or as output
    """
    def __init__(self, batch_size, input_dimensions, n_actions, random_state, action_as_input=False):
        self._input_dimensions=input_dimensions
        self._batch_size=batch_size
        self._random_state=random_state
        self._n_actions=n_actions
        self._action_as_input=action_as_input
        self._internal_dim=3 # size random vector
        O=5 # size output distribution

    def generator_encoder_model():
        inputs = [ Input( shape=(4,48,48,) ) ]
        # input_distr, conditional info
        
        x=Convolution2D(16, 4, 4, border_mode='same', activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid')(x)
        x=Convolution2D(16, 4, 4, border_mode='same', activation='relu')(x)
        
        model = Model(input=inputs, output=x)
        
        return model

    def generator_transition_model():
        inputs = [ Input( shape=(1,1,O,) ), Input( shape=(n_actions,1,O,) ) ]
        
        x=merge(inputs,mode='concat',concat_axis=-3)
        x=Convolution2D(10, 1,3, border_mode='same', activation='relu')(x)
        x=Convolution2D(1, 1,3, border_mode='same', activation='relu')(x)
        
        model = Model(input=inputs, output=x)
        
        return model
    
    def discriminator_model():
        inputs = [ Input( shape=(1,1,O,) ), Input( shape=(1,1,O,) ), Input( shape=(n_actions,1,O,) ) ]
        # output_distr, input_distr, conditional info
     
        x=merge(inputs,mode='concat',concat_axis=-3)
        x=Convolution2D(10, 1,3, border_mode='same', activation='relu')(x)
        x=Convolution2D(1, 1,3, border_mode='same', activation='relu')(x)
        x=Flatten()(x)
        true_or_model=Dense(1, activation='sigmoid')(x)
        model = Model(input=inputs, output=true_or_model)
        return model
        
    def full_model(generator, discriminator):
        inputs = [ Input( shape=(1,1,O,) ), Input( shape=(n_actions,1,O,) ) ]
        # input_distr, conditional info
        H = generator(inputs)
        discriminator.trainable = False
        gan_V = discriminator([H, inputs[0], inputs[1]])
        model = Model(input=inputs, output=gan_V)
        return model

    def _buildDQN(self):
        """
        Build a network consistent with each type of inputs
        """
        layers=[]
        outs_conv=[]
        inputs=[]

        for i, dim in enumerate(self._input_dimensions):
            # - observation[i] is a FRAME
            if len(dim) == 3:
                input = Input(shape=(dim[0],dim[1],dim[2]))
                inputs.append(input)
                reshaped=Permute((2,3,1), input_shape=(dim[0],dim[1],dim[2]))(input)    #data_format='channels_last'
                x = Conv2D(8, 4, 4, activation='relu', border_mode='valid')(reshaped)   #Conv on the frames
                x = Conv2D(16, 3, 3, activation='relu', border_mode='valid')(x)         #Conv on the frames
                x = MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid')(x)
                x = Conv2D(16, 3, 3, activation='relu', border_mode='valid')(x)         #Conv on the frames
                
                out = Flatten()(x)
                
            # - observation[i] is a VECTOR
            elif len(dim) == 2:
                if dim[0] > 3:
                    input = Input(shape=(dim[0],dim[1]))
                    inputs.append(input)
                    reshaped=Reshape((dim[0],dim[1],1), input_shape=(dim[0],dim[1]))(input) 
                    x = Conv2D(16, (2, 1), activation='relu', border_mode='valid')(reshaped)#Conv on the history
                    x = Conv2D(16, (2, 2), activation='relu', border_mode='valid')(x)       #Conv on the history & features

                    out = Flatten()(x)
                else:
                    input = Input(shape=(dim[0],dim[1]))
                    inputs.append(input)
                    out = Flatten()(input)

            # - observation[i] is a SCALAR -
            else:
                if dim[0] > 3:
                    # this returns a tensor
                    input = Input(shape=(dim[0],))
                    inputs.append(input)
                    reshaped=Reshape((1,dim[0],1), input_shape=(dim[0],))(input)  
                    x = Conv2D(8, (1,2), activation='relu', border_mode='valid')(reshaped)  #Conv on the history
                    x = Conv2D(8, (1,2), activation='relu', border_mode='valid')(x)         #Conv on the history
                    
                    out = Flatten()(x)
                                        
                else:
                    input = Input(shape=(dim[0],))
                    inputs.append(input)
                    out=input
                    
            outs_conv.append(out)

        if (self._action_as_input==True):
            if ( isinstance(self._n_actions,int)):
                print("Error, env.nActions() must be a continuous set when using actions as inputs in the NN")
            else:
                input = Input(shape=(len(self._n_actions),))
                inputs.append(input)
                outs_conv.append(input)
        
        if len(outs_conv)>1:
            x = merge(outs_conv, mode='concat')
        else:
            x= outs_conv [0]
        
        # we stack a deep fully-connected network on top
        x = Dense(50, activation='relu')(x)
        x = Dense(20, activation='relu')(x)
        
        if (self._action_as_input==False):
            if ( isinstance(self._n_actions,int)):
                out = Dense(self._n_actions)(x)
            else:
                out = Dense(len(self._n_actions))(x)
        else:
            out = Dense(1)(x)
                
        model = Model(input=inputs, output=out)
        layers=model.layers
        
        # Grab all the parameters together.
        params = [ param
                    for layer in layers 
                    for param in layer.trainable_weights ]
        
        if (self._action_as_input==True):
            return model, params, inputs
        else:
            return model, params

if __name__ == '__main__':
    pass
    