"""
Code for general deep Q-learning using Keras that can take as inputs scalars, vectors and matrices

.. Author: Vincent Francois-Lavet
"""

import numpy as np
from keras.optimizers import SGD,RMSprop
from keras import backend as K
from ..base_classes import QNetwork
from .NN_keras_distrib import NN # Default Neural network used
import copy

class MyQNetwork(QNetwork):
    """
    Deep Q-learning network using Keras (with any backend)
    
    Parameters
    -----------
    environment : object from class Environment
    rho : float
        Parameter for rmsprop. Default : 0.9
    rms_epsilon : float
        Parameter for rmsprop. Default : 0.0001
    momentum : float
        Default : 0
    clip_delta : float
        Not implemented.
    freeze_interval : int
        Period during which the target network is freezed and after which the target network is updated. Default : 1000
    batch_size : int
        Number of tuples taken into account for each iteration of gradient descent. Default : 32
    update_rule: str
        {sgd,rmsprop}. Default : rmsprop
    random_state : numpy random number generator
    double_Q : bool, optional
        Activate or not the double_Q learning.
        More informations in : Hado van Hasselt et al. (2015) - Deep Reinforcement Learning with Double Q-learning.
    neural_network : object, optional
        default is deer.qnetworks.NN_keras
    """

    def __init__(self, environment, rho=0.9, rms_epsilon=0.0001, momentum=0, clip_delta=0, freeze_interval=1000, batch_size=32, update_rule="rmsprop", random_state=np.random.RandomState(), double_Q=False, neural_network=NN):
        """ Initialize environment
        
        """
        QNetwork.__init__(self,environment, batch_size)

        
        self._rho = rho
        self._rms_epsilon = rms_epsilon
        self._momentum = momentum
        self._update_rule = update_rule
        #self.clip_delta = clip_delta
        self._freeze_interval = freeze_interval
        self._double_Q = double_Q
        self._random_state = random_state
        self._Vmin=0
        self._Vmax=1
        self._Natoms=11
        
        
        self.update_counter = 0
                
        Q_net = neural_network(self._batch_size, self._input_dimensions, self._n_actions, self._random_state, self._Natoms)
        self.q_vals, self.params = Q_net._buildDQN()
                
        self._compile()

        self.next_q_vals, self.next_params = Q_net._buildDQN()
        self.next_q_vals.compile(optimizer='rmsprop', loss='mse') #The parameters do not matter since training is done on self.q_vals

        self._resetQHat()

    def getAllParams(self):
        params_value=[]
        for i,p in enumerate(self.params):
            params_value.append(K.get_value(p))
        return params_value

    def setAllParams(self, list_of_values):
        for i,p in enumerate(self.params):
            K.set_value(p,list_of_values[i])

    def train(self, states_val, actions_val, rewards_val, next_states_val, terminals_val):
        """
        Train one batch.

        1. Set shared variable in states_shared, next_states_shared, actions_shared, rewards_shared, terminals_shared         
        2. perform batch training

        Parameters
        -----------
        states_val : list of batch_size * [list of max_num_elements* [list of k * [element 2D,1D or scalar]])
        actions_val : b x 1 numpy array of integers
        rewards_val : b x 1 numpy array
        next_states_val : list of batch_size * [list of max_num_elements* [list of k * [element 2D,1D or scalar]])
        terminals_val : b x 1 numpy boolean array

        Returns
        -------
        Average loss of the batch training (RMSE)
        Individual (square) losses for each tuple
        """
        
        if self.update_counter % self._freeze_interval == 0:
            self._resetQHat()
        
        not_terminals=np.ones_like(terminals_val) - terminals_val
        z_vals=np.arange(self._Natoms,dtype=float)/(self._Natoms-1)
#        print "states_val"
#        print states_val[0][0]
#        print actions_val[0]
#        print rewards_val[0]
        #print not_terminals
        #print "z_vals"
        #print z_vals
        Tz=np.expand_dims(rewards_val,axis=1) + np.expand_dims(not_terminals,axis=1) * self._df * np.expand_dims(z_vals,axis=0)
        #print "Tz"
        #print Tz
        #V_min=0 and \Delta_z=1!
        Tz=np.clip(Tz, 0.0001,0.9999) #to avoid lj=uj=0 or lj=uj=10
        bj=(Tz-0)/0.1 #
        #print "bj"
        #print bj
        lj=np.floor(bj).astype(int)
        uj=np.ceil(bj).astype(int)
        #print lj,uj

        # Get the z proba given by z(s',a*) with a=argmax Q(s',a) 
        next_states_val=next_states_val.tolist()
        next_z_vals_concat=[]
        next_q_vals_concat=[]
        for a in range(self._n_actions):
            next_s_a=copy.deepcopy(next_states_val)
            b = np.zeros((self._batch_size, self._n_actions))
            b[np.arange(self._batch_size),a]=1
            b=np.array(b)
        
            next_s_a.append(b)
            #print next_s_a
            
            next_z_vals = self.next_q_vals.predict(next_s_a)
            avg_next_z_vals=[np.sum(np.array(n)*np.arange(self._Natoms))/self._Natoms for n in next_z_vals]
            
            #print next_z_vals
            #print avg_next_z_vals
            next_z_vals_concat.append(next_z_vals)
            next_q_vals_concat.append(avg_next_z_vals)
                
        argmax_next_q_vals=np.argmax(np.transpose(next_q_vals_concat), axis=1)
        #print "argmax_next_q_vals"
        #print argmax_next_q_vals        
        
        #print np.swapaxes(next_z_vals_concat,0,1)
        #print np.swapaxes(next_z_vals_concat,0,1)[np.arange(self._batch_size),argmax_next_q_vals]
        next_z_vals_with_max_q=np.swapaxes(next_z_vals_concat,0,1)[np.arange(self._batch_size),argmax_next_q_vals]

        #print "next_z_vals_with_max_q[0]"
        #print next_z_vals_with_max_q[0]
        

        ml=np.zeros((self._batch_size,self._Natoms))
        for i in range(self._batch_size):
            for j,a in enumerate(lj[i]):
                ml[i,a]+=next_z_vals_with_max_q[i,j]*(a+1-bj[i,j])
                ml[i,a+1]+=next_z_vals_with_max_q[i,j]*(bj[i,j]-a)
        
        #print np.sum(ml,axis=1)
        print ml[0]


        # Build list s_a
        states_val=states_val.tolist()
        s_a=states_val
        b = np.zeros((self._batch_size, self._n_actions))
        b[np.arange(self._batch_size),actions_val[:,0]]=1
        b=np.array(b)
        s_a.append(b)
        #print "s_a"
        #print s_a
        # Get z proba for s,a
        #z_proba = self.next_q_vals.predict(s_a)        
        #print "z_proba"
        #print z_proba
        #                
        ## In order to obtain the individual losses, ...
        # TO DO
        loss_ind=np.zeros(self._batch_size)
                                
        loss=self.q_vals.train_on_batch(states_val, ml ) 
                
        self.update_counter += 1        

        return np.sqrt(loss),loss_ind


    def qValues(self, state_val, debug=False):
        """ Get the q values for one belief state

        Arguments
        ---------
        state_val : one belief state

        Returns
        -------
        The q values for the provided belief state
        """ 
        # Get the z proba given by z(s,a) and then calculate Q(s,a) 
        #print state_val
        q_vals_concat=[]
        for a in range(self._n_actions):
            s_a=copy.deepcopy(state_val)
            b = np.zeros((self._n_actions))
            b[a]=1
            b=np.expand_dims(np.array(b),axis=0)
        
            s_a.append(b)
            if(debug==True):
                print s_a
            
            z_vals = self.q_vals.predict(s_a)
            avg_z_vals=[np.sum(np.array(n)*np.arange(self._Natoms))/self._Natoms for n in z_vals]
            
            if(debug==True):
                print z_vals
            #print avg_z_vals
            q_vals_concat.append(avg_z_vals)

        #print q_vals_concat
        
        return q_vals_concat

    def chooseBestAction(self, state):
        """ Get the best action for a belief state

        Arguments
        ---------
        state : one belief state

        Returns
        -------
        The best action : int
        """        
        q_vals = self.qValues(state)
        #print q_vals
        #print "np.argmax(q_vals)"
        #print np.argmax(q_vals)

        return np.argmax(q_vals),np.max(q_vals)
        
    def _compile(self):
        """ compile self.q_vals
        """
        if (self._update_rule=="sgd"):
            optimizer = SGD(lr=self._lr, momentum=self._momentum, nesterov=False)
        elif (self._update_rule=="rmsprop"):
            optimizer = RMSprop(lr=self._lr, rho=self._rho, epsilon=self._rms_epsilon)
        else:
            raise Exception('The update_rule '+self._update_rule+' is not implemented.')
        
        self.q_vals.compile(optimizer=optimizer, loss='categorical_crossentropy')

    def _resetQHat(self):
        for i,(param,next_param) in enumerate(zip(self.params, self.next_params)):
            K.set_value(next_param,K.get_value(param))
