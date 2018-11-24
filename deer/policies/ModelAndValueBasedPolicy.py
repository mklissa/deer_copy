from ..base_classes import Policy
import numpy as np

class ModelAndValueBasedPolicy(Policy):
    """The policy that acts greedily w.r.t. a given Q-network with probability 1-\epsilon, and acts randomly otherwise.
    It is now used as a default policy for the neural agent.
    """
    def __init__(self, q_network, n_actions, random_state, epsilon):
        Policy.__init__(self, q_network, n_actions, random_state)
        self._epsilon = epsilon

    def act(self, state):
        #print [[s.tolist()] for s in state]
        #print [np.expand_dims(s, axis=0) for s in state]
        
        if self.random_state.rand() < self._epsilon:
            action = self.random_state.randint(0, self.n_actions)
            V = 0
        else:
            action, V = self.bestAction(state)

        list_state_action=[np.expand_dims(s, axis=0) for s in state]
        list_state_action.append(np.array([action]))
        #print list_state_action
        outs=self.q_network.full_model.predict(list_state_action)
        print outs[3:]
        
        return action, V

    def setEpsilon(self, e):
        """ Set the epsilon used for :math:`\epsilon`-greedy exploration
        """
        self._epsilon = e

    def epsilon(self):
        """ Get the epsilon for :math:`\epsilon`-greedy exploration
        """
        return self._epsilon
