""" Simple e-greedy planning policy

Authors: Vincent Francois-Lavet
"""

from ..base_classes import Policy
import itertools
import random
import copy


class PlanningPolicy(Policy):
    """
    """
    def __init__(self, q_network, env, random_state, epsilon, max_horizon=1):
        Policy.__init__(self, q_network, env.nActions(), random_state)
        self._env = env
        self._epsilon = epsilon
        self._max_horizon = max_horizon
        self._node0=copy.deepcopy(self._env)

    def act(self, state):
        for a in n_actions
            selected_node=copy.deepcopy(self._node0)
            selected_state=copy.deepcopy(self._env._state)
            reward = selected_node.act(action)
            
            obs = selected_node.observe()
            is_terminal = self._environment.inTerminalState()
            
            if (is_terminal==True):
                Q=reward
            else:
                for i in range(len(obs)):
                    self._state[i][0:-1] = self._state[i][1:]
                    self._state[i][-1] = obs[i]

        
        return action, V

    def setEpsilon(self, e):
        """ Set the epsilon used for :math:`\epsilon`-greedy exploration
        """
        self._epsilon = e

    def epsilon(self):
        """ Get the epsilon for :math:`\epsilon`-greedy exploration
        """
        return self._epsilon