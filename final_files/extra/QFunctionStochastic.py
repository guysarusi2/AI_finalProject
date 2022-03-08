import torch
import numpy as np
import Utillity as U


class QFunctionStochastic:

    def __init__(self, weight_vector):
        """
        Initialize the class with given weight vector
        :param weight_vector :(Tensor) initial weight vector
        """
        self.q = weight_vector

    def __call__(self, state, action):
        """
        Calculate state-action value given state and action
        value is linear combination of weights and state-action feature
        :param state : (Tensor) current state
        :param action : (Tensor) current action
        :return: state-action value
        """
        state_action_feature = U.generate_state_action_feature(state, action)
        q_val = torch.matmul(torch.transpose(self.q, 0, 1), state_action_feature)

        return q_val
