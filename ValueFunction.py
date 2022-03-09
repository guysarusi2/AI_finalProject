import torch


class ValueFunction:

    def __init__(self, weight_vector):
        """
        Initialize the class with given weight vector
        :param weight_vector (Tensor): initial weight vector
        """
        self.v = weight_vector

    def __call__(self, state):
        """
        Calculate state value given state
        value is linear combination of weights and state feature
        :param state (Tensor): current state
        :return: state value
        """
        val = torch.matmul(torch.transpose(self.v, 0, 1), state)
        return val
