import torch


class DeterministicPolicy:

    def __init__(self, weight_vector):
        """Initialize the class with given weight vector.
        :param weight_vector : (Tensor) initial weight vector
        """
        self.y = weight_vector

    def __call__(self, state):
        """
        Calculate action given the state
        action is linear combination of weights and state feature
        :param state : (Tensor) current state
        :return: action squeezed between [-1,1]
        """
        action = torch.matmul(torch.transpose(self.y, 0, 1), state)
        action = torch.tanh(action)
        return action
