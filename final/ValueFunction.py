# class for value function
import torch


class ValueFunction:

    # initialize set of parameters
    def __init__(self, weight_vector):
        self.v = weight_vector

    # get value
    def __call__(self, state):
        # linear combination of parameters and state
        # val = mult(self.v, state)
        # val = torch.matmul(self.v, state)
        val = torch.matmul(torch.transpose(self.v, 0, 1), state)


        return val
