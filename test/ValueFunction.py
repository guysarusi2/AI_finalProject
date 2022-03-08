# class for value function
import torch
import numpy as np

class ValueFunction:

    # initialize set of parameters
    def __init__(self, weight_vector):
        self.v = weight_vector

    # get value
    def __call__(self, state):
        s_numpy = state.numpy()
        s = np.copy(s_numpy)
        s1 = s[0]
        s2 = s[1]
        p = torch.tensor([[1.], [s1], [s2], [s1 * s2]])
        val = torch.matmul(torch.transpose(self.v, 0, 1), p)

        # linear combination of parameters and state
        # val = torch.matmul(torch.transpose(self.v, 0, 1), state)

        return val
