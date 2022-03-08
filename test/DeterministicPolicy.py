# class for the deterministic policy
import torch
import numpy as np


class DeterministicPolicy:

    # initialize set of parameters
    def __init__(self, weight_vector):
        self.y = weight_vector

    # gets action
    def __call__(self, state):
        s_numpy = state.numpy()
        s = np.copy(s_numpy)
        s1 = s[0]
        s2 = s[1]
        p = torch.tensor([[1.], [s1], [s2], [s1 * s2]])
        action = torch.matmul(torch.transpose(self.y, 0, 1), p)



        # linear combination of parameters and state
        # action = torch.matmul(torch.transpose(self.y, 0, 1), state)

        # tanh to constrain action to be in the range of [-1, 1]
        action = torch.tanh(action)

        return action
