# class for the deterministic policy
import torch


class DeterministicPolicy:

    # initialize set of parameters
    def __init__(self, weight_vector):
        self.y = weight_vector

    # gets action
    def __call__(self, state):
        # linear combination of parameters and state
        # action = mult(self.θ, state)
        action = torch.matmul(torch.transpose(self.y, 0, 1), state)
        # action = torch.matmul(self.y, state)

        # tanh to constrain action to be in the range of [-1, 1]
        action = torch.tanh(action)

        return action
