import torch
import numpy as np

class GaussianPolicy:
    def __init__(self, weight_vector):
        self.y = weight_vector

    def choose_action(self, state):
        s_numpy = state.numpy()
        s = np.copy(s_numpy)
        s1 = s[0]
        s2 = s[1]
        p = torch.tensor([[1.], [s1], [s2], [s1 * s2]])
        mean = torch.matmul(torch.transpose(self.y, 0, 1), p)



        # mean = torch.matmul(torch.transpose(self.y, 0, 1), state)
        var = torch.Tensor([0.1])
        # normal = torch.normal(mean, var)
        normal = torch.distributions.Normal(mean, var)
        action = normal.sample()
        action = torch.tanh(action)
        return action

    def get_grad(self, state, action):
        s_numpy = state.numpy()
        s = np.copy(s_numpy)
        s1 = s[0]
        s2 = s[1]
        p = torch.tensor([[1.], [s1], [s2], [s1 * s2]])
        mean = torch.matmul(torch.transpose(self.y, 0, 1), p)


        # normal = torch.normal(mean, var)
        # normal = torch.distributions.Normal(mean, var)
        # a_pred = normal.sample()
        # a_pred = torch.tanh(a_pred)
        # mean = torch.matmul(torch.transpose(self.y, 0, 1), state)
        var = torch.Tensor([0.1])
        # score = ((action - mean) * (state)).div(var)
        score = ((action - mean) * (p)).div(var)
        return score
