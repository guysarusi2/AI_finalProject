import torch


class GaussianPolicy:
    def __init__(self, weight_vector):
        self.y = weight_vector

    def choose_action(self, state):
        # mean = mult(self.θ, state)
        mean = torch.matmul(torch.transpose(self.y, 0, 1), state)
        # print(f'weight {self.y} mean {mean}')
        var = torch.Tensor([0.1])
        # normal = torch.normal(mean, var)
        normal = torch.distributions.Normal(mean, var)
        action = normal.sample()
        action = torch.tanh(action)
        return action

    def get_grad(self, state, action):
        # mean = mult(self.θ, state)
        mean = torch.matmul(torch.transpose(self.y, 0, 1), state)
        var = torch.Tensor([0.1])
        # normal = torch.normal(mean, var)
        # normal = torch.distributions.Normal(mean, var)
        # a_pred = normal.sample()
        # a_pred = torch.tanh(a_pred)
        score = ((action - mean) * (state)).div(var)
        return score
