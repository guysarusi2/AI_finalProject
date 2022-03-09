import torch


class GaussianPolicy:

    def __init__(self, weight_vector):
        """Initialize the class with given weight vector.
        :param weight_vector : (Tensor) initial weight vector
        """
        self.y = weight_vector

    def __call__(self, state):
        """
        Sample action from Gaussian distribution.
        Variance is fixed, and mean is linear combination of weights and state feature.
        :param state : (Tensor) current state
        :return: sampled action squeezed between [-1,1]
        """
        mean = torch.matmul(torch.transpose(self.y, 0, 1), state)
        var = torch.Tensor([0.1])
        normal_dist = torch.distributions.Normal(mean, var)
        action = normal_dist.sample()
        action = torch.tanh(action)
        return action

    def get_score_function(self, state, action):
        """
        Calculate score function.
        Variance is fixed, and mean is linear combination of weights and state feature.
        :param state : (Tensor) current state
        :param action : (Tensor) current action
        :return: score function value
        """
        mean = torch.matmul(torch.transpose(self.y, 0, 1), state)
        var = torch.Tensor([0.1])
        score_function = ((action - mean) * (state)).div(var)
        return score_function
