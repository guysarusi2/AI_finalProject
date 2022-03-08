import torch


class GaussianPolicy2:
    def __init__(self, weight_vector1, weight_vector2):
        self.y = weight_vector1
        self.x = weight_vector2

    def choose_action(self, state):
        # mean = mult(self.θ, state)
        mean = torch.matmul(torch.transpose(self.y, 0, 1), state)
        var = torch.exp(torch.matmul(torch.transpose(self.x, 0, 1), state))
        if var.item() == 0:
            print(var.item())
            print("var equal 0")
            var = torch.Tensor([0.1])
        # normal = torch.normal(mean, var)
        normal = torch.distributions.Normal(mean, var)
        action = normal.sample()
        action = torch.tanh(action)
        return action

    def get_grad(self, state, action):
        # mean = mult(self.θ, state)
        mean = torch.matmul(torch.transpose(self.y, 0, 1), state)
        var = torch.exp(torch.matmul(torch.transpose(self.x, 0, 1), state))
        if var.item() == 0:
            var = torch.Tensor([0.1])
        # normal = torch.normal(mean, var)
        # normal = torch.distributions.Normal(mean, var)
        # a_pred = normal.sample()
        # a_pred = torch.tanh(a_pred)
        score = ((action - mean) * (state)).div(var)
        return score


def main():
    a = torch.Tensor([[-40]])
    b = torch.exp(a)
    print(b)


if __name__ == '__main__':
    main()
