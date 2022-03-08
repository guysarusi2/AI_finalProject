# class for the Q function
import numpy as np
import torch


class QFunctionStochastic():

    # takes in set of parameters
    def __init__(self, weight_vector):
        self.q = weight_vector

    # outputs an action-state value given a state and action
    def __call__(self, state, action):
        # # action from policy μ
        # a_pred = μ.choose_action(state)
        # # a_pred.requires_grad = True
        #
        # # error between given action and the action from the policy μ
        # δ = action - a_pred
        # # zero the current gradient and take gradient of μ(s)
        # μ.θ.grad = None
        # a_pred.backward()
        #
        # # state-action feature
        # ϕ_sa = μ.θ.grad * δ
        #
        # # advantage of taking action a instead of the policy's action
        # Aʷ_sa = mult(ϕ_sa, self.w)
        #
        # # q value, advantage + value
        # q_val = Aʷ_sa + V(state)

        # action = torch.Tensor([action.item()])
        s_ = state.numpy()
        s = np.copy(s_)
        a_ = action.numpy()
        a = np.copy(a_)
        s1 = s[0][0]
        s2 = s[1][0]
        a = a[0]

        p = torch.tensor([[1.], [s1], [s2], [a], [s1 * s2], [s1 * a], [s2 * a]])

        # a = torch.Tensor([[action.item()]])
        # s = torch.cat((state, a), 0)
        q_val = torch.matmul(torch.transpose(self.q, 0, 1), p)

        return q_val
