# class for the Q function
import torch
import numpy as np


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
        s1 = s[0]
        # print(s1.dtype)
        s2 = s[1]
        s__ = np.array([s1, s2, a[0]], dtype=object)
        # print(s__)
        s__ = np.vstack(s__).astype(np.float64)
        s_1 = np.vstack((s1, s2, a))
        s = torch.from_numpy(s_1)

        # print(s__)

        # a = torch.Tensor([[action.item()]])
        # s = torch.cat((state, a), 0)
        # s = torch.from_numpy(s__)
        s = torch.from_numpy(s_1)
        # s = torch.tensor([s1, s2, a])
        q_val = torch.matmul(torch.transpose(self.q, 0, 1), s)

        return q_val
