# class for the Q function
import torch


class QFunctionDeterministic:

    # takes in set of parameters
    def __init__(self, weight_vector):
        self.q = weight_vector

    # outputs an action-state value given a state and action
    def __call__(self, state, action, policy, V):
        # action from policy μ
        # a_pred = μ(state)
        a_pred = policy(state)

        # error between given action and the action from the policy μ
        δ = action - a_pred

        # zero the current gradient and take gradient of μ(s)
        # μ.θ.grad = None
        # a_pred.backward()
        policy.y.grad = None
        a_pred.backward()

        # state-action feature
        # ϕ_sa = μ.θ.grad * δ
        ϕ_sa = policy.y.grad * δ

        # advantage of taking action a instead of the policy's action
        # Aʷ_sa = mult(ϕ_sa, self.w)
        Aʷ_sa = torch.matmul(torch.transpose(ϕ_sa, 0, 1), self.q)

        # q value, advantage + value
        q_val = Aʷ_sa + V(state)

        return q_val
