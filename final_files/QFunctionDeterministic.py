import torch


class QFunctionDeterministic:

    def __init__(self, weight_vector):
        """
        Initialize the class with given weight vector
        :param weight_vector :(Tensor) initial weight vector
        """
        self.q = weight_vector

    def __call__(self, state, action, agent):
        """
        Calculate state-action value given state and action.
        State-action value is the sum of the advantage function and the value function.
        Advantage function and value function are linear combination of features and weights.
        :param state : (Tensor) current state
        :param action : (Tensor) current action
        :param agent :  deterministic actor-critic agent
        :return: state-action value
        """
        # chosen action from policy
        policy_action = agent.policy(state)

        # error between action and the chosen action from the policy
        action_error = action - policy_action

        # calculate policy's gradient
        agent.policy.y.grad = None
        policy_action.backward()
        policy_gradient = agent.policy.y.grad

        # state-action feature
        state_action_feature = policy_gradient * action_error

        # advantage function is linear combination of state-action feature and weights
        advantage_function = torch.matmul(torch.transpose(state_action_feature, 0, 1), self.q)

        # Q(s,a) = A(s,a) + V(s)
        q_val = advantage_function + agent.V(state)

        return q_val
