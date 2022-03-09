import torch
import gym
import numpy as np
from collections import deque
from DeterministicPolicy import DeterministicPolicy
from ValueFunction import ValueFunction
from QFunctionDeterministic import QFunctionDeterministic


class DeterministicActorCritic:
    def __init__(self):
        self.gama = 0.99
        self.num_episodes = 900
        self.max_steps = 250
        self.a_y = 0.0005
        self.a_q = 0.005
        self.a_v = 0.005
        self.BATCH_SIZE = 32
        self.replay_buffer = deque(maxlen=10000)
        self.update_frequency = 100
        self.state_space = None
        self.action_space = None
        self.state_mean = None
        self.state_std = None
        self.init_env_information()

        # init_weights
        self.y = torch.zeros([self.state_space, self.action_space])
        self.y.requires_grad = True
        self.q = torch.zeros([self.state_space, self.action_space])
        self.v = torch.zeros([self.state_space, self.action_space])

        # init network
        self.policy = DeterministicPolicy(self.y)
        self.Q = QFunctionDeterministic(self.q)
        self.V = ValueFunction(self.v)

    def init_env_information(self):
        """
        Initialize of environment information- state mean and state standard deviation.
        """
        env = gym.make('MountainCarContinuous-v0')
        self.state_space = env.observation_space.shape[0]
        self.action_space = env.action_space.shape[0]
        samples = np.array([env.observation_space.sample() for _ in range(100000)])
        self.state_mean = np.mean(samples, axis=0)
        self.state_std = np.std(samples, axis=0)
        env.close()

    def update_weights(self, batch):
        """
        Update weights given batch of samples
        Value function and state-action value function weights are updated by linear TD(0).
        Policy weights are updated by natural policy gradient.
        :param batch : array of samples used to update weights
        """

        for item in batch:
            state, action, new_state, reward, done = item

            # convert to Tensor
            state_tensor = torch.from_numpy(state).float().unsqueeze(1)
            new_state_tensor = torch.from_numpy(new_state).float().unsqueeze(1)

            # get action from policy
            new_action = self.policy(new_state_tensor)

            # td error of Q
            td_Q = reward + self.gama * self.Q(new_state_tensor, new_action, self) - self.Q(state_tensor, action, self)

            # calculate policy gradient
            self.policy.y.grad = None
            self.policy(state_tensor).backward()
            policy_gradient = self.policy.y.grad

            # get state-action feature
            state_action_feature = (action - self.policy(state_tensor)) * policy_gradient

            # calculate q update - using TD(0)
            q_update = td_Q.detach() * state_action_feature.detach()

            # calculate v update - using TD(0)
            v_update = td_Q * state_tensor

            # calculate policy update - using the natural gradient
            y_update = self.Q.q

            # update weights
            # we set requires grad flag to True
            self.policy.y = self.policy.y.detach() + self.a_y * y_update
            self.policy.y.requires_grad = True
            self.Q.q = self.Q.q.detach() + self.a_q * q_update
            self.V.v = self.V.v.detach() + self.a_v * v_update
