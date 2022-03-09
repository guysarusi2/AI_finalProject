import torch
import gym
import numpy as np
from collections import deque
from GaussianPolicy import GaussianPolicy
from ValueFunction import ValueFunction


class StochasticActorCritic:
    def __init__(self):
        self.gama = 0.99
        self.num_episodes = 900
        self.max_steps = 250
        self.a_y = 0.0005
        self.a_v = 0.005
        self.BATCH_SIZE = 512
        self.replay_buffer = deque(maxlen=10000)
        self.update_frequency = 5000
        self.state_space = None
        self.action_space = None
        self.state_mean = None
        self.state_std = None
        self.init_env_information()

        # init weight
        self.y = torch.zeros([self.state_space, self.action_space])
        self.v = torch.zeros([self.state_space, self.action_space])

        # init network
        self.policy = GaussianPolicy(self.y)
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
        Value function weights are updated by linear TD(0).
        Policy weights are updated by policy gradient.
        :param batch : array of samples used to update weights
        """

        for item in batch:
            state, action, new_state, reward, done = item

            # convert to Tensor
            state_tensor = torch.from_numpy(state).float().unsqueeze(1)
            new_state_tensor = torch.from_numpy(new_state).float().unsqueeze(1)

            # td error of V
            td_V = reward + self.gama * self.V(new_state_tensor) - self.V(state_tensor)

            # calculate policy update - using policy gradient
            y_update = td_V * self.policy.get_score_function(state_tensor, action)

            # calculate v update - using TD(0)
            v_update = td_V.detach() * state_tensor.detach()

            # update weights
            self.policy.y = self.policy.y.detach() + self.a_y * y_update
            self.V.v = self.V.v.detach() + self.a_v * v_update

            if abs(self.policy.y[0]) > 20 or abs(self.policy.y[1]) > 20:
                self.policy.y = self.policy.y / 20
