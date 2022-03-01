import random

import gama as gama
import torch
import gym
import numpy as np
from collections import deque
import DeterministicPolicy
from GaussianPolicy import GaussianPolicy
from ValueFunction import ValueFunction
import QFunctionDeterministic
from QFunctionStochastic import QFunctionStochastic


class StochasticActorCritic:
    def __init__(self):
        self.gama = 0.99
        self.num_episodes = 2000
        self.max_steps = 250
        self.a_y = 0.0005
        self.a_q = 0.005
        self.a_v = 0.005
        self.BATCH_SIZE = 8
        self.replay_buffer = deque(maxlen=8000)
        # self.y = None
        # self.w = None
        # self.v = None
        # self.policy = None
        # self.Q = None
        # self.V = None
        self.state_space = None
        self.action_space = None
        self.state_mean = None
        self.state_std = None
        self.init_env_information()
        # init weight
        self.y = torch.zeros([self.state_space, self.action_space])
        self.q = torch.zeros([self.state_space + 1, self.action_space])
        self.v = torch.zeros([self.state_space, self.action_space])

        # init network
        self.policy = GaussianPolicy(self.y)
        self.Q = QFunctionStochastic(self.q)
        self.V = ValueFunction(self.v)

    def init_env_information(self):
        env = gym.make('MountainCarContinuous-v0')
        self.state_space = env.observation_space.shape[0]
        self.action_space = env.action_space.shape[0]
        samples = np.array([env.observation_space.sample() for _ in range(100000)])
        self.state_mean = np.mean(samples, axis=0)
        self.state_std = np.std(samples, axis=0)
        env.close()

    def update_parameters(self, batch):
        ''' Update parameters given batch of samples
        Args:
        - batch (Array): array of samples used to update parameters
        - state_mean (float): normalizing mean
        - state_std (float): normalizing standard deviation
        '''

        # iterate through batch
        for item in batch:
            # decompose the item
            state, action, new_state, reward, done = item

            # normalize states
            state = self.normalize_state(state)
            new_state = self.normalize_state(new_state)

            # convert to Tensor
            state_tensor = torch.from_numpy(state).float().unsqueeze(1)
            new_state_tensor = torch.from_numpy(new_state).float().unsqueeze(1)

            # policy's action
            # new_action = μ(new_state_tensor)
            new_action = self.policy.choose_action(new_state_tensor)

            # td error of Q value
            # δ = reward + γ * Q(new_state_tensor, new_action) - Q(state_tensor, action)
            # td_V = reward + γ * V(new_state_tensor) - V(state_tensor)
            td_V = reward + self.gama * self.V(new_state_tensor) - self.V(state_tensor)
            td_Q = reward + self.gama * self.Q(new_state_tensor, new_action) - self.Q(state_tensor, action)

            # calculate θ update
            # zero gradients and get gradient of μ(s)
            # μ.θ.grad = None
            # μ(action, state_tensor).backward()
            # θ_update = td_V * μ.θ.grad
            # θ_update = td_V * μ.get_grad(state_tensor, action)

            y_update = td_V * self.policy.get_grad(state_tensor, action)

            A_s = self.Q(state_tensor, action) - self.V(state_tensor)
            A_s_next = self.Q(new_state_tensor, new_action) - self.V(new_state_tensor)
            td_A = reward + self.gama * A_s_next - A_s
            # y_update = td_V * self.policy.get_grad(state_tensor, action)
            # y_update = td_A * self.policy.get_grad(state_tensor, action)

            # get jacobian matrix
            # jacob_matrix = μ.θ.grad
            # print(jacob_matrix)
            # log_func = torch.log(μ.θ)
            # print(log_func)
            # jacob_matrix = log_func.backward()

            # here we are using the natural gradient instead
            # θ_update = jacob_matrix * mult(jacob_matrix, Q.w)
            # θ_update = Q.w
            # θ_update = jacob_matrix * Q(state_tensor, action)

            # calculate w update
            # get state-action features
            # ϕ_sa = (action - μ.choose_action(state_tensor)) * jacob_matrix
            a = torch.Tensor([[action.item()]])
            ϕ_sa = torch.cat((state_tensor, a), 0)
            q_update = td_V.detach() * ϕ_sa.detach()
            v_update = td_V.detach() * state_tensor.detach()
            # q_update = td_Q.detach() * ϕ_sa.detach()

            # print(f'policy update {y_update}')
            # print(f'Q update {q_update}')
            # print(f'V update {v_update}')
            # calculate v update
            # v_update = δ * jacob_matrix

            # update parameters
            # here we set requires_grad flag to True again since we used detach to create new leaf tensor
            # μ.θ = μ.θ.detach() + α_θ * θ_update
            
            if abs(self.policy.y[0])>20 or abs(self.policy.y[1]) > 20 :
                pass
            else :
                self.policy.y = self.policy.y.detach() + self.a_y * y_update
            
            # μ.θ.requires_grad = True

            # Q.w = Q.w.detach() + αw * w_update
            # V.v = V.v.detach() + αv * v_update

            self.Q.q = self.Q.q.detach() + self.a_q * q_update
            self.V.v = self.V.v.detach() + self.a_v * v_update

    def learn(self):
        env = gym.make('MountainCarContinuous-v0')
        step = 0
        env2 = gym.make('MountainCarContinuous-v0')
        for episode in range(self.num_episodes):
            state = env.reset()
            # re = self.simulation(env2)
            # print(f'step {step} reward {re}')
            for i in range(self.max_steps):
                if step % 5000 == 0 and step != 0:
                    # env2 = gym.make('MountainCarContinuous-v0')
                    re = self.simulation(env2)
                    # env2.close()
                    print(f'step {step} reward {re}')
                    print(f'weight {self.policy.y}')

                # sample random action from action space
                action = env.action_space.sample()[0]
                # print(action)
                # env.render()
                # step env
                new_state, reward, done, _ = env.step([action])

                # track score
                # score += reward
                # calculate change in mechanical energy
                normal_state = self.normalize_state(state)
                normal_new_state = self.normalize_state(new_state)
                reward += 100 * ((np.sin(3 * normal_new_state[0]) * 0.0025 + 0.5 * normal_new_state[1] *
                                  normal_new_state[1]) - (
                                         np.sin(3 * normal_state[0]) * 0.0025 + 0.5 * normal_state[1] * normal_state[
                                     1]))
                # reward += 100 * ((np.sin(3 * new_state[0]) * 0.0025 + 0.5 * new_state[1] * new_state[1]) - (
                #         np.sin(3 * state[0]) * 0.0025 + 0.5 * state[1] * state[1]))

                # push item into replay buffer
                item = [state, action, new_state, reward, done]
                self.replay_buffer.append(item)

                # every 10 steps, sample batch and update parameters
                if i % 10 == 0 and len(self.replay_buffer) > self.BATCH_SIZE:
                    # print(step)
                    replay = random.sample(self.replay_buffer, self.BATCH_SIZE)
                    # print(replay)
                    self.update_parameters(replay)
                    # total_updates += 1

                if done:
                    break
                step += 1
                state = new_state
        env.close()
        env2.close()

    def simulation(self, env):
        total = 0
        current_state = env.reset()
        for i in range(self.max_steps):
            env.render()
            current_state = self.normalize_state(current_state)
            state_tensor = torch.from_numpy(current_state).float().unsqueeze(1)
            action = self.policy.choose_action(state_tensor)
            next_state, reward, done, info = env.step([action.item()])
            current_state = next_state
            total += reward
            if done:
                break
        # env.close()
        return total

    def mult(self, weight_vector, feature_vector):
        ''' Mulitplies weight vector by feature vector
        Args:
        - weight_vector (Tensor): vector of weights
        - feature_vector (Tensor): vector of features

        Return:
        - product (Tensor): product of vectors

        '''

        # Transpose weight vector and multiply by feature vector
        product = torch.matmul(torch.transpose(weight_vector, 0, 1), feature_vector)

        # Return product
        return product

    def normalize_state(self, state):
        ''' Normalizes state with given mean and standard deviation
        Args:
        - state (Array): current state
        - mean (float): normalizing mean
        - std (float): normalizing standard deviation
        '''

        # calculate normalized state
        normalized = (state - self.state_mean) / self.state_std

        return normalized


def main():
    ac = StochasticActorCritic()
    ac.learn()


if __name__ == "__main__":
    main()
