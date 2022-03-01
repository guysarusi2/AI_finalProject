import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

import gym
from tqdm import tqdm_notebook
import numpy as np
from collections import deque
import random
from copy import deepcopy


# class for the Q function
class QFunction():

    # takes in set of parameters
    def __init__(self, weight_vector):
        self.w = weight_vector

    # outputs an action-state value given a state and action
    def __call__(self, state, action):
        # action from policy μ
        a_pred = μ(state)

        # error between given action and the action from the policy μ
        δ = action - a_pred

        # zero the current gradient and take gradient of μ(s)
        μ.θ.grad = None
        a_pred.backward()

        # state-action feature
        ϕ_sa = μ.θ.grad * δ

        # advantage of taking action a instead of the policy's action
        Aʷ_sa = mult(ϕ_sa, self.w)

        # q value, advantage + value
        q_val = Aʷ_sa + V(state)

        return q_val


# class for the deterministic policy
class DeterministicPolicy():

    # initialize set of parameters
    def __init__(self, weight_vector):
        self.θ = weight_vector

    # gets action
    def __call__(self, state):
        # linear combination of parameters and state
        action = mult(self.θ, state)

        # tanh to constrain action to be in the range of [-1, 1]
        action = torch.tanh(action)

        return action


# class for value function
class ValueFunction():

    # initialize set of parameters
    def __init__(self, weight_vector):
        self.v = weight_vector

    # get value
    def __call__(self, state):
        # linear combination of parameters and state
        val = mult(self.v, state)

        return val


def mult(weight_vector, feature_vector):
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


def normalize_state(state, mean, std):
    ''' Normalizes state with given mean and standard deviation
    Args:
    - state (Array): current state
    - mean (float): normalizing mean
    - std (float): normalizing standard deviation
    '''

    # calculate normalized state
    normalized = (state - mean) / std

    return normalized


def update_parameters(batch, state_mean, state_std):
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
        state = normalize_state(state, state_mean, state_std)
        new_state = normalize_state(new_state, state_mean, state_std)

        # convert to Tensor
        state_tensor = torch.from_numpy(state).float().unsqueeze(1)
        new_state_tensor = torch.from_numpy(new_state).float().unsqueeze(1)

        # policy's action
        new_action = μ(new_state_tensor)

        # td error of Q value
        δ = reward + γ * Q(new_state_tensor, new_action) - Q(state_tensor, action)

        # calculate θ update
        # zero gradients and get gradient of μ(s)
        μ.θ.grad = None
        μ(state_tensor).backward()

        # get jacobian matrix
        jacob_matrix = μ.θ.grad

        # here we are using the natural gradient instead
        # θ_update = jacob_matrix * mult(jacob_matrix, Q.w)
        θ_update = Q.w

        # calculate w update
        # get state-action features
        ϕ_sa = (action - μ(state_tensor)) * jacob_matrix
        w_update = δ.detach() * ϕ_sa.detach()

        # calculate v update
        v_update = δ * jacob_matrix

        # update parameters
        # here we set requires_grad flag to True again since we used detach to create new leaf tensor
        μ.θ = μ.θ.detach() + α_θ * θ_update
        μ.θ.requires_grad = True

        Q.w = Q.w.detach() + αw * w_update
        V.v = V.v.detach() + αv * v_update


# set float precision point
torch.set_printoptions(precision=10)

# discount factor
γ = 0.99

# number of episodes to run
NUM_EPISODES = 1

# max steps per episode
MAX_STEPS = 5000

# score agent needs for environment to be solved
SOLVED_SCORE = 90

# learning rate for policy
α_θ = 0.005

# learning rate for value function
αv = 0.03

# learning rate for Q function
αw = 0.03

# batch size
BATCH_SIZE = 8
# Make environments
env = gym.make('MountainCarContinuous-v0').env
env2 = deepcopy(env)

# environment parameters
obs_space = env.observation_space.shape[0]
action_space = env.action_space.shape[0]

# set seeds
# set seeds
np.random.seed(0)
random.seed(0)
env.seed(0)
torch.manual_seed(0)

# Init weight vectors, should be matrices of dimensions (input, output)
stdv = 1 / np.sqrt(obs_space)
θ = torch.Tensor(np.random.uniform(low=-stdv, high=stdv, size=(obs_space, action_space)) * 0.03)
θ.requires_grad = True
w = torch.Tensor(np.random.uniform(low=-stdv, high=stdv, size=(obs_space, 1)) * 0.03)
v = torch.Tensor(np.random.uniform(low=-stdv, high=stdv, size=(obs_space, 1)) * 0.03)

# Init network
μ = DeterministicPolicy(θ)
Q = QFunction(w)
V = ValueFunction(v)

# samples = np.array(samples)
samples = np.array([env.observation_space.sample() for _ in range(10000)])
state_mean = np.mean(samples, axis=0)
state_std = np.std(samples, axis=0) + 1.0e-6
# training scores
scores = []

# policy scores
policy_scores = []

# count of total updates
total_updates = 0

# buffer used for experience replay
replay_buffer = deque(maxlen=8000)

# run episodes
for episode in (range(NUM_EPISODES)):

    # reset episode state and variables
    state = env.reset()
    state2 = env2.reset()

    score = 0
    score2 = 0

    done = False
    done2 = False

    # iterate through episode
    for step in range(MAX_STEPS):
        # sample random action from action space
        action = env.action_space.sample()[0]

        # step env
        new_state, reward, done, _ = env.step([action])

        # track score
        score += reward
        # calculate change in mechanical energy
        reward += 100 * ((np.sin(3 * new_state[0]) * 0.0025 + 0.5 * new_state[1] * new_state[1]) - (
                    np.sin(3 * state[0]) * 0.0025 + 0.5 * state[1] * state[1]))

        # push item into replay buffer
        item = [state, action, new_state, reward, done]
        replay_buffer.append(item)

        # every 10 steps, sample batch and update parameters
        if step % 10 == 0 and len(replay_buffer) > BATCH_SIZE:
            replay = random.sample(replay_buffer, BATCH_SIZE)
            update_parameters(replay, state_mean, state_std)
            total_updates += 1

        if done:
            break

        state = new_state

    # track episode score
    scores.append(score)

    # iterate through testing environment, here we test our policy at current episode
    for step in range(MAX_STEPS):
        # env2.render()

        # normalize state and get action from policy
        state2 = normalize_state(state2, state_mean, state_std)
        state_tensor2 = torch.from_numpy(state2).float().unsqueeze(1)
        action2 = μ(state_tensor2)

        # step env
        new_state2, reward2, done2, _ = env2.step([action2.item()])

        # track score
        score2 += reward2

        if done2:
            break

        state2 = new_state2

    # track score
    policy_scores.append(score2)
# HBox(children=(IntProgress(value=0, max=1), HTML(value='')))
import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression
# import seaborn as sns
import numpy as np

# sns.set()

plt.plot(scores, color='grey', label='Training score')
plt.plot(policy_scores, color='blue', label='Target Policy score')
plt.ylabel('score')
plt.xlabel('episodes')
plt.title('Score history of MountainCarContinuous with COPDAC-Q')
plt.legend()

# reg = LinearRegression().fit(np.arange(len(policy_scores)).reshape(-1, 1), np.array(policy_scores).reshape(-1, 1))
# y_pred = reg.predict(np.arange(len(policy_scores)).reshape(-1, 1))
# plt.plot(y_pred, color='orange')
plt.show()

testing_scores = []

for _ in (range(100)):
    state = env.reset()
    done = False
    score = 0
    for step in range(MAX_STEPS):
        env.render()
        state = normalize_state(state, state_mean, state_std)
        state_tensor = torch.from_numpy(state).float().unsqueeze(1)
        action = μ(state_tensor)
        new_state, reward, done, info = env.step([action.item()])

        score += reward

        state = new_state

        if done:
            break
    testing_scores.append(score)
env.close()
# HBox(children=(IntProgress(value=0), HTML(value='')))
np.array(testing_scores).mean()
98.7038103428186
np.array(testing_scores).var()
0.06571936450200667
env.close()
