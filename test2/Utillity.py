import random
import numpy as np
import torch
import gym


def learn(self):
    env = gym.make('MountainCarContinuous-v0')
    step = 0
    env2 = gym.make('MountainCarContinuous-v0')
    for episode in range(self.num_episodes):
        state = env.reset()
        # re = self.simulation(env2)
        # print(f'step {step} reward {re}')
        for i in range(self.max_steps):
            if step % 10000 == 0 and step != 0:
                # env2 = gym.make('MountainCarContinuous-v0')
                re = self.simulation(env2)
                # env2.close()
                print(f'step {step} reward {re}')

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
