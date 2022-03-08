import random
import numpy as np
import torch
import gym
import matplotlib.pyplot as plt


def learn(agent):
    """
    The main function of the agent learning process.
    Performs environment simulations and creating a replay buffer.
    Every predefined time steps update the weights using a batch sample from the replay buffer.
    :param agent: stochastic / deterministic actor-critic agent
    :return: scores- (list) contains agent average rewards in simulations
             time_steps- (list) The time steps where the scores were calculated.
    """
    scores_list = []
    time_steps = []
    current_step = 0
    env = gym.make('MountainCarContinuous-v0')
    env2 = gym.make('MountainCarContinuous-v0')
    for episode in range(agent.num_episodes):
        state = env.reset()
        for i in range(agent.max_steps):
            # evaluate the current policy with simulations
            if current_step % 5000 == 0 and current_step != 0:
                score = n_simulation(agent, env2, 30)
                scores_list.append(score)
                time_steps.append(current_step)

            # sample action from action space
            action = env.action_space.sample()[0]
            new_state, reward, done, _ = env.step([action])

            # normalize states
            normalized_state = normalize_state(agent, state)
            normalized_new_state = normalize_state(agent, new_state)

            # calculate reward with regard to the change in mechanical energy
            reward += 100 * ((np.sin(3 * normalized_new_state[0]) * 0.0025 + 0.5 * normalized_new_state[1] *
                              normalized_new_state[1]) - (
                                     np.sin(3 * normalized_state[0]) * 0.0025 + 0.5 * normalized_state[1] *
                                     normalized_state[
                                         1]))
            # push item into replay buffer
            item = [normalized_state, torch.tensor([action]), normalized_new_state, reward, done]
            agent.replay_buffer.append(item)

            # every fixed steps, sample batch and update weights
            if (current_step % agent.update_frequency) == 0 and len(agent.replay_buffer) > agent.BATCH_SIZE:
                replay = random.sample(agent.replay_buffer, agent.BATCH_SIZE)
                agent.update_weights(replay)

            if done:
                break
            current_step += 1
            state = new_state
    env.close()
    env2.close()
    return scores_list, time_steps


def simulation(agent, env, display=False):
    """
    Simulate environment episode with the policy of the given agent.
    :param agent: stochastic / deterministic actor-critic agent
    :param env: environment
    :param display: (boolean) determines if required displaying the environment state.
    :return: total (discounted) reward
    """
    total = 0
    current_state = env.reset()
    for i in range(agent.max_steps):
        if display:
            env.render()
        current_state = normalize_state(agent, current_state)
        state_tensor = torch.from_numpy(current_state).float().unsqueeze(1)
        action = agent.policy(state_tensor)
        next_state, reward, done, info = env.step([action.item()])
        current_state = next_state
        total += reward
        if done:
            break
    # env.close()
    return total


def n_simulation(agent, env, n):
    """
    Simulate n-times environment episodes with the policy of the given agent.
    :param agent: stochastic / deterministic actor-critic agent
    :param env: environment
    :param n: numer of episodes
    :return: average reward per episode
    """
    sum = 0
    for i in range(n):
        sum += simulation(agent, env, display=False)
    return sum / n


def normalize_state(agent, state):
    """
    Normalizes given state with agent predefined mean and standard deviation.
    :param agent: stochastic / deterministic actor-critic agent
    :param state : state to normalize
    """
    normalized = (state - agent.state_mean) / agent.state_std

    return normalized


def generate_state_action_feature(state, action):
    """
    Creates a state-action feature.
    :param state: current state (state=[s1,s2])
    :param action: current action
    :return: (Tensor) feature with the shape [s1,s2,action]
    """
    state_np = state.numpy()
    action_np = action.numpy()
    state_np_copy = np.copy(state_np)
    action_np_copy = np.copy(action_np)
    s1 = state_np_copy[0]
    s2 = state_np_copy[1]
    state_action_feature_np = np.vstack((s1, s2, action_np_copy))
    state_action_feature = torch.from_numpy(state_action_feature_np)
    return state_action_feature


def graph(stochastic_scores, deterministic_scores, steps):
    """
    Creates the graph with the scores off the agents.
    :param stochastic_scores: Stochastic agent scores.
    :param deterministic_scores: Deterministic agent scores.
    :param steps: The time steps where the scores were calculated.
    """
    plt.plot(steps, stochastic_scores, color='green', label='Stochastic AC')
    plt.plot(steps, deterministic_scores, color='blue', label='Deterministic AC')
    plt.ylabel('Average Reward Per Episode')
    plt.xlabel('Steps')
    plt.title('Comparison of stochastic AC and deterministic AC')
    plt.legend()
    plt.show()
