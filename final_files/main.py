import gym
import time
from DeterministicActorCritic import DeterministicActorCritic
from StochasticActorCritic import StochasticActorCritic
import matplotlib.pyplot as plt
import numpy as np
import Utillity as U


def graph(a, b, c, d, e, i):
    plt.plot(i, a, color='green', label='first')
    plt.plot(i, b, color='blue', label='second')
    plt.plot(i, c, color='gray', label='third')
    plt.plot(i, d, color='orange', label='forth')
    plt.plot(i, e, color='pink', label='fifth')
    plt.ylabel('score')
    plt.xlabel('steps')
    plt.title('deterministic')
    plt.legend()
    plt.show()


def main():
    print("stoch  start")
    # stoc = StochasticActorCritic()
    # stoc_scores1, i3 = U.learn(stoc)

    det1 = DeterministicActorCritic()
    # det2 = DeterministicActorCritic()
    # det3 = DeterministicActorCritic()
    # det4 = DeterministicActorCritic()
    # det5 = DeterministicActorCritic()
    print("DET start")
    det_scores1, i1 = U.learn(det1)
    # U.graph(det_scores1, stoc_scores1, i3)
    # print("DET start")
    # det_scores2, i2 = U.learn(det2, is_deterministic=True)
    # print("DET start")
    # det_scores3, i3 = U.learn(det3, is_deterministic=True)
    # print("DET start")
    # det_scores4, i4 = U.learn(det4, is_deterministic=True)
    # print("DET start")
    # det_scores5, i5 = U.learn(det5, is_deterministic=True)
    # graph(det_scores1, det_scores2, det_scores3, det_scores4, det_scores5, i1)
    env = gym.make('MountainCarContinuous-v0')
    # print("simulation")
    # time.sleep(10)
    for i in range(500):
        U.simulation(det1, env, display=True)


if __name__ == '__main__':
    main()
