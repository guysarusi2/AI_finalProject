from DeterministicActorCritic import DeterministicActorCritic
from StochasticActorCritic import StochasticActorCritic
from StochasticActorCritic2 import StochasticActorCritic2
from StochasticActorCritic3 import StochasticActorCritic3
from StochasticActorCritic4 import StochasticActorCritic4
import matplotlib.pyplot as plt
import numpy as np


def graph(a, b,  i):
    plt.plot(i, a, color='green', label='first')
    plt.plot(i, b, color='blue', label='second')
    # plt.plot(i, c, color='red', label='third')
    # plt.plot(i, d, color='green', label='forth')
    plt.ylabel('score')
    plt.xlabel('steps')
    plt.title('Score without poly weights')
    plt.legend()
    plt.show()


def main():
    # graph([1, 2, 3], [2, 4, 6],[10,20,30])
    # stoc1 = StochasticActorCritic()
    # stoc2 = StochasticActorCritic2()
    stoc3 = StochasticActorCritic3()
    det = DeterministicActorCritic()
    # stoc4 = StochasticActorCritic4()
    # print("stoch 1 start")
    # stoc_scores1, i1 = stoc1.learn()
    # print("stoch 2 start")
    # stoc_scores2, i2 = stoc2.learn()
    print("stoch 3 start")
    stoc_scores3, i3 = stoc3.learn()
    # print("stoch 4 start")
    # stoc_scores4, i4 = stoc4.learn()
    print("DET start")
    det_scores, i1 = det.learn()
    graph(stoc_scores3, det_scores, i3)


if __name__ == '__main__':
    main()
