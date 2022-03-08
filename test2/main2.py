from DeterministicActorCritic import DeterministicActorCritic
from StochasticActorCritic import StochasticActorCritic
from StochasticActorCritic2 import StochasticActorCritic2
from StochasticActorCritic3 import StochasticActorCritic3
from StochasticActorCritic4 import StochasticActorCritic4
import matplotlib.pyplot as plt
import numpy as np


def graph(a, b,c,d , i):
    plt.plot(i, a, color='green', label='first')
    plt.plot(i, b, color='blue', label='second')
    plt.plot(i, c, color='red', label='third')
    plt.plot(i, d, color='gray', label='forth')
    plt.ylabel('score')
    plt.xlabel('steps')
    plt.title('4 stochastic')
    plt.legend()
    plt.show()


def main():
    # stoc3 = StochasticActorCritic3()
    det1 = StochasticActorCritic3()
    det2 = StochasticActorCritic3()
    det3 = StochasticActorCritic3()
    det4 = StochasticActorCritic3()
    print("det 1")
    det_scores1, i1 = det1.learn()
    print("det 2")
    det_scores2, i2 = det2.learn()
    print("det 3")
    det_scores3, i3 = det3.learn()
    print("det 4")
    det_scores4, i4 = det4.learn()

    # stoc4 = StochasticActorCritic4()
    # print("stoch 1 start")
    # stoc_scores1, i1 = stoc1.learn()
    # print("stoch 2 start")
    # stoc_scores2, i2 = stoc2.learn()
    # print("stoch 3 start")
    # stoc_scores3, i3 = stoc3.learn()
    # print("stoch 4 start")
    # stoc_scores4, i4 = stoc4.learn()
    # print("DET start")
    # det_scores, i1 = det.learn()
    graph(det_scores1,det_scores2,det_scores3,det_scores4, i1)


if __name__ == '__main__':
    main()
