from DeterministicActorCritic import DeterministicActorCritic
from StochasticActorCritic3 import StochasticActorCritic3
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
    stoc3 = StochasticActorCritic3()
    det = DeterministicActorCritic()
    print("stoch 3 start")
    stoc_scores3, i3 = stoc3.learn()
    print("DET start")
    det_scores, i1 = det.learn()
    graph(stoc_scores3, det_scores, i3)


if __name__ == '__main__':
    main()
