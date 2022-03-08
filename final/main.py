from DeterministicActorCritic import DeterministicActorCritic
from StochasticActorCritic import StochasticActorCritic
import matplotlib.pyplot as plt


def graph(a, b, i):
    plt.plot(i,a, color='grey', label='Deterministic score')
    plt.plot(i,b, color='blue', label='Stochastic score')
    plt.ylabel('score')
    plt.xlabel('steps')
    plt.title('Score without poly weights')
    plt.legend()
    plt.show()


def main():
    # graph([1, 2, 3], [2, 4, 6],[10,20,30])
    det = DeterministicActorCritic()
    stoc = StochasticActorCritic()
    print("stochastic start")
    stoc_scores, i2 = stoc.learn()
    print("deterministic start")
    det_scores, i1 = det.learn()
    graph(det_scores, stoc_scores, i1)

if __name__ == '__main__':
    main()
