from DeterministicActorCritic import DeterministicActorCritic
from StochasticActorCritic import StochasticActorCritic
import Utillity as U


def main():
    print("Stochastic Actor-Critic initialization")
    stochastic_AC = StochasticActorCritic()
    print("Stochastic Actor-Critic learning....")
    stochastic_AC_scores, stochastic_steps = U.learn(stochastic_AC)
    print("Deterministic Actor Critic initialization")
    deterministic_AC = DeterministicActorCritic()
    print("Deterministic Actor-Critic learning....")
    deterministic_AC_scores, deterministic_steps = U.learn(deterministic_AC)
    U.graph(stochastic_AC_scores, deterministic_AC_scores, stochastic_steps)


if __name__ == '__main__':
    main()
