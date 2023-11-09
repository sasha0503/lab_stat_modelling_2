import random
import math

import matplotlib.pyplot as plt

"""
simulate a poisson process with rate lambda
"""


def poisson(lam):
    return -math.log(1 - random.random()) / lam


def plot_graph(L=0.5, N=50):
    """
    plot a graph of a poisson process with rate lambda
    :param L: lambda
    :param N: number of jumps
    :return: None
    """
    jumps = [0]
    y_values = [0]

    jumps2 = [0]

    plt.figure(figsize=(10, 4))
    for i in range(N):
        next_time = poisson(L) + (0 if i == 0 else jumps[-1])
        jumps.append(next_time)

        next_time2 = poisson(L) + (0 if i == 0 else jumps2[-1])
        jumps2.append(next_time2)

        y_values.append(i)
    plt.plot(jumps, y_values, drawstyle='steps-post')
    plt.plot(jumps2, y_values, drawstyle='steps-post')
    plt.xlabel('time')
    plt.text(0.7, 0.1, f"lambda = {L}", transform=plt.gca().transAxes)
    plt.xlim(0, max(jumps[-1], jumps2[-1]))
    plt.show()


def find_n_probability_formula(L, n, T):
    """
    find the probability of n jumps in time T with a formula
    :param L: lambda
    :param n: exact number of jumps
    :param T: time
    :return: probability of n jumps in time T
    """
    return (L * T) ** n * math.exp(-L * T) / math.factorial(n)


def find_n_probability_manual(L, n, T, M=1000):
    """
    find the probability of n jumps in time T manually
    :param L: lambda
    :param n: exact number of jumps
    :param T: time
    :param M: number of simulations
    :return: probability of n jumps in time T
    """
    counters = []
    for i in range(M):
        time = 0
        counter = 0
        while time < T:
            time += poisson(L)
            counter += 1
        counters.append(counter)

    return len([x for x in counters if x == n]) / M


if __name__ == '__main__':
    plot_graph()

    prob_formula = find_n_probability_formula(0.5, 100, 200)
    prob_manual = find_n_probability_manual(0.5, 100, 200)

    print(f"Probability of 100 jumps in 200 second with lambda = 0.5")
    print(f"Formula: {prob_formula}")
    print(f"Manual: {prob_manual}")
