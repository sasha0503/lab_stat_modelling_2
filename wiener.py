import math
import random

import matplotlib.pyplot as plt
import numpy as np

""""
model the path of a particle undergoing Brownian motion
"""

random.seed(0)


def wiener1(M, t):
    """
    simulate a path of a particle undergoing Brownian motion for time [0; 1]
    :param M: number of additions
    :param t: time
    :param n: number of time steps
    :return: a list of M+1 values of the path of a particle undergoing Brownian motion
    """
    eta_0 = random.gauss(0, 1)
    etas = [random.gauss(0, 1) for _ in range(M)]
    res = eta_0 * t + math.sqrt(2) * sum(
        [etas[i - 1] * math.sin(i * math.pi * t) / (i * math.pi) for i in range(1, M + 1)])
    return res


def wiener2(M, t):
    """
    simulate a path of a particle undergoing Brownian motion for time [0; 1]
    :param M: number of additions
    :param t: time
    :return: a list of M+1 values of the path of a particle undergoing Brownian motion
    """
    eta_0 = random.gauss(0, 1)
    etas_1 = [random.gauss(0, 1) for _ in range(M)]
    etas_2 = [random.gauss(0, 1) for _ in range(M)]
    res = eta_0 * t + math.sqrt(2) * sum([etas_1[i - 1] * math.sin(2 * i * math.pi * t) / (2 * i * math.pi) +
                                          etas_2[i - 1] * (1 - math.cos(2 * i * math.pi * t)) / (2 * i * math.pi)
                                          for i in range(1, M + 1)])
    return res


def variation(sequence_):
    return np.cumsum(abs(np.diff(sequence_, axis=0, prepend=0.)), axis=0)


if __name__ == '__main__':
    n = 100
    the_value = 1
    time = [x / n for x in range(n + 1)]  # time from 0 to 1 in 1000 steps

    # plot the graph
    plt.figure(figsize=(10, 4))
    samples = 100
    variations = []
    means = []
    reach_the_value = []
    for j in range(samples):
        print(f"{j} / {samples}")
        sequence = [wiener1(1000, t) for t in time]
        variations.append(variation(sequence))
        mean = sum(sequence) / len(sequence)
        means.append(mean)
        above_the_value = [x for x in sequence if x > the_value]
        time_to_reach_the_value = 1 if len(above_the_value) == 0 else sequence.index(above_the_value[0]) / n
        reach_the_value.append(time_to_reach_the_value)
        plt.plot(time, sequence)

    plt.xlabel('time')
    plt.ylabel('position')
    plt.show()

    # plot the quadratic variation
    plt.figure(figsize=(10, 4))
    for j in range(samples):
        plt.plot(time, variations[j])
    plt.xlabel('time')
    plt.ylabel('quadratic variation')
    plt.show()

    # plot the means
    plt.figure(figsize=(10, 4))
    plt.plot([x for x in range(samples)], means)
    plt.xlabel('sample')
    plt.ylabel('mean')
    plt.show()

    # plot the time to reach the value
    plt.figure(figsize=(10, 4))
    plt.plot([x for x in range(samples)], reach_the_value)
    # plot the mean time to reach the value
    plt.plot([x for x in range(samples)], [sum(reach_the_value) / len(reach_the_value) for _ in range(samples)])
    plt.xlabel('sample')
    plt.ylabel(f'time to reach the value a={the_value}')
    plt.show()
