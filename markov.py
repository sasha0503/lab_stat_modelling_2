import random
import numpy as np

import matplotlib.pyplot as plt

"""
simulate markov chains
"""


def create_matrix(n):
    """
    create a random matrix of size n x n
    :param n: size of the matrix
    :return: a random transition matrix of size n x n
    """
    matrix = np.random.rand(n, n)
    matrix = matrix / np.sum(matrix, axis=0)
    return matrix


def step(matrix_, state):
    """
    simulate a step of a markov chain by multiplying the current state by the matrix
    :param matrix_: transition matrix
    :param state: current state
    :return: next state
    """
    return np.matmul(matrix_, state)


def find_formula_solution(matrix_):
    """
    solve xM = x such that the sum of elements in x is 1
    """
    eigenvalues, eigenvectors = np.linalg.eig(matrix_)
    selected_eigenvector = eigenvectors[:, 0]
    normalized_eigenvector = selected_eigenvector / np.sum(selected_eigenvector)
    constant = 1 / np.sum(normalized_eigenvector)

    return constant * normalized_eigenvector


if __name__ == '__main__':
    num_steps = 100
    num_repetitions = 1000
    num_states_list = [3, 5, 10, 20, 50]
    break_points = []

    for state_idx, num_states in enumerate(num_states_list):
        print(f"{state_idx} / {len(num_states_list)}")
        break_points.append([])
        for _ in range(num_repetitions):
            prob_matrix = create_matrix(num_states)
            solution = find_formula_solution(prob_matrix)

            state = [0 for _ in range(num_states)]
            random_idx = random.randint(0, num_states - 1)
            state[random_idx] = 1

            transitions = [state]
            for i in range(num_steps):
                state = step(prob_matrix, state)
                transitions.append(state)
                if np.allclose(state, solution):
                    break_points[state_idx].append(i)
                    break

    plt.figure(figsize=(10, 4))
    plt.boxplot(break_points)
    break_points_mean = [np.mean(x) for x in break_points]
    print(break_points_mean)
    plt.xticks(range(1, len(num_states_list) + 1), num_states_list)
    plt.xlabel('number of states')
    plt.ylabel('number of steps')
    plt.show()
