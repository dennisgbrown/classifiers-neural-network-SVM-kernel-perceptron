# -*- coding: utf-8 -*-
"""
Perceptron homework

Dennis Brown, COMP6636, 21 FEB 2021
"""

import numpy as np


def Perceptron(X, Y, beta, step_limit):
    """
    Perceptron. Given a 2-D set of data X (samples are rows, columns
    features), a vector Y of classifications, a learning rate (beta),
    and a step limit, train and return a weight vector that
    can be used to classify the given data.
    """

    # Initialize the weight vector and add entry for bias term
    w = np.zeros(len(X[0]) + 1)

    # Initialize Y_hat
    Y_hat = np.zeros(len(X))

    # Repeat the main loop until we have convergence or reach the
    # iteration limit
    steps = 0
    converged = False
    while(not(converged) and (steps < step_limit)):

        # For each sample in X, calculate w's classification error
        # and update w.
        for i in range(len(X)):
            # Add a 1 to the front of every term to account for w's bias
            sample = np.insert(X[i], 0, 1)
            Y_hat[i] = 1 if (np.matmul(w.T, sample) > 0) else -1
            error = Y[i] - Y_hat[i]
            w += sample * error * beta
            steps += 1

        # If the difference between Y ajd Y_hat is effectively 0,
        # consider it converged.
        if (np.linalg.norm(Y - Y_hat) < .0000001):
            converged = True

    print('Final w = ', w, 'in', steps, 'steps; converged?', converged)

    return w


def testPerceptron(X, Y, w):
    Y_hat = np.zeros(len(Y))
    for i in range(len(X)):
        sample = np.insert(X[i], 0, 1)
        Y_hat[i] = 1 if (np.matmul(w.T, sample) > 0) else -1
    print('Y   :', Y)
    print('Y^  :', Y_hat)
    print('Diff:', Y - Y_hat)


# Test it out

# # XOR does not converge
# X = np.array([[0, 0],
#               [0, 1],
#               [1, 0],
#               [1, 1]])
# Y = np.array([1, -1, -1, 1])

# Augmented XOR does converge
X = np.array([[0, 0, 1, 0, 0, 0],
              [0, 1, 0, 1, 0, 0],
              [1, 0, 0, 0, 1, 0],
              [1, 1, 0, 0, 0, 1]])
Y = np.array([1, -1, -1, 1])

w = Perceptron(X, Y, .01, 9999)
testPerceptron(X, Y, w)

