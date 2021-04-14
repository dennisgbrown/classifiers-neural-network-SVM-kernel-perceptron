# -*- coding: utf-8 -*-
"""
Perceptron-kernel homework

Dennis Brown, COMP6636, 28 MAR 2021
"""

import numpy as np


def poly_kernel(x, z, a, b, d):
    """
    Calculate polynomial kernel for samples x and z.
    a, b, and d are hyperparameters.
    """
    return (a + (b * (np.matmul(x.T, z))) ** d)


def train_perceptron_kernel(X, Y, beta, step_limit):
    """
    Perceptron with a kernel. Given a 2-D set of data X (samples are rows,
    columns features), a vector Y of classifications, a learning rate (beta),
    and a step limit, train and return a weight vector that
    can be used to classify the given data.
    """

    # Initialize the alpha vector
    a = np.zeros(len(X))

    # Initialize Y_hat
    Y_hat = np.zeros(len(X))

    # Repeat the main loop until we have convergence or reach the
    # iteration limit
    steps = 0
    converged = False
    while(not(converged) and (steps < step_limit)):
        converged = True # assume converged until we determine otherwise

        # For each sample in X, calculate alpha's classification error
        # and update alpha.
        for i in range(len(X)):

            # Find current prediction based on kernel
            predict_val = 0
            for j in range(len(X)):
                predict_val += a[i] * poly_kernel(X[j], X[i], 0.0, 1.0, 1.0)
            Y_hat[i] = 1 if (predict_val > 0) else -1

            # If error on this element is > a very small value (is not
            # effectively 0), we need to update alpha, and have not converged.
            error = Y[i] - Y_hat[i]
            if (abs(error) > 0.000001):
                a[i] += beta * Y[i]
                converged = False
            steps += 1

    print('Final alpha = ', a, 'in', steps, 'steps; converged?', converged)

    return a


def test_perceptron_kernel(X, Y, a):
    Y_hat = np.zeros(len(Y))
    for i in range(len(X)):
        predict_val = 0
        for j in range(len(X)):
            predict_val += a[i] * poly_kernel(X[j], X[i], 0.0, 1.0, 1.0)
        Y_hat[i] = 1 if (predict_val > 0) else -1
    print('Y   :', Y)
    print('Y^  :', Y_hat)
    print('Diff:', Y - Y_hat)


# Test it out

# # XOR still does not converge
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

a = train_perceptron_kernel(X, Y, .01, 9999)
test_perceptron_kernel(X, Y, a)

